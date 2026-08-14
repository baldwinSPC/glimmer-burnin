package main

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"time"
)

// aggregation says how a field's samples become one number.
type aggregation int

const (
	// aggMax reports the largest value seen across the whole test window. The
	// registry defines gpuTempC as the PEAK temperature observed during the
	// test, so a single reading taken after the diagnostic finished — when the
	// part has already begun to cool — would be a different quantity wearing
	// the same name.
	aggMax aggregation = iota
	// aggDelta reports how much a lifetime counter moved during the test.
	// DCGM's aggregate counters are lifetime totals; reporting them raw would
	// mean a part that took one ECC error two years ago fails every burn-in it
	// is ever given, and a threshold of "0" would be untunable.
	//
	// ONLY FOR FIELDS DCGM DOCUMENTS AS MONOTONIC COUNTERS. dcgm_fields.h says
	// so explicitly for the ones that qualify ("Note: monotonically
	// increasing"); a field that merely holds a small integer does not.
	// Subtracting two readings of something that is not a counter produces a
	// plausible small number that is not a count of anything — see aggLast.
	aggDelta
	// aggLast reports the final reading, for a field that holds a STATE rather
	// than a count. No arithmetic is done on it, because none is meaningful.
	aggLast
)

// dcgmField is one DCGM field this runner samples.
//
// sym is the symbol in DCGM's own dcgm_fields.h. It is not decoration: the
// Dockerfile runs this binary with --print-field-ids against the DCGM source it
// just built and FAILS THE BUILD if any id here disagrees with the header. A
// wrong field id does not produce an error at runtime — it silently reports a
// different measurement under our name, which is the one failure mode nothing
// downstream could ever detect.
type dcgmField struct {
	sym string
	id  int
	key string
	agg aggregation
}

var sampledFields = []dcgmField{
	{sym: "DCGM_FI_DEV_GPU_TEMP", id: 150, key: keyGPUTempC, agg: aggMax},
	// FIELD 230 IS NOT A COUNTER, and it was read as one until #311.
	// dcgm_fields.h describes it as "XID errors. The value is the specific XID
	// error" — the CODE of the last XID encountered. Compare the two ECC
	// entries below, which dcgm_fields.h marks "Note: monotonically
	// increasing"; that note is the licence to take a delta, and 230 does not
	// carry it.
	//
	// Subtracting two codes produced neither a count nor a code. A GPU sitting
	// at XID 79 for the whole window reported 79-79 = 0 and CERTIFIED CLEAN,
	// which is a false pass on a faulting part; a GPU that went 13 -> 79
	// reported "66 Xid errors". The shipped sample gated `xidEvents Equal 0` on
	// exactly this number.
	//
	// This is the same substitution the 393/394 note below describes — a
	// plausible small integer, thresholdable, and wrong — reached by a
	// different route. The build's field-id assertion cannot catch it: the id
	// is right and the SEMANTICS are wrong, so the check that matters is
	// reading the field's description before choosing an aggregation.
	//
	// It is reported as evidence now, under its own name, and `xidEvents` comes
	// from host-health's /dev/kmsg path, which counts events rather than
	// subtracting codes.
	{sym: "DCGM_FI_DEV_XID_ERRORS", id: 230, key: keyLastXidCode, agg: aggLast},
	{sym: "DCGM_FI_DEV_PCIE_REPLAY_COUNTER", id: 202, key: keyPCIeReplayCount, agg: aggDelta},
	{sym: "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL", id: 312, key: keyECCSbeTotal, agg: aggDelta},
	{sym: "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL", id: 313, key: keyECCDbeTotal, agg: aggDelta},
	// Correctable and uncorrectable remapped rows are two fields and one
	// quantity: the registry's remappedRows is "rows the driver has remapped",
	// and both kinds are remapped. Summing them here is deliberate, and is not
	// the same as summing SBE and DBE — those two say different things about
	// the part's health and are kept apart on purpose.
	//
	// These two were 394/395 until the build's field-id assertion was actually
	// run against dcgm_fields.h: 394 is the CORRECTABLE count and 395 is
	// DCGM_FI_DEV_ROW_REMAP_FAILURE, a flag rather than a count. The old pair
	// therefore summed the correctable rows with a boolean and reported the
	// total as remappedRows — a plausible small integer, thresholdable, and
	// wrong, which is precisely the substitution the assertion exists to catch.
	{sym: "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS", id: 393, key: keyRowsRemapped, agg: aggDelta},
	{sym: "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS", id: 394, key: keyRowsRemapped, agg: aggDelta},
}

// printFieldIDs writes the symbol=id pairs the Dockerfile asserts against
// dcgm_fields.h. The format is deliberately trivial to parse from shell.
func printFieldIDs(w io.Writer) {
	for _, f := range sampledFields {
		fmt.Fprintf(w, "%s=%d\n", f.sym, f.id)
	}
}

type seriesKey struct {
	field int
	gpu   string
}

type series struct {
	first, last, max float64
	n                int
}

func (s *series) observe(v float64) {
	if s.n == 0 || v > s.max {
		s.max = v
	}
	if s.n == 0 {
		s.first = v
	}
	s.last = v
	s.n++
}

// sampler polls DCGM field values alongside the diagnostic.
//
// It shells out to `dcgmi dmon -c 1` on a ticker rather than streaming a
// long-lived `dcgmi dmon`. A streamed child would be one process instead of a
// few hundred, but its stdout is block-buffered into a pipe, and this runner
// kills it the moment the diagnostic ends — so the samples that matter most,
// the ones taken while the part was hottest, are exactly the ones still sitting
// in an unflushed buffer. A short-lived process flushes when it exits.
type sampler struct {
	cfg config

	mu      sync.Mutex
	state   map[seriesKey]*series
	gpus    map[string]struct{}
	samples int
	// mismatched counts dmon lines whose column count did not match the fields
	// requested. Such a line is dropped rather than parsed positionally: a
	// column shift would report the PCIe replay counter as a temperature, and
	// both are plausible-looking numbers.
	mismatched int

	stop chan struct{}
	done chan struct{}
}

func newSampler(cfg config) *sampler {
	return &sampler{
		cfg:   cfg,
		state: map[seriesKey]*series{},
		gpus:  map[string]struct{}{},
		stop:  make(chan struct{}),
		done:  make(chan struct{}),
	}
}

func (s *sampler) fieldIDs() string {
	ids := make([]string, 0, len(sampledFields))
	for _, f := range sampledFields {
		ids = append(ids, strconv.Itoa(f.id))
	}
	return strings.Join(ids, ",")
}

// sampleOnce takes one reading. Errors are logged and swallowed: field sampling
// is evidence gathering, and a node must not be reported as unjudged because a
// side-channel poll failed while the diagnostic itself ran fine.
func (s *sampler) sampleOnce(ctx context.Context) {
	stdout, stderr, code, err := runCmd(ctx, 20*time.Second, s.cfg.dcgmiBin,
		dcgmiArgs("dmon", s.cfg.address, "-e", s.fieldIDs(), "-c", "1")...)
	if err != nil || code != 0 {
		logf("field sample failed (exit %d): %v %s", code, err, firstLine(stderr))
		return
	}
	s.observeAll(stdout)
}

func (s *sampler) observeAll(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rows := 0
	for _, line := range strings.Split(text, "\n") {
		gpu, values, ok := parseDmonSample(line, len(sampledFields))
		if !ok {
			if isDmonDataLine(line) {
				s.mismatched++
			}
			continue
		}
		rows++
		s.gpus[gpu] = struct{}{}
		for i, v := range values {
			if !v.ok {
				continue
			}
			k := seriesKey{field: i, gpu: gpu}
			ser := s.state[k]
			if ser == nil {
				ser = &series{}
				s.state[k] = ser
			}
			ser.observe(v.value)
		}
	}
	if rows > 0 {
		s.samples++
	}
}

// run polls until stopped.
func (s *sampler) run(ctx context.Context) {
	go func() {
		defer close(s.done)
		t := time.NewTicker(s.cfg.sampleInterval)
		defer t.Stop()
		for {
			select {
			case <-s.stop:
				return
			case <-ctx.Done():
				return
			case <-t.C:
				s.sampleOnce(ctx)
			}
		}
	}()
}

func (s *sampler) shutdown() {
	close(s.stop)
	<-s.done
}

type sampleValue struct {
	value float64
	ok    bool
}

// isDmonDataLine reports whether a line looks like it was meant to carry
// samples. Used only to count lines we refused to parse.
func isDmonDataLine(line string) bool {
	f := strings.Fields(line)
	return len(f) >= 2 && strings.EqualFold(f[0], "GPU")
}

// parseDmonSample reads one `dcgmi dmon` data row.
//
// dmon prints a two-line header (both starting with "#", or indented) and then
// one row per entity: "GPU 0   45   0   0 …", with the value columns in exactly
// the order the fields were requested in. The column count must match what we
// asked for or the row is refused — see sampler.mismatched.
func parseDmonSample(line string, want int) (string, []sampleValue, bool) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return "", nil, false
	}
	f := strings.Fields(line)
	if len(f) != want+2 || !strings.EqualFold(f[0], "GPU") {
		return "", nil, false
	}
	if _, err := strconv.Atoi(f[1]); err != nil {
		return "", nil, false
	}
	values := make([]sampleValue, want)
	for i, raw := range f[2:] {
		v, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			// "N/A" is DCGM saying the part does not expose this field. That is
			// a missing measurement, and a missing measurement is never a zero.
			continue
		}
		values[i] = sampleValue{value: v, ok: true}
	}
	return f[1], values, true
}

// report writes the sampled fields into the run's metrics.
//
// A metric is emitted only when it was actually measured. Nothing here defaults
// to zero: verdict evaluation fails closed on a threshold whose metric is
// absent, so an unmeasured counter printed as "0" would convert a fails-closed
// FAILURE into a silent pass — the exact substitution the project forbids.
func (s *sampler) report(rep *report) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.gpus) > 0 {
		rep.setInt(keyGPUCount, int64(len(s.gpus)))
	}

	byKey := map[string][]int{}
	var order []string
	for i, f := range sampledFields {
		if _, seen := byKey[f.key]; !seen {
			order = append(order, f.key)
		}
		byKey[f.key] = append(byKey[f.key], i)
	}

	reset := false
	for _, key := range order {
		idx := byKey[key]
		agg := sampledFields[idx[0]].agg

		switch agg {
		case aggMax:
			best, found := 0.0, false
			for _, i := range idx {
				for gpu := range s.gpus {
					if ser := s.state[seriesKey{field: i, gpu: gpu}]; ser != nil && ser.n > 0 {
						if !found || ser.max > best {
							best, found = ser.max, true
						}
					}
				}
			}
			if found {
				rep.setNumber(key, best)
			}
		case aggDelta:
			// A delta needs two readings to exist. With one, the honest answer
			// is that the window was never measured, not that nothing happened
			// in it.
			if s.samples < 2 {
				continue
			}
			total, found, wentBackwards := 0.0, false, false
			for _, i := range idx {
				for gpu := range s.gpus {
					ser := s.state[seriesKey{field: i, gpu: gpu}]
					if ser == nil || ser.n < 2 {
						continue
					}
					found = true
					if d := ser.last - ser.first; d < 0 {
						// A lifetime counter that went backwards means the
						// driver was reloaded or the device reset mid-test.
						wentBackwards = true
					} else {
						total += d
					}
				}
			}
			switch {
			case !found:
			case wentBackwards:
				// UNMEASURABLE, not zero.
				//
				// This used to clamp the negative movement to 0 and publish the
				// sum anyway, reasoning that 0 was "the movement we can defend
				// (none)". It is not: 0 is the assertion that NOTHING HAPPENED,
				// pkg/runner stores it in Result.Metrics, and `eccErrors Equal
				// 0` is satisfied by it.
				//
				// The condition is the one under which we know least. A counter
				// going backwards means the driver reloaded or the device
				// reset, which is exactly when ECC and Xid counters are most
				// likely to have recorded something before being zeroed — so
				// the old behaviour reported "clean" for precisely the window
				// whose evidence was destroyed (#312).
				//
				// `n/a` is this project's reserved word for "we looked and
				// there is nothing we can report". pkg/runner puts it in
				// Result.Unmeasurable rather than Metrics, so a Required
				// threshold fails closed and RequiredIfMeasurable reports NOT
				// EVALUATED. Both are honest; a fabricated 0 is neither.
				//
				// The whole key goes unmeasurable, not just the affected GPU: a
				// sum missing one of its terms is not a smaller sum, it is an
				// unknown one.
				reset = true
				rep.set(key, metricUnmeasurable)
			default:
				rep.setNumber(key, total)
			}
		case aggLast:
			// The node-level value of a per-GPU STATE. The largest final
			// reading across the GPUs, which for an XID code is not an
			// arithmetic claim — it is "some GPU ended the window reporting
			// this code, and none reported a higher one". What matters for a
			// gate is that it is non-zero exactly when some GPU reports an
			// error, and it is deterministic on identical input.
			best, found := 0.0, false
			for _, i := range idx {
				for gpu := range s.gpus {
					if ser := s.state[seriesKey{field: i, gpu: gpu}]; ser != nil && ser.n > 0 {
						if !found || ser.last > best {
							best, found = ser.last, true
						}
					}
				}
			}
			if found {
				rep.setNumber(key, best)
			}
		}
	}

	rep.setInt(keySampleCount, int64(s.samples))
	if reset {
		rep.set(keyCounterReset, "true")
	}
	if s.mismatched > 0 {
		logf("dropped %d dcgmi dmon rows whose column count did not match the requested fields", s.mismatched)
	}
}
