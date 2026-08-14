// Command dcgm-diag is the runner for the "dcgm-diag" TestKind.
//
// It wraps NVIDIA DCGM's diagnostic suite (`dcgmi diag -r <level> -j`) and
// reports the result in the burn-in runner contract: key=value metrics on
// stdout, then one marker line, then an exit code.
//
//	0  the diagnostic ran and every subtest passed
//	1  the diagnostic ran and a subtest failed — a verdict about the hardware
//	2  the diagnostic does not apply to this hardware: DCGM does not support
//	   the part. Expected on GB10, and the reason exit 2 exists at all
//	3  anything else — the runner, the host engine or DCGM malfunctioned, and
//	   the hardware is UNJUDGED
//
// Exit 3 is never used for a hardware judgement and exit 1 is never used for a
// runner problem. "We could not tell" and "the part is bad" lead to different
// actions by whoever reads the verdict, and a runner that blurs them makes the
// distinction unrecoverable downstream.
//
// Every metric is printed BEFORE the marker on every path, so a failing,
// skipping or erroring run still yields its evidence. Prose goes to stderr;
// stdout carries only key=value lines and the final marker.
package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	exitPass  = 0
	exitFail  = 1
	exitSkip  = 2
	exitError = 3
)

const (
	markerPass  = "DCGM_DIAG_PASS"
	markerFail  = "DCGM_DIAG_FAIL"
	markerSkip  = "DCGM_DIAG_SKIP"
	markerError = "DCGM_DIAG_ERROR"
)

// prunedManifestPath lists the DCGM objects the image build removed because
// they needed an NVIDIA redistributable library this project may not ship. See
// the Dockerfile. An absent file means nothing was pruned.
const prunedManifestPath = "/usr/share/glimmer-burnin/dcgm-pruned.txt"

func main() {
	// The Dockerfile runs this against the DCGM headers it just built, and
	// fails the build if a field id here disagrees with dcgm_fields.h.
	if len(os.Args) > 1 && os.Args[1] == "--print-field-ids" {
		printFieldIDs(os.Stdout)
		return
	}
	os.Exit(run())
}

type config struct {
	dcgmiBin       string
	hostEngineBin  string
	hostEngineArgs []string
	address        string
	startEngine    bool
	duration       time.Duration
	level          string
	sampleInterval time.Duration
	engineTimeout  time.Duration
	prunedManifest string
}

func loadConfig() (config, error) {
	c := config{
		dcgmiBin:       envOr("BURNIN_DCGM_BIN", "dcgmi"),
		hostEngineBin:  envOr("BURNIN_NV_HOSTENGINE_BIN", "nv-hostengine"),
		level:          os.Getenv("BURNIN_DCGM_LEVEL"),
		sampleInterval: 5 * time.Second,
		engineTimeout:  30 * time.Second,
		prunedManifest: envOr("BURNIN_DCGM_PRUNED_MANIFEST", prunedManifestPath),
	}

	if v := os.Getenv("BURNIN_DURATION_SECONDS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			return c, fmt.Errorf("BURNIN_DURATION_SECONDS=%q is not a non-negative integer", v)
		}
		c.duration = time.Duration(n) * time.Second
	}
	if v := os.Getenv("BURNIN_DCGM_SAMPLE_INTERVAL_SECONDS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return c, fmt.Errorf("BURNIN_DCGM_SAMPLE_INTERVAL_SECONDS=%q is not a positive integer", v)
		}
		c.sampleInterval = time.Duration(n) * time.Second
	}

	// A node that already runs a DCGM host engine (the GPU operator's
	// dcgm-exporter, say) must not get a second one competing for the same
	// device watches, so connecting to the existing one is supported.
	if addr := os.Getenv("BURNIN_DCGM_HOSTENGINE_ADDRESS"); addr != "" {
		c.address = addr
		c.startEngine = false
		return c, nil
	}
	port := envOr("BURNIN_DCGM_PORT", "5555")
	c.address = "127.0.0.1:" + port
	c.startEngine = true
	// Bound to loopback deliberately: the host engine speaks an unauthenticated
	// protocol, and this pod may run with hostNetwork.
	// --home-dir is not optional in a container, and leaving it out cost a real
	// misdiagnosis (#304). The image runs as uid 65532 with a read-only-ish
	// root, so nvvs cannot create its working file in "/" — DCGM reports that
	// as a FAILED subtest reading:
	//
	//   Permissions and OS Blocks: No permission to create a file in directory
	//   '/'. Please restart the hostengine with parameter --home-dir ...
	//
	// which arrives as a hardware verdict about a perfectly healthy GPU. /tmp is
	// writable in this image and is where a scratch file belongs.
	c.hostEngineArgs = []string{"-n", "-b", "127.0.0.1", "-p", port, "--home-dir", "/tmp"}
	if raw := os.Getenv("BURNIN_NV_HOSTENGINE_ARGS"); raw != "" {
		c.hostEngineArgs = strings.Fields(raw)
	}
	return c, nil
}

func run() int {
	out := os.Stdout
	rep := newReport()

	cfg, err := loadConfig()
	if err != nil {
		return finish(out, rep, exitError, markerError, err.Error())
	}

	level, levelSource, err := resolveLevel(cfg.level, cfg.duration)
	if err != nil {
		return finish(out, rep, exitError, markerError, err.Error())
	}
	rep.setInt(keyDiagLevel, int64(level))
	rep.set(keyDiagLevelSource, levelSource)

	// The runner bounds itself at the duration it was given. The pod's
	// activeDeadlineSeconds sits 120s further out, so this timer always fires
	// first — which is the point: a self-imposed deadline exits through finish()
	// with every metric gathered so far, whereas the kubelet's deadline is a
	// SIGKILL that publishes nothing at all.
	budget := cfg.duration
	if budget <= 0 {
		budget = 4 * levelBudget(level)
		logf("BURNIN_DURATION_SECONDS is unset; bounding the diagnostic at %s", budget)
	}
	ctx, cancel := context.WithTimeout(context.Background(), budget)
	defer cancel()

	// Checked before anything else is attempted, because "DCGM is not in this
	// image" has a fix and "exec: nv-hostengine: executable file not found in
	// $PATH" does not read like one. It is an Error, never a Skip: the absence
	// of a tool says nothing whatsoever about the hardware, and a node that
	// silently reported "not applicable" here would sail through a gate it was
	// never actually measured by.
	if err := checkDCGMPresent(cfg); err != nil {
		rep.set(keyReason, err.Error())
		return finish(out, rep, exitError, markerError, err.Error())
	}

	if n := prunedObjectCount(cfg.prunedManifest); n > 0 {
		rep.setInt(keyPrunedObjects, int64(n))
		logf("%d DCGM objects were removed from this image because they required an "+
			"NVIDIA redistributable library (see /usr/share/glimmer-burnin/dcgm-pruned.txt); "+
			"subtests needing them will report as not run", n)
	}

	if cfg.startEngine {
		stopEngine, err := startHostEngine(ctx, cfg)
		if err != nil {
			return finish(out, rep, exitError, markerError,
				"could not start "+cfg.hostEngineBin+": "+err.Error())
		}
		defer stopEngine()
	}

	gpus, err := waitForEngine(ctx, cfg)
	if err != nil {
		return finish(out, rep, exitError, markerError, err.Error())
	}
	if gpus >= 0 {
		rep.setInt(keyGPUCount, int64(gpus))
	}
	if gpus == 0 {
		// DCGM reached the driver and found nothing it recognises. That is the
		// definition of "this test does not apply here", not a failure: the
		// part is unjudged, and calling it a failure would condemn hardware
		// DCGM never looked at.
		return finish(out, rep, exitSkip, markerSkip,
			"DCGM discovered no supported GPUs on this node")
	}

	s := newSampler(cfg)
	s.sampleOnce(ctx) // baseline, before any load
	s.run(ctx)

	start := time.Now()
	stdout, stderr, code, runErr := runDiag(ctx, cfg, level)
	elapsed := time.Since(start)

	s.shutdown()
	s.sampleOnce(ctx) // final reading, after the diagnostic has settled

	if res, parseErr := parseDiagJSON(stdout); parseErr == nil {
		reportDiag(rep, res)
		s.report(rep)
		rep.setInt(keyDcgmiExitCode, int64(code))
		rep.setNumber(keyElapsedS, round1(elapsed.Seconds()))
		return verdict(out, rep, res, code, ctx.Err(), stdout, stderr, budget)
	}

	// No parseable document. Everything measured is still evidence.
	s.report(rep)
	rep.setInt(keyDcgmiExitCode, int64(code))
	rep.setNumber(keyElapsedS, round1(elapsed.Seconds()))

	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return finish(out, rep, exitError, markerError, fmt.Sprintf(
			"the diagnostic did not finish within %s, so the hardware is unjudged; "+
				"give the test more time or set BURNIN_DCGM_LEVEL lower", budget))
	}
	if runErr != nil {
		return finish(out, rep, exitError, markerError,
			"could not execute "+cfg.dcgmiBin+": "+runErr.Error())
	}
	if sig := unsupportedSignature(stdout + "\n" + stderr); sig != "" {
		rep.set(keyReason, sig)
		return finish(out, rep, exitSkip, markerSkip,
			"DCGM will not run against this part: "+sig)
	}
	return finish(out, rep, exitError, markerError, fmt.Sprintf(
		"dcgmi exited %d and produced no readable JSON result: %s",
		code, firstLine(stderr+"\n"+stdout)))
}

func reportDiag(rep *report, res *diagResults) {
	if res.Version != "" {
		rep.set(keyDcgmVersion, res.Version)
	}
	if res.DriverVersion != "" {
		rep.set(keyDriverVersion, res.DriverVersion)
	}
	c := res.Counts()
	rep.setInt(keyTestsRun, int64(c.Executed))
	// DCGM's "tests failed" is a count of subtests, not a verdict. A run that
	// exits non-zero with zero failed subtests means the suite could not run,
	// which is why the two are reported separately.
	rep.setInt(keyTestsFailed, int64(c.Failed))
	rep.setInt(keyTestsWarned, int64(c.Warned))
	rep.setInt(keyTestsNotRun, int64(c.NotRun))
	rep.setInt(keyTestsSkipped, int64(c.Skipped))
}

// verdict turns a parsed diagnostic into an exit code.
//
// The order of these cases is the policy, and it is not arbitrary:
//
//   - a timeout comes first, because a diagnostic we cut short cannot be read
//     as a pass no matter how many subtests had passed by then;
//   - a failed subtest outranks an incomplete one: partial coverage does not
//     erase an observed failure;
//   - an incomplete subtest outranks a pass, so a suite that could only run
//     half its tests reports Error rather than a pass that overstates what was
//     checked;
//   - a non-zero dcgmi exit alongside an all-pass document is a contradiction,
//     and a contradiction is not evidence of health.
func verdict(
	out io.Writer, rep *report, res *diagResults,
	code int, ctxErr error, stdout, stderr string, budget time.Duration,
) int {
	if errors.Is(ctxErr, context.DeadlineExceeded) {
		return finish(out, rep, exitError, markerError, fmt.Sprintf(
			"the diagnostic did not finish within %s, so the hardware is unjudged", budget))
	}

	c := res.Counts()

	if c.Total == 0 {
		// A Skip removes a node from scope without anyone looking at it again,
		// so it may only be reached from evidence we actually READ.
		//
		// "We recorded no subtests" is not "DCGM produced no subtests", and the
		// two had the same representation here. A status key spelled
		// differently by a future DCGM, or a document extractJSON only partly
		// recovered, lands on this branch too — and matching unsupported-part
		// prose anywhere across stdout and stderr then turned a document that
		// may have REPORTED FAILURES into "not applicable to this hardware"
		// (#322). The guard's premise was not what it tested.
		if res.SawResultStructure {
			return finish(out, rep, exitError, markerError,
				"the diagnostic reported subtests this runner could not read, so the hardware is "+
					"unjudged; a document we cannot parse is not a part DCGM refused to test")
		}
		if sig := unsupportedSignature(stdout + "\n" + stderr); sig != "" {
			rep.set(keyReason, sig)
			return finish(out, rep, exitSkip, markerSkip,
				"DCGM will not run against this part: "+sig)
		}
		return finish(out, rep, exitError, markerError,
			"the diagnostic returned a result document containing no subtests")
	}

	if c.Executed == 0 {
		// DCGM ran and skipped everything it was asked to do. Nothing about the
		// hardware was established, so this is not applicable rather than
		// passed — a suite that checks nothing must never report a green node.
		reason := strings.Join(res.skipReasons, "; ")
		if reason == "" {
			reason = fmt.Sprintf("DCGM skipped all %d subtests without saying why", c.Total)
		}
		rep.set(keyReason, reason)
		return finish(out, rep, exitSkip, markerSkip,
			"DCGM skipped every subtest on this part: "+reason)
	}

	// #304. Reported whatever the verdict turns out to be: a node whose
	// persistence mode is off should say so on a pass, on an error, and on a
	// fail alike, because it is the thing an operator has to act on.
	if len(c.ConfigFindings) > 0 {
		rep.set(keyConfigFindings, strings.Join(c.ConfigFindings, " | "))
	}
	// Same reasoning as the config findings above, for the warned subtests: a
	// profile that gates testsWarned needs to be told what it tripped on, and
	// stderr is not somewhere a verdict can read (#323).
	if len(res.warnReasons) > 0 {
		rep.set(keyWarnFindings, strings.Join(res.warnReasons, " | "))
	}
	if len(c.UnreadableFindings) > 0 {
		// UNMEASURABLE, not zero. `n/a` is the reserved value this project uses
		// for exactly this, and it is what lets a threshold with
		// RequiredIfMeasurable report NOT EVALUATED instead of failing closed
		// against a field the tool could not read.
		rep.set(keyUnreadableFields, "n/a")
		rep.set(keyExcusedSubtests, strings.Join(c.ExcusedNames, ","))
	} else if len(c.ExcusedNames) > 0 {
		rep.set(keyExcusedSubtests, strings.Join(c.ExcusedNames, ","))
	}

	if c.Failed > 0 {
		reason := fmt.Sprintf("%d of %d DCGM subtests failed: %s",
			c.Failed, c.Executed, strings.Join(c.FailedNames, ", "))
		if c.BlockingFinding != "" {
			// FIRST, because the displayed reason is truncated and this is the
			// one finding that decided the verdict. Everything else follows as
			// context.
			reason += " — the finding that indicts the hardware: " + c.BlockingFinding
		}
		if len(res.failReasons) > 0 {
			reason += " — all findings: " + strings.Join(res.failReasons, "; ")
		}
		return finish(out, rep, exitFail, markerFail, reason)
	}

	if c.NotRun > 0 {
		if len(c.ExcusedNames) > 0 {
			// Distinguished from an ordinary partial run because the remedy is
			// completely different: this node needs configuring, not replacing.
			detail := strings.Join(append(append([]string{}, c.ConfigFindings...), c.UnreadableFindings...), "; ")
			return finish(out, rep, exitError, markerError, fmt.Sprintf(
				"%d of %d DCGM subtests are UNJUDGED because every finding against them was a node "+
					"setting or a field DCGM could not read, not a fault in the part — %s. Fix the "+
					"configuration and re-run; this is deliberately not a Fail, which would condemn "+
					"the hardware for something a command fixes (issue #304)",
				c.NotRun, c.Total, detail))
		}
		return finish(out, rep, exitError, markerError, fmt.Sprintf(
			"%d of %d DCGM subtests did not run, so this node is only partly checked; "+
				"treating a partial suite as a pass would overstate what was verified",
			c.NotRun, c.Total))
	}

	// A SUBTEST DCGM DECLINED TO RUN LEAVES THAT ASPECT UNVERIFIED.
	//
	// The same rule as NotRun directly above, and it was missing: the verdict
	// checked Executed==0, Failed and NotRun, then fell through to Pass with no
	// rule for Skipped at all. A run where DCGM executed one subtest and skipped
	// four reported a GREEN NODE — "a suite that checks nothing must never
	// report a green node" is already stated by the Executed==0 branch, but was
	// applied only to the all-or-nothing case.
	//
	// Not hypothetical. DCGM gates every CUDA-backed plugin behind a per-SKU
	// allowlist that GB10 is not on, so a level-2+ run on a DGX Spark executes
	// the deployment check and skips the memory, PCIe and stress plugins. Under
	// the old rule that node passed the DCGM gate having had no memory test, no
	// PCIe test and no stress test run against it — the exact false negative the
	// exit-2 path exists to prevent, arriving through the pass path instead.
	//
	// ERROR, not Fail: DCGM declining to test something says nothing about the
	// silicon, so it must not condemn the part. Unjudged and retryable is what
	// the NotRun branch already returns for the same situation.
	//
	// The all-skipped case never reaches here — Executed==0 returns Skip above,
	// which is right for a kind that does not apply to this part at all. This is
	// the PARTIAL case, where some of the node was checked and some was not.
	if c.Skipped > 0 {
		// Emitted HERE rather than beside the counts, so the names travel with
		// the verdict that depends on them — verdict() is reachable on its own
		// and a caller that never ran run()'s reporting still gets the evidence.
		if len(c.SkippedNames) > 0 {
			rep.set(keySkippedSubtests, strings.Join(c.SkippedNames, ","))
		}
		if len(res.skipReasons) > 0 {
			rep.set(keySkipReason, strings.Join(res.skipReasons, " | "))
		}
		return finish(out, rep, exitError, markerError, fmt.Sprintf(
			"%d of %d DCGM subtests were SKIPPED, so this node is only partly checked (%d executed, "+
				"%d skipped). Passing it would certify hardware DCGM never tested. If the skips are the "+
				"per-SKU plugin allowlist (GB10 and other unlisted parts), enable them explicitly with "+
				"`-p <plugin>.is_allowed=true`; if they genuinely do not apply to this part, target the "+
				"profile with a node selector rather than accepting an unverified pass. Reasons: %s",
			c.Skipped, c.Total, c.Executed, c.Skipped, orNone(strings.Join(res.skipReasons, "; "))))
	}

	if code != 0 {
		return finish(out, rep, exitError, markerError, fmt.Sprintf(
			"every DCGM subtest passed but dcgmi exited %d; the evidence contradicts "+
				"itself, so the hardware is unjudged", code))
	}

	if c.Warned > 0 {
		// A DCGM warning is a real observation that did not cross DCGM's own
		// failure line. It is reported (testsWarned) and left to the profile's
		// thresholds to act on, rather than being promoted to a failure here —
		// the runner's job is to measure, and the profile's is to decide.
		logf("%d subtests passed with warnings: %s", c.Warned, strings.Join(res.failReasons, "; "))
	}
	return finish(out, rep, exitPass, markerPass, "")
}

// dcgmiArgs builds a dcgmi command line.
//
// The subsystem MUST come first. dcgmi reads argv[1] as the name of the
// subsystem and only then hands the remaining arguments to that subsystem's own
// parser, so the natural-looking `dcgmi --host H diag -r 1 -j` is rejected with
// "ERROR: Invalid subsystem" and exit 255 — on every node, against every part,
// with no output resembling a diagnostic. Verified against DCGM 4.2.3.
//
// That failure mode is why this is a function rather than three literal slices:
// the ordering is a property of the tool, and it has to be stated once, in a
// place a test can pin.
func dcgmiArgs(subsystem, address string, rest ...string) []string {
	args := make([]string, 0, 3+len(rest))
	args = append(args, subsystem, "--host", address)
	return append(args, rest...)
}

// runDiag executes the diagnostic. The returned error is reserved for failures
// to execute dcgmi at all; a non-zero exit is a result, not an error.
func runDiag(ctx context.Context, cfg config, level int) (string, string, int, error) {
	args := dcgmiArgs("diag", cfg.address, "-r", strconv.Itoa(level), "-j")
	logf("running %s %s", cfg.dcgmiBin, strings.Join(args, " "))
	return runCmd(ctx, 0, cfg.dcgmiBin, args...)
}

// checkDCGMPresent reports whether the DCGM tools this runner drives are
// reachable.
//
// The published image deliberately does not ship them: NVIDIA's DCGM binary
// packages are licensed under the NVIDIA Data Center GPU Manager License, which
// grants installation and use but no right to redistribute, so a public image
// carrying them would be a licence violation as well as a breach of this
// project's permissive-only rule. DCGM is supplied by the site instead. See the
// README.
func checkDCGMPresent(cfg config) error {
	need := []struct{ what, bin, env string }{
		{"the DCGM client", cfg.dcgmiBin, "BURNIN_DCGM_BIN"},
	}
	if cfg.startEngine {
		need = append(need, struct{ what, bin, env string }{
			"the DCGM host engine", cfg.hostEngineBin, "BURNIN_NV_HOSTENGINE_BIN"})
	}
	for _, n := range need {
		if _, err := exec.LookPath(n.bin); err != nil {
			return fmt.Errorf(
				"%s (%q) is not executable on PATH, so this node is unjudged: %v. "+
					"This image ships no DCGM — NVIDIA's DCGM binaries may not be "+
					"redistributed. Mount a DCGM installation at /usr/local/dcgm "+
					"(bin/ and lib/), or set %s, or point "+
					"BURNIN_DCGM_HOSTENGINE_ADDRESS at a host engine that is already "+
					"running on this node. See the runner README",
				n.what, n.bin, err, n.env)
		}
	}
	return nil
}

// startHostEngine launches nv-hostengine and returns a function that stops it.
func startHostEngine(ctx context.Context, cfg config) (func(), error) {
	cmd := exec.CommandContext(ctx, cfg.hostEngineBin, cfg.hostEngineArgs...)
	// Never stdout: stdout is the runner contract, and a host engine banner
	// landing in it would be read as the run's Message.
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	stopped := make(chan struct{})
	go func() {
		_ = cmd.Wait()
		close(stopped)
	}()
	return func() {
		if cmd.Process == nil {
			return
		}
		_ = cmd.Process.Signal(os.Interrupt)
		select {
		case <-stopped:
		case <-time.After(5 * time.Second):
			_ = cmd.Process.Kill()
		}
	}, nil
}

// gpuCountPattern matches dcgmi discovery's summary line ("1 GPU found.").
var gpuCountPattern = regexp.MustCompile(`(?m)^\s*(\d+)\s+GPUs?\s+found`)

// waitForEngine blocks until dcgmi can talk to the host engine, and reports how
// many GPUs DCGM sees. It returns -1 when the count could not be read, which is
// not an error: an unparseable banner is a reason to carry on to the
// diagnostic, not a reason to condemn or excuse the node.
func waitForEngine(ctx context.Context, cfg config) (int, error) {
	deadline := time.Now().Add(cfg.engineTimeout)
	var last string
	for {
		stdout, stderr, code, err := runCmd(ctx, 15*time.Second, cfg.dcgmiBin,
			dcgmiArgs("discovery", cfg.address, "-l")...)
		if err == nil && code == 0 {
			if m := gpuCountPattern.FindStringSubmatch(stdout); m != nil {
				n, convErr := strconv.Atoi(m[1])
				if convErr == nil {
					return n, nil
				}
			}
			return -1, nil
		}
		last = strings.TrimSpace(firstLine(stderr))
		if last == "" && err != nil {
			last = err.Error()
		}
		if ctx.Err() != nil {
			return -1, fmt.Errorf("gave up waiting for the DCGM host engine at %s: %v",
				cfg.address, ctx.Err())
		}
		if time.Now().After(deadline) {
			return -1, fmt.Errorf(
				"the DCGM host engine at %s did not become reachable within %s: %s",
				cfg.address, cfg.engineTimeout, last)
		}
		select {
		case <-ctx.Done():
			return -1, fmt.Errorf("gave up waiting for the DCGM host engine at %s: %v",
				cfg.address, ctx.Err())
		case <-time.After(500 * time.Millisecond):
		}
	}
}

// runCmd runs a command and captures both streams. A non-zero exit is returned
// as a code with a nil error, so callers can tell "the tool said no" from "the
// tool did not run".
func runCmd(ctx context.Context, timeout time.Duration, name string, args ...string) (string, string, int, error) {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		code = exitErr.ExitCode()
		err = nil
	}
	return stdout.String(), stderr.String(), code, err
}

// prunedObjectCount reads the build's prune manifest. A missing file means the
// build shipped DCGM whole, which is the expected case.
func prunedObjectCount(path string) int {
	if path == "" {
		return 0
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	n := 0
	for _, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) != "" {
			n++
		}
	}
	return n
}

func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return line
		}
	}
	return ""
}

func round1(v float64) float64 {
	return float64(int64(v*10+0.5)) / 10
}
