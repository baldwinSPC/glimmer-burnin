// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Defaults.
//
// BURNIN_DURATION_SECONDS is set by the operator for every runner it schedules
// (internal/controller/pods.go), and defaultDurationSeconds matches the value
// the operator itself falls back to, so a runner started by hand behaves like
// one started by a BurnInRun.
//
// The rest are memory-stress specific and exist so a profile can retune the
// workload through spec.runner.env without cutting a new image — the runner
// contract is an exit code and key=value metrics, not a fixed command line.
const (
	defaultDurationSeconds = 600
	defaultFraction        = 0.70
	defaultMinMB           = 256
	defaultVerbosity       = 8
	defaultBinary          = "/usr/local/bin/stressapptest"

	// mib is what stressapptest means by "MB": its -M argument and its MB/s
	// figures are both in units of 2^20 bytes.
	mib int64 = 1 << 20
)

// Reserve and cap, in seconds.
//
// stressapptest's -s clock starts only after it has allocated the region and
// filled it with patterns, and that setup is not free: on a large region it is
// tens of seconds. A -s equal to the whole duration therefore overruns the
// pod's activeDeadlineSeconds (duration + 120s of grace, which exists for image
// pull and container start — not for us to spend), and a deadline kill is
// recorded as an Error for every node in the run. Hold part of the budget back.
const (
	minReserveSeconds = 15
	maxReserveSeconds = 60
	minRunSeconds     = 10
	// hardCapGraceSeconds is how long past BURNIN_DURATION_SECONDS the wrapper
	// lets stressapptest live before killing it itself. It sits inside the
	// pod's grace so the wrapper — not the kubelet — is what ends a wedged run,
	// which means the metrics gathered so far still reach stdout.
	hardCapGraceSeconds = 60
)

// config is the runner's tunable surface. Every field has a default that is
// sound on an untuned node; nothing here is required.
type config struct {
	durationSeconds int
	// memoryMB, when > 0, is an explicit -M and overrides auto-sizing entirely.
	memoryMB int64
	// fraction of the usable memory budget to hand to stressapptest when
	// auto-sizing. The remainder is headroom: the tool's own bookkeeping, the
	// page cache, and whatever else shares the cgroup. Allocating the whole
	// budget gets the container OOM-killed, which reports as an infrastructure
	// Error and burns the run's time without judging the hardware.
	fraction float64
	// minMB is the smallest region worth calling a memory stress test.
	minMB         int64
	threads       int
	invertThreads int
	checkThreads  int
	cpuThreads    int
	verbosity     int
	stressfulCopy bool
	extraArgs     []string
	binary        string
}

type getenvFunc func(string) (string, bool)

func loadConfig(getenv getenvFunc) (config, error) {
	cfg := config{
		fraction:  defaultFraction,
		minMB:     defaultMinMB,
		verbosity: defaultVerbosity,
		binary:    defaultBinary,
	}

	duration, err := intVar(getenv, "BURNIN_DURATION_SECONDS", defaultDurationSeconds, 1, 1<<31-1)
	if err != nil {
		return config{}, err
	}
	cfg.durationSeconds = int(duration)

	if cfg.memoryMB, err = intVar(getenv, "BURNIN_MEMORY_MB", 0, 0, 1<<40); err != nil {
		return config{}, err
	}
	if cfg.fraction, err = floatVar(getenv, "BURNIN_MEMORY_FRACTION", defaultFraction, 0.01, 1.0); err != nil {
		return config{}, err
	}
	if cfg.minMB, err = intVar(getenv, "BURNIN_MEMORY_MIN_MB", defaultMinMB, 1, 1<<40); err != nil {
		return config{}, err
	}
	threads, err := intVar(getenv, "BURNIN_MEMORY_THREADS", 0, 0, 4096)
	if err != nil {
		return config{}, err
	}
	cfg.threads = int(threads)
	invert, err := intVar(getenv, "BURNIN_MEMORY_INVERT_THREADS", 0, 0, 4096)
	if err != nil {
		return config{}, err
	}
	cfg.invertThreads = int(invert)
	check, err := intVar(getenv, "BURNIN_MEMORY_CHECK_THREADS", 0, 0, 4096)
	if err != nil {
		return config{}, err
	}
	cfg.checkThreads = int(check)
	cpu, err := intVar(getenv, "BURNIN_MEMORY_CPU_THREADS", 0, 0, 4096)
	if err != nil {
		return config{}, err
	}
	cfg.cpuThreads = int(cpu)
	verbosity, err := intVar(getenv, "BURNIN_MEMORY_VERBOSITY", defaultVerbosity, 0, 20)
	if err != nil {
		return config{}, err
	}
	cfg.verbosity = int(verbosity)
	if cfg.stressfulCopy, err = boolVar(getenv, "BURNIN_MEMORY_STRESSFUL_COPY", false); err != nil {
		return config{}, err
	}
	if raw, ok := getenv("BURNIN_MEMORY_EXTRA_ARGS"); ok {
		cfg.extraArgs = strings.Fields(raw)
	}
	if raw, ok := getenv("BURNIN_STRESSAPPTEST_BIN"); ok && raw != "" {
		cfg.binary = raw
	}
	return cfg, nil
}

// intVar reads an integer environment variable.
//
// A malformed or out-of-range value is an error rather than a silent fall back
// to the default. A profile author who writes BURNIN_MEMORY_FRACTION=70 meaning
// 70% must find out at the first run, not discover months later that every node
// in the fleet was accepted on a default-sized workload that nobody chose.
func intVar(getenv getenvFunc, name string, def, minimum, maximum int64) (int64, error) {
	raw, ok := getenv(name)
	if !ok || strings.TrimSpace(raw) == "" {
		return def, nil
	}
	v, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%s=%q is not an integer", name, raw)
	}
	if v < minimum || v > maximum {
		return 0, fmt.Errorf("%s=%d is outside the accepted range [%d, %d]", name, v, minimum, maximum)
	}
	return v, nil
}

func floatVar(getenv getenvFunc, name string, def, minimum, maximum float64) (float64, error) {
	raw, ok := getenv(name)
	if !ok || strings.TrimSpace(raw) == "" {
		return def, nil
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil {
		return 0, fmt.Errorf("%s=%q is not a number", name, raw)
	}
	if v < minimum || v > maximum {
		return 0, fmt.Errorf("%s=%v is outside the accepted range [%v, %v]", name, v, minimum, maximum)
	}
	return v, nil
}

func boolVar(getenv getenvFunc, name string, def bool) (bool, error) {
	raw, ok := getenv(name)
	if !ok || strings.TrimSpace(raw) == "" {
		return def, nil
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "yes", "on":
		return true, nil
	case "0", "false", "no", "off":
		return false, nil
	}
	// A typo here would quietly disable the stronger copy for a whole fleet's
	// burn-in, so it is refused rather than interpreted.
	return false, fmt.Errorf("%s=%q is not a boolean (use 1/0, true/false, yes/no, on/off)", name, raw)
}

// plan is the concrete workload one execution will run.
type plan struct {
	testMB     int64
	threads    int
	satSeconds int
	// coveredPct is testMB as a percentage of the node's TOTAL physical RAM,
	// not of the slice the runner was allowed to use. It answers the question
	// a pass actually depends on — how much of this machine's memory was
	// touched — which a cgroup-limited pod can make very small without any of
	// the other metrics changing.
	coveredPct float64
	// explicitSize records that BURNIN_MEMORY_MB chose the size, so the log can
	// say why the auto-sizing floor did not apply.
	explicitSize bool
}

// args builds the stressapptest command line.
//
// Only flags that are actually needed are passed: the tool's own defaults are
// well chosen, and a flag we do not pass is one that cannot go wrong.
func (p plan) args(cfg config) []string {
	args := []string{
		"-M", strconv.FormatInt(p.testMB, 10),
		"-s", strconv.Itoa(p.satSeconds),
		"-m", strconv.Itoa(p.threads),
		"-v", strconv.Itoa(cfg.verbosity),
	}
	if cfg.invertThreads > 0 {
		args = append(args, "-i", strconv.Itoa(cfg.invertThreads))
	}
	if cfg.checkThreads > 0 {
		args = append(args, "-c", strconv.Itoa(cfg.checkThreads))
	}
	if cfg.cpuThreads > 0 {
		args = append(args, "-C", strconv.Itoa(cfg.cpuThreads))
	}
	if cfg.stressfulCopy {
		args = append(args, "-W")
	}
	return append(args, cfg.extraArgs...)
}

// satSeconds is how long stressapptest is told to run, given the whole budget.
func satSeconds(duration int) int {
	reserve := duration / 10
	if reserve < minReserveSeconds {
		reserve = minReserveSeconds
	}
	if reserve > maxReserveSeconds {
		reserve = maxReserveSeconds
	}
	s := duration - reserve
	if s < minRunSeconds {
		s = minRunSeconds
	}
	return s
}

// planRun decides the workload, or refuses with a verdict-carrying error.
func planRun(cfg config, sys sysInfo) (plan, error) {
	p := plan{threads: cfg.threads, satSeconds: satSeconds(cfg.durationSeconds)}
	if p.threads <= 0 {
		p.threads = sys.cpus
	}
	if p.threads < 1 {
		p.threads = 1
	}

	budget := sys.memAvailableBytes
	if sys.limitBytes > 0 && sys.limitBytes < budget {
		budget = sys.limitBytes
	}

	if cfg.memoryMB > 0 {
		// An explicit size is obeyed, floor and all: the operator asked for a
		// specific region and knows what the node has. It is not clamped to the
		// budget either — clamping would silently test less memory than the
		// profile says it tests.
		p.testMB = cfg.memoryMB
		p.explicitSize = true
	} else {
		p.testMB = int64(float64(budget)*cfg.fraction) / mib
		if p.testMB < cfg.minMB {
			return plan{}, tooSmall(cfg, sys, p.testMB)
		}
	}

	if sys.memTotalBytes > 0 {
		p.coveredPct = 100 * float64(p.testMB) / float64(sys.memTotalBytes/mib)
	}
	return p, nil
}

// tooSmall decides what "there is not enough memory to test" means.
//
// The two cases are not the same verdict and must not be collapsed:
//
//   - A cgroup limit is in force. Somebody configured this pod, the node's own
//     memory is untouched by the question, and the hardware has not been
//     judged: that is an Error, and the message says which knob to turn.
//   - No limit is in force and the machine itself is this small. Then the test
//     genuinely does not apply to this hardware, which is what exit 2 means.
//
// Reporting the first as a Skip would quietly excuse a misconfigured profile
// from ever testing memory; reporting either as a Fail would condemn hardware
// nobody measured.
func tooSmall(cfg config, sys sysInfo, testMB int64) error {
	if sys.limitBytes > 0 {
		return &runnerError{code: exitError, msg: fmt.Sprintf(
			"the pod's memory limit of %d MiB leaves only %d MiB testable, below the %d MiB minimum stress region — raise spec.resources.limits.memory or set BURNIN_MEMORY_MB; this node's memory was not judged",
			sys.limitBytes/mib, testMB, cfg.minMB)}
	}
	return &runnerError{code: exitSkip, msg: fmt.Sprintf(
		"node has %d MiB of usable host memory, which leaves %d MiB testable — below the %d MiB minimum stress region",
		sys.memAvailableBytes/mib, testMB, cfg.minMB)}
}

// runnerError is an error that already knows which exit code it deserves.
type runnerError struct {
	code int
	msg  string
}

func (e *runnerError) Error() string { return e.msg }
