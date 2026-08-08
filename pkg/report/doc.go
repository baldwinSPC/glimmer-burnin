// Package report turns stored burn-in verdicts into documents.
//
// The operator already produces more than it can currently show. A failed test
// records every threshold it did not satisfy, classified by cause — whether the
// hardware fell short, the runner's report could not support a judgement, or the
// threshold itself is broken. That classification is the most decision-relevant
// thing in the system: it is the difference between replacing a part and fixing a
// profile. Until this package existed there was nowhere for it to go except a
// consumer's own renderer, and every consumer would have written a different one.
//
// # Two rules, and they are not stylistic
//
// FIRST: this package imports pkg/contract and the standard library, and nothing
// else. Not api/v1alpha1, which pkg/verdict takes on because it must speak
// Threshold. The audience for a renderer — a CI job, a bare-metal CLI, an ingest
// service — is exactly the audience that should not be made to carry apimachinery
// to read a verdict. TestPackageImportsNothingUnexpected enforces it.
//
// SECOND: a report never fabricates. A serial number that was not captured is
// absent, not an empty string and not "unknown". A driver version reaches a
// document only if a fingerprint carried it. Where a renderer cannot say
// something, it says nothing, and where this package has to degrade — a run that
// had not finished, an envelope that arrived without its peers — it records that
// in Resolved.Warnings rather than papering over it.
//
// The second rule is the same one the runners are held to, for the same reason.
// A runner that prints a zero it did not measure certifies hardware nobody
// looked at. A report that fills in a plausible blank does it one step further
// from anyone who could catch it, in the artifact most likely to be forwarded,
// printed, and attached to a warranty claim.
//
// # What a renderer receives
//
// Renderers do not walk raw envelopes. Resolve reduces a delivery stream — a
// terminal verdict, the per-test completions that preceded it, any checkpoints —
// into one Resolved view, applying the precedence rules in resolve.go once so
// that three renderers cannot disagree about which delivery was authoritative.
package report
