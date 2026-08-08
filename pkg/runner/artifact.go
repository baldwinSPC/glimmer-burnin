package runner

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"unicode/utf8"
)

// The artifact channel: how a runner returns evidence that is not a metric.
//
// # Why a fence on stdout rather than a file or an upload
//
// A runner's whole contract is an exit code and key=value lines on stdout, and
// that is what makes a runner trivially testable, trivially portable, and
// unable to reach anything it was not given. Evidence that is not a number —
// dcgmi's own JSON document, a captured nvidia-smi topology, a failing kernel's
// disassembly — had nowhere to go, so the operator kept the parsed metrics and
// dropped the rest.
//
// The alternatives were considered and refused. A shared volume needs the
// operator to provision one per attempt and to trust what a runner writes into
// it. Object storage puts egress credentials in the blast radius of a component
// whose entire job is running semi-trusted workloads on cordoned nodes. Both
// widen what a runner can touch; a fence on the channel that already exists
// widens nothing.
//
// # The format
//
//	-----BEGIN BURNIN ARTIFACT <name> <mediaType>-----
//	<payload>
//	-----END BURNIN ARTIFACT-----
//
// # The rules, and each one is a way this could corrupt a verdict
//
//   - A line inside a fence is PAYLOAD, never a metric. A dcgmi JSON document
//     is full of "key": value lines and several would parse as metrics; without
//     this a runner emitting evidence would silently rewrite its own
//     measurements.
//   - An UNTERMINATED fence is dropped whole. Accepting a partial payload would
//     store a truncated JSON document that a renderer then fails on, at report
//     time, far from the runner that produced it.
//   - A BEGIN line whose HEADER does not parse is still a fence: the artifact is
//     refused and its payload is consumed. Treating it as ordinary text instead
//     was a real defect — a runner naming an artifact "has space.json" produced
//     a marker with three fields, which fell through to the metric scanner along
//     with the entire payload it was announcing. A line that says it opens a
//     fence opens one; the only question left is whether its contents can be
//     kept.
//   - A NESTED begin marker refuses the outer artifact and RE-SCANS from that
//     marker, treating it as the start of a fresh one. Resuming ordinary metric
//     parsing there instead would be the first rule's own defect by another
//     route: the inner block's payload lines would reach the scanner. Re-scanning
//     also handles the likely cause, which is a runner that emitted two
//     artifacts and forgot an END between them.
//   - An OVERSIZED payload is dropped and RECORDED as dropped. A report that
//     quietly omits evidence is worse than one that says the evidence was too
//     large: the first is indistinguishable from a runner that produced none.
//
// Nothing here can change a metric, a verdict or an exit code. An artifact is
// evidence about a verdict, never part of one.

func isSpace(b byte) bool { return b == ' ' || b == '\t' }

const (
	artifactBegin = "-----BEGIN BURNIN ARTIFACT"
	artifactEnd   = "-----END BURNIN ARTIFACT-----"
	// markerSuffix closes the BEGIN line.
	markerSuffix = "-----"
)

// MaxArtifactBytes bounds ONE artifact after extraction.
//
// It matches the operator's per-artifact ConfigMap budget. Bounding here as well
// means an oversized payload is refused at the point it is parsed rather than
// carried through the operator to be refused by the apiserver, where the failure
// would be a rejected status write that loses the whole verdict rather than one
// piece of evidence.
const MaxArtifactBytes = 256 * 1024

// Artifact is one piece of non-metric evidence a runner returned.
type Artifact struct {
	// Name identifies it within the attempt, e.g. "dcgmi.json".
	Name string
	// MediaType is what the payload IS, so a renderer does not have to guess.
	MediaType string
	// Payload is the extracted bytes. Empty when Dropped is set.
	//
	// The channel is LINE-oriented — it is stdout — so a payload is a sequence
	// of lines and always ends in a newline. A runner with genuinely binary
	// evidence must encode it (base64, and say so in the media type); a raw
	// blob containing a 0x0a is indistinguishable from a line break here, and
	// nothing downstream could tell them apart either.
	Payload string
	// SizeBytes is the payload's TRUE size, which is not len(Payload) when the
	// artifact was dropped for being oversized — that is the whole point of
	// recording it, since "how much evidence did we lose" is the question a
	// reader asks next.
	SizeBytes int
	// Digest is "sha256:<hex>" over the accepted payload, so a consumer
	// fetching it separately can verify what it got. Empty when Dropped.
	Digest string
	// Dropped says why this artifact was refused, and is empty when it was
	// accepted. A refused artifact is still REPORTED: silence would be
	// indistinguishable from a runner that emitted nothing.
	Dropped string
}

// extractArtifacts pulls fenced blocks out of stdout and returns the remaining
// lines for ordinary key=value parsing.
//
// The two are done in one pass rather than two, because a second pass would have
// to re-derive which lines were inside a fence — and any disagreement between
// the two passes is a payload line parsed as a metric.
func extractArtifacts(stdout string) (artifacts []Artifact, rest []string) {
	lines := strings.Split(stdout, "\n")
	for i := 0; i < len(lines); i++ {
		line := strings.TrimRight(lines[i], "\r")
		name, mediaType, kind := classifyBegin(line)
		if kind == notAMarker {
			rest = append(rest, lines[i])
			continue
		}
		// A header that does not parse still opened a fence. Its payload is
		// consumed below and the artifact is refused; see the file comment.
		headerErr := ""
		if kind == malformedMarker {
			headerErr = fmt.Sprintf(
				"the fence header %q is not \"<name> <mediaType>\"; both are required and "+
					"neither may contain a space", name)
			name, mediaType = "", ""
		}

		var payload strings.Builder
		closed, nested := false, false
		j := i + 1
		for ; j < len(lines); j++ {
			body := strings.TrimRight(lines[j], "\r")
			if body == artifactEnd {
				closed = true
				break
			}
			if _, _, k := classifyBegin(body); k != notAMarker {
				// A fence inside a fence. The payload has to encode it; guessing
				// which marker was meant would corrupt one of the two.
				nested = true
				break
			}

			payload.WriteString(body)
			payload.WriteString("\n")
		}

		switch {
		case headerErr != "":
			// Reported before the other cases: a fence whose header is
			// unreadable cannot be described any other way, and its size and
			// closure say nothing useful about evidence nobody can name.
			artifacts = append(artifacts, Artifact{Dropped: headerErr})
		case nested:
			artifacts = append(artifacts, Artifact{Name: name, MediaType: mediaType,
				Dropped: "a BEGIN marker appeared inside the payload; encode it or split the artifact"})
		case !closed:
			artifacts = append(artifacts, Artifact{Name: name, MediaType: mediaType,
				Dropped: "the fence was never closed; a partial payload is not evidence"})
		case payload.Len() > MaxArtifactBytes:
			artifacts = append(artifacts, Artifact{Name: name, MediaType: mediaType,
				SizeBytes: payload.Len(),
				Dropped: fmt.Sprintf("payload is %d bytes, over the %d-byte cap",
					payload.Len(), MaxArtifactBytes)})
		default:
			body := payload.String()
			sum := sha256.Sum256([]byte(body))
			artifacts = append(artifacts, Artifact{
				Name: name, MediaType: mediaType, Payload: body,
				SizeBytes: len(body), Digest: "sha256:" + hex.EncodeToString(sum[:]),
			})
		}
		// Everything from the BEGIN marker to wherever this block ended is
		// consumed, so no payload line can reach the metric scanner. An
		// unterminated fence therefore swallows the rest of stdout, which is
		// correct: those lines were the runner's payload, whatever it intended.
		//
		// On a nested marker j is left one BEFORE it, so the loop's own i++
		// lands back ON it and it is re-read as a fresh BEGIN. Skipping past it
		// would hand the inner block's payload to the metric scanner.
		if nested {
			j--
		}
		i = j
	}
	return artifacts, rest
}

// markerKind is how much of a BEGIN line could be understood.
type markerKind int

const (
	// notAMarker: an ordinary line of stdout.
	notAMarker markerKind = iota
	// validMarker: "<name> <mediaType>" both read.
	validMarker
	// malformedMarker: the line opens a fence but its header does not parse.
	// The distinction from notAMarker is the whole point — see the file
	// comment. The returned name is the raw header, clamped, so the refusal can
	// say which artifact was lost.
	malformedMarker
)

// classifyBegin reads "-----BEGIN BURNIN ARTIFACT <name> <mediaType>-----".
//
// Both fields are required. A nameless artifact cannot be referenced and an
// untyped one cannot be rendered, and inventing a default for either would put
// this package in the business of guessing what a runner meant.
func classifyBegin(line string) (name, mediaType string, kind markerKind) {
	if !strings.HasPrefix(line, artifactBegin) || !strings.HasSuffix(line, markerSuffix) {
		return "", "", notAMarker
	}
	rest := strings.TrimSuffix(strings.TrimPrefix(line, artifactBegin), markerSuffix)
	// The prefix stops at the keyword, so what follows must be empty or start
	// with a space. Otherwise "-----BEGIN BURNIN ARTIFACTS a b-----" would open
	// a fence, and a near-miss in a runner's own output would eat the lines
	// after it. A header that is merely EMPTY does open one: the line plainly
	// announces an artifact and simply failed to name it.
	if rest != "" && !isSpace(rest[0]) {
		return "", "", notAMarker
	}
	inner := strings.TrimSpace(rest)
	fields := strings.Fields(inner)
	if len(fields) != 2 {
		return truncateAtRune(inner, 120, "…"), "", malformedMarker
	}
	return fields[0], fields[1], validMarker
}

// truncateAtRune cuts to at most max bytes without splitting a rune.
//
// A clamped header is put in an operator-written message, and a message ending
// in half a rune renders as a replacement character that reads like corrupted
// evidence rather than like a clamp.
func truncateAtRune(s string, max int, suffix string) string {
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + suffix
}
