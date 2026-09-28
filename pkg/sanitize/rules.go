// Package sanitize produces a copy of a burn-in result that can leave the
// site: machine identity replaced by pseudonyms, network addresses mapped into
// documentation ranges, secrets masked, and every measurement left exactly as
// it was (#548).
//
// The rules are a port of an independent GB10 acceptance toolkit's sanitizer
// ruleset, which was designed and adversarially reviewed before it was built.
// Row numbers in the comments below are that ruleset's, so the two can be read
// side by side.
//
// Four properties hold everything up, and none may be relaxed:
//
//   - PSEUDONYMS ARE UNLINKABLE ACROSS OUTPUTS. Each is an HMAC-SHA256 under a
//     32-byte salt drawn fresh per invocation and never written anywhere, with
//     the category in the message so two categories cannot collide on one raw
//     value. Within one output the same value always maps to the same
//     pseudonym, so the result still reads coherently.
//   - NUMBERS ARE NEVER TOUCHED. A measurement is the point of sharing a
//     result, so a JSON number, and a JSON string that parses as a finite
//     number (every metric value in an envelope), is left byte-for-byte.
//   - IT FAILS CLOSED. A residual scan re-runs every detector over the output,
//     and any hit means no output at all.
//   - A SANITIZED RESULT IS TERMINAL. Its identity is a pseudonym, so "is this
//     the same hardware?" can no longer be answered honestly, and nothing that
//     compares results may accept one.
package sanitize

import (
	"regexp"
	"strings"
)

// Category is what kind of identifying value was found.
type Category string

const (
	CatSerial         Category = "serial"
	CatUsername       Category = "username"
	CatWifiSSID       Category = "wifi_ssid"
	CatEnvDump        Category = "env_dump"
	CatSecret         Category = "secret"
	CatMachineID      Category = "machine_id"
	CatUUID           Category = "uuid"
	CatMAC            Category = "mac"
	CatRDMAGUID       Category = "rdma_guid"
	CatSSHFingerprint Category = "ssh_fingerprint"
	CatSSHHostKey     Category = "ssh_host_key"
	CatIPv6Local      Category = "ipv6_link_local_or_ula"
	CatIPv4Private    Category = "ipv4_private"
	CatIPv4Public     Category = "ipv4_public"
	CatIPv6Public     Category = "ipv6_public"
	CatHostname       Category = "hostname"

	// catKeep claims a span that must stay verbatim, such as a public registry
	// reference, so no weaker rule can rewrite part of it.
	catKeep Category = ""
)

// prefixes name the pseudonym namespace of each pseudonymised category. Both
// SSH categories share one namespace, as in the source ruleset.
var prefixes = map[Category]string{
	CatSerial:         "serial",
	CatUsername:       "user",
	CatWifiSSID:       "ssid",
	CatMachineID:      "mid",
	CatUUID:           "uuid",
	CatMAC:            "mac",
	CatRDMAGUID:       "guid",
	CatSSHFingerprint: "ssh",
	CatSSHHostKey:     "ssh",
	CatHostname:       "host",
}

const (
	maskEnv    = "<REDACTED-ENV>"
	maskSecret = "<REDACTED-SECRET>"
)

// Key-triggered rules: the JSON key names the category whatever the value's
// shape, and the whole string value is replaced.
var (
	keySerial   = regexp.MustCompile(`(?i)serial`)                                            // row 3
	keyUsername = regexp.MustCompile(`(?i)^(user|username|operator|promoter)$`)               // row 4
	keySSID     = regexp.MustCompile(`(?i)(^|[^b])ssid$`)                                     // row 5: never bssid
	keyEnv      = regexp.MustCompile(`(?i)(^env$|environ|environment_dump)`)                  // row 6
	keySecret   = regexp.MustCompile(`(?i)(authorization|token|api_key|secret|password)`)     // row 7
	keyHostname = regexp.MustCompile(`(?i)((hostname|fqdn|nodename)$|^(host|control_host)$)`) // row 18
)

// envSafe are the env-dump keys whose values are kept (row 6).
var envSafe = map[string]bool{"PATH": true, "LANG": true, "TERM": true, "PWD": true, "SHELL": true}

// Content rules, matched over every string.
var (
	reHomePath      = regexp.MustCompile(`/(?:home|Users)/([^/\s"']+)`)                                                                    // row 4
	reAuthHeader    = regexp.MustCompile(`(?im)^Authorization:[ \t]*(\S.*)$`)                                                              // row 7
	reMachineID     = regexp.MustCompile(`\b[0-9a-fA-F]{32}\b`)                                                                            // row 8
	reUUID          = regexp.MustCompile(`\b(?:[A-Za-z]+-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b`) // row 9
	reMAC           = regexp.MustCompile(`\b[0-9a-fA-F]{2}(?:[:-][0-9a-fA-F]{2}){5}\b`)                                                    // row 10
	reRDMAGUID      = regexp.MustCompile(`\b[0-9a-fA-F]{4}:[0-9a-fA-F]{4}:[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\b`)                                // row 11
	reSSHFP         = regexp.MustCompile(`\bSHA256:[A-Za-z0-9+/]{43}`)                                                                     // row 12
	reSSHHostKey    = regexp.MustCompile(`\b(?:ssh-ed25519|ssh-rsa|ecdsa-sha2-\S+) [A-Za-z0-9+/]{40,}={0,2}`)                              // row 13
	reIPv4          = regexp.MustCompile(`\b(?:\d{1,3}\.){3}\d{1,3}\b`)                                                                    // rows 15-16
	reIPv6Candidate = regexp.MustCompile(`[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}`)                                                     // rows 14, 17
	reHostname      = regexp.MustCompile(`\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b`)                           // row 18
	// row 19: an image reference. Host (with an optional port) and path are one
	// span; a trailing :tag or @sha256:digest is left as written.
	reRegistryRef = regexp.MustCompile(`\b((?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?::\d+)?(?:/[a-z0-9._-]+)+)((?::[A-Za-z0-9_.-]+)?(?:@sha256:[0-9a-f]{64})?)`)
	reEntropyRun  = regexp.MustCompile(`[A-Za-z0-9+/=_-]{24,}`) // row 7, §4
)

// entropyMinBitsPerChar is the Shannon-entropy threshold of the
// last-evaluated secret rule (§4).
const entropyMinBitsPerChar = 3.5

// DefaultAllowDomains are registrable domains whose hosts are public,
// non-identifying infrastructure: container registries, package sources, and
// the vendor and API domains a result legitimately names.
var DefaultAllowDomains = []string{
	"nvcr.io", "docker.io", "ghcr.io", "quay.io", "gcr.io", "public.ecr.aws",
	"pypi.org", "pythonhosted.org", "ubuntu.com", "github.com",
	"k8s.io", "kubernetes.io", "nvidia.com", "amd.com", "glimmer.ai",
}

// isLocalhost reports the names that never identify anything.
func isLocalhost(h string) bool {
	h = strings.ToLower(h)
	return h == "localhost" || h == "localhost.localdomain"
}

func domainAllowed(host string, allow []string) bool {
	host = strings.ToLower(strings.TrimSuffix(host, "."))
	for _, d := range allow {
		d = strings.ToLower(d)
		if host == d || strings.HasSuffix(host, "."+d) {
			return true
		}
	}
	return false
}
