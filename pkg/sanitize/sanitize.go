package sanitize

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Options configure one sanitisation.
type Options struct {
	// Hosts and Users are the inventory: names the result already records
	// (node names, a cluster name) and any the caller adds. A single-label
	// node name like "spark-043a" matches no content pattern, so without the
	// inventory it would pass straight through (§3a).
	Hosts []string
	Users []string
	// AllowDomains are registrable domains left untouched. Nil means
	// DefaultAllowDomains.
	AllowDomains []string
	// Salt is for tests only. Nil draws a fresh 32 bytes, which is the only
	// way a real sanitisation should run: a known salt makes every pseudonym
	// reversible by dictionary.
	Salt []byte
}

// Sanitizer holds one invocation's salt, inventory and pseudonym tables. Use
// one per output and discard it: reusing it across outputs links them.
type Sanitizer struct {
	salt   []byte
	hosts  map[string]bool
	users  map[string]bool
	allow  []string
	counts map[Category]int
	ipMap  map[string]string
	nextV4 int
	nextV6 int
}

// New returns a Sanitizer with a fresh salt unless one is given.
func New(o Options) (*Sanitizer, error) {
	s := &Sanitizer{
		salt:   o.Salt,
		hosts:  map[string]bool{},
		users:  map[string]bool{},
		allow:  o.AllowDomains,
		counts: map[Category]int{},
		ipMap:  map[string]string{},
	}
	if s.allow == nil {
		s.allow = DefaultAllowDomains
	}
	if s.salt == nil {
		s.salt = make([]byte, 32)
		if _, err := rand.Read(s.salt); err != nil {
			return nil, err
		}
	}
	for _, h := range o.Hosts {
		s.AddHost(h)
	}
	for _, u := range o.Users {
		if u = strings.TrimSpace(u); len(u) >= 2 {
			s.users[strings.ToLower(u)] = true
		}
	}
	return s, nil
}

// AddHost adds a name to the host inventory. Localhost, allowed domains, IP
// literals and one-character names are never inventory: the first two
// identify nothing, the third is the IP rules' job, and the last would
// rewrite every occurrence of a common letter.
func (s *Sanitizer) AddHost(h string) {
	h = strings.ToLower(strings.TrimSpace(h))
	if len(h) < 2 || isLocalhost(h) || domainAllowed(h, s.allow) || net.ParseIP(h) != nil {
		return
	}
	s.hosts[h] = true
}

// Counts returns how many values of each category were replaced.
func (s *Sanitizer) Counts() map[Category]int {
	out := make(map[Category]int, len(s.counts))
	for k, v := range s.counts {
		out[k] = v
	}
	return out
}

func (s *Sanitizer) pseudonym(cat Category, value string) string {
	m := hmac.New(sha256.New, s.salt)
	m.Write([]byte(string(cat) + ":" + value))
	return prefixes[cat] + "-" + hex.EncodeToString(m.Sum(nil))[:12]
}

// span is one identifying value found in a string.
type span struct {
	start, end int
	cat        Category
	inventory  bool
}

func isTokenByte(b byte) bool {
	return b == '_' || b == '-' || (b >= '0' && b <= '9') || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

// inventorySpans finds inventory names as whole tokens, or as a '-'-bounded
// run inside one (a run name like "burnin-spark-043a-7"). Token characters are
// [A-Za-z0-9_-], so "spark-10" never matches "spark-1".
func (s *Sanitizer) inventorySpans(str string) []span {
	var out []span
	lower := strings.ToLower(str)
	for i := 0; i < len(str); {
		if !isTokenByte(str[i]) {
			i++
			continue
		}
		j := i
		for j < len(str) && isTokenByte(str[j]) {
			j++
		}
		tok := lower[i:j]
		for _, set := range []struct {
			names map[string]bool
			cat   Category
		}{{s.hosts, CatHostname}, {s.users, CatUsername}} {
			for name := range set.names {
				for k := 0; k+len(name) <= len(tok); {
					idx := strings.Index(tok[k:], name)
					if idx < 0 {
						break
					}
					a, b := k+idx, k+idx+len(name)
					if (a == 0 || tok[a-1] == '-') && (b == len(tok) || tok[b] == '-') {
						out = append(out, span{i + a, i + b, set.cat, true})
					}
					k = a + 1
				}
			}
		}
		i = j
	}
	return out
}

var docV4 = []*net.IPNet{mustCIDR("192.0.2.0/24"), mustCIDR("198.51.100.0/24"), mustCIDR("203.0.113.0/24")}
var docV6 = mustCIDR("2001:db8::/32")
var privV4 = []*net.IPNet{mustCIDR("10.0.0.0/8"), mustCIDR("172.16.0.0/12"), mustCIDR("192.168.0.0/16"), mustCIDR("100.64.0.0/10"), mustCIDR("169.254.0.0/16")}
var localV6 = []*net.IPNet{mustCIDR("fe80::/10"), mustCIDR("fc00::/7")}

func mustCIDR(c string) *net.IPNet {
	_, n, err := net.ParseCIDR(c)
	if err != nil {
		panic(err)
	}
	return n
}

func inAny(ip net.IP, nets []*net.IPNet) bool {
	for _, n := range nets {
		if n.Contains(ip) {
			return true
		}
	}
	return false
}

// classifyIP returns the category of an IP literal, or "" when it is exempt:
// loopback, unspecified, broadcast and the documentation ranges identify
// nothing, and the documentation ranges are also where pseudonyms land.
func classifyIP(lit string) Category {
	ip := net.ParseIP(lit)
	if ip == nil || ip.IsLoopback() || ip.IsUnspecified() {
		return ""
	}
	if v4 := ip.To4(); v4 != nil && !strings.Contains(lit, ":") {
		if v4.Equal(net.IPv4bcast) || inAny(v4, docV4) {
			return ""
		}
		if inAny(v4, privV4) {
			return CatIPv4Private
		}
		return CatIPv4Public
	}
	if docV6.Contains(ip) {
		return ""
	}
	if inAny(ip, localV6) {
		return CatIPv6Local
	}
	return CatIPv6Public
}

// detect finds every identifying span in str, resolved the way the source
// ruleset resolves them: an inventory token is never split, overlapping
// candidates go to the leftmost-longest, and the entropy rule only claims
// what nothing more specific did.
func (s *Sanitizer) detect(str string) []span {
	inv := s.inventorySpans(str)
	var other []span
	add := func(re *regexp.Regexp, cat Category, group int) {
		for _, m := range re.FindAllStringSubmatchIndex(str, -1) {
			a, b := m[2*group], m[2*group+1]
			if a >= 0 {
				other = append(other, span{a, b, cat, false})
			}
		}
	}
	add(reHomePath, CatUsername, 1)
	add(reAuthHeader, CatSecret, 1)
	add(reMachineID, CatMachineID, 0)
	add(reUUID, CatUUID, 0)
	add(reMAC, CatMAC, 0)
	add(reRDMAGUID, CatRDMAGUID, 0)
	add(reSSHFP, CatSSHFingerprint, 0)
	add(reSSHHostKey, CatSSHHostKey, 0)
	for _, m := range reIPv4.FindAllStringIndex(str, -1) {
		if c := classifyIP(str[m[0]:m[1]]); c != "" {
			other = append(other, span{m[0], m[1], c, false})
		}
	}
	for _, m := range reIPv6Candidate.FindAllStringIndex(str, -1) {
		lit := str[m[0]:m[1]]
		if strings.Count(lit, ":") < 2 || net.ParseIP(lit) == nil {
			continue
		}
		if c := classifyIP(lit); c != "" {
			other = append(other, span{m[0], m[1], c, false})
		}
	}
	// Row 19 before row 18: a reference whose host is allowed is kept whole,
	// path and tag included, and that claim is what keeps the entropy rule off
	// a long repository path. Any other host has host and path pseudonymised
	// together and the tag kept.
	for _, m := range reRegistryRef.FindAllStringSubmatchIndex(str, -1) {
		host := str[m[2]:m[3]]
		if i := strings.IndexAny(host, ":/"); i >= 0 {
			host = host[:i]
		}
		if isLocalhost(host) || domainAllowed(host, s.allow) {
			other = append(other, span{m[0], m[1], catKeep, false})
		} else {
			other = append(other, span{m[2], m[3], CatHostname, false})
		}
	}
	for _, m := range reHostname.FindAllStringIndex(str, -1) {
		h := str[m[0]:m[1]]
		if isLocalhost(h) || domainAllowed(h, s.allow) {
			continue
		}
		other = append(other, span{m[0], m[1], CatHostname, false})
	}

	// An inventory token is never split: a candidate that partially overlaps
	// one is dropped. One that CONTAINS it competes below and wins as longer.
	kept := inv
	for _, o := range other {
		split := false
		for _, v := range inv {
			overlaps := o.start < v.end && v.start < o.end
			contains := o.start <= v.start && o.end >= v.end
			if overlaps && !contains {
				split = true
				break
			}
		}
		if !split {
			kept = append(kept, o)
		}
	}
	sort.SliceStable(kept, func(i, j int) bool {
		if kept[i].start != kept[j].start {
			return kept[i].start < kept[j].start
		}
		return kept[i].end-kept[i].start > kept[j].end-kept[j].start
	})
	var chosen []span
	end := -1
	for _, c := range kept {
		if c.start >= end {
			chosen = append(chosen, c)
			end = c.end
		}
	}

	// Entropy last, only over what nothing else claimed.
	for _, m := range reEntropyRun.FindAllStringIndex(str, -1) {
		if shannon(str[m[0]:m[1]]) < entropyMinBitsPerChar {
			continue
		}
		free := true
		for _, c := range chosen {
			if m[0] < c.end && c.start < m[1] {
				free = false
				break
			}
		}
		if free {
			chosen = append(chosen, span{m[0], m[1], CatSecret, false})
		}
	}
	sort.Slice(chosen, func(i, j int) bool { return chosen[i].start < chosen[j].start })
	return chosen
}

func shannon(s string) float64 {
	var freq [256]int
	for i := 0; i < len(s); i++ {
		freq[s[i]]++
	}
	var h float64
	n := float64(len(s))
	for _, f := range freq {
		if f > 0 {
			p := float64(f) / n
			h -= p * math.Log2(p)
		}
	}
	return h
}

func canonical(cat Category, v string) string {
	switch cat {
	case CatHostname, CatUUID, CatMachineID, CatRDMAGUID, CatUsername:
		return strings.ToLower(v)
	case CatMAC:
		return strings.ReplaceAll(strings.ToLower(v), "-", ":")
	}
	return v
}

func (s *Sanitizer) replace(cat Category, v string) (string, error) {
	s.counts[cat]++
	switch cat {
	case CatSecret:
		return maskSecret, nil
	case CatEnvDump:
		return maskEnv, nil
	case CatIPv4Private, CatIPv4Public, CatIPv6Local, CatIPv6Public:
		return s.pooledIP(v)
	}
	return s.pseudonym(cat, canonical(cat, v)), nil
}

// pooledIP maps an address into the documentation ranges (SNT-7), first seen
// first assigned, so one address keeps one stand-in throughout the output
// and two addresses never share one.
func (s *Sanitizer) pooledIP(lit string) (string, error) {
	ip := net.ParseIP(lit)
	key := ip.String()
	if got, ok := s.ipMap[key]; ok {
		return got, nil
	}
	var out string
	if ip.To4() != nil && !strings.Contains(lit, ":") {
		i := s.nextV4
		if i >= 3*254 {
			return "", fmt.Errorf("more than %d distinct IPv4 addresses; the documentation pool is exhausted", 3*254)
		}
		base := docV4[i/254].IP.To4()
		out = net.IPv4(base[0], base[1], base[2], byte(i%254+1)).String()
		s.nextV4++
	} else {
		s.nextV6++
		out = fmt.Sprintf("2001:db8::%x", s.nextV6)
	}
	s.ipMap[key] = out
	return out, nil
}

// Text sanitises free text: a runner's raw output, a log.
func (s *Sanitizer) Text(str string) (string, error) {
	spans := s.detect(str)
	if len(spans) == 0 {
		return str, nil
	}
	var b strings.Builder
	last := 0
	for _, sp := range spans {
		b.WriteString(str[last:sp.start])
		if sp.cat == catKeep {
			b.WriteString(str[sp.start:sp.end])
		} else {
			r, err := s.replace(sp.cat, str[sp.start:sp.end])
			if err != nil {
				return "", err
			}
			b.WriteString(r)
		}
		last = sp.end
	}
	b.WriteString(str[last:])
	return b.String(), nil
}

// Path rewrites inventory names in a relative path's components, and nothing
// else: the content rules would read "run.json" as a hostname, and a renamed
// file breaks the tools that read the directory.
func (s *Sanitizer) Path(rel string) (string, error) {
	parts := strings.Split(rel, "/")
	for i, p := range parts {
		spans := s.inventorySpans(p)
		if len(spans) == 0 {
			continue
		}
		var b strings.Builder
		last := 0
		for _, sp := range spans {
			if sp.start < last {
				continue
			}
			b.WriteString(p[last:sp.start])
			r, err := s.replace(sp.cat, p[sp.start:sp.end])
			if err != nil {
				return "", err
			}
			b.WriteString(r)
			last = sp.end
		}
		b.WriteString(p[last:])
		parts[i] = b.String()
	}
	return strings.Join(parts, "/"), nil
}

// exemptLeafKeys hold structural vocabulary, timestamps and content hashes:
// values that identify no machine and that a reader of the result needs
// verbatim. A commit or digest is also exactly what the entropy rule would
// otherwise mask.
var exemptLeafKeys = map[string]bool{
	"version": true, "reason": true, "phase": true, "sentAt": true,
	"startedAt": true, "finishedAt": true, "kind": true, "scope": true,
	"metric": true, "cause": true, "comparison": true, "applicability": true,
	"mediaType": true, "digest": true, "commit": true, "trigger": true,
	"role": true,
}

// exemptObjects are objects whose members are all kept. producer names the
// build, which is the provenance a shared result must keep.
var exemptObjects = map[string]bool{"producer": true, "summary": true}

// exemptStringArrays are arrays of metric names.
var exemptStringArrays = map[string]bool{"unmeasurable": true}

func isFiniteNumber(v string) bool {
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	return err == nil && !math.IsNaN(f) && !math.IsInf(f, 0)
}

// JSON sanitises one JSON document. Numbers are decoded as json.Number and
// re-emitted as written.
func (s *Sanitizer) JSON(data []byte) ([]byte, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	out, err := s.walk(v, "")
	if err != nil {
		return nil, err
	}
	// SetEscapeHTML(false): the masks are "<REDACTED-…>", and \u003c in a
	// document a person reads is noise. The encoder adds the trailing newline.
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (s *Sanitizer) walk(v any, key string) (any, error) {
	switch t := v.(type) {
	case map[string]any:
		if exemptObjects[key] {
			return t, nil
		}
		out := make(map[string]any, len(t))
		for k, val := range t {
			nk := k
			if key != "metrics" {
				// Keys carry identity too: an envelope's fingerprint is keyed
				// by node name. Metric names are the contract and are kept.
				var err error
				if nk, err = s.Text(k); err != nil {
					return nil, err
				}
			}
			if _, dup := out[nk]; dup {
				return nil, fmt.Errorf("two keys sanitise to %q; refusing to merge them", nk)
			}
			nv, err := s.walkMember(k, val, key == "metrics")
			if err != nil {
				return nil, err
			}
			out[nk] = nv
		}
		return out, nil
	case []any:
		out := make([]any, len(t))
		for i, e := range t {
			if str, ok := e.(string); ok && exemptStringArrays[key] {
				out[i] = str
				continue
			}
			ne, err := s.walk(e, key)
			if err != nil {
				return nil, err
			}
			out[i] = ne
		}
		return out, nil
	case string:
		return s.Text(t)
	}
	return v, nil
}

func (s *Sanitizer) walkMember(k string, val any, isMetric bool) (any, error) {
	if str, ok := val.(string); ok {
		switch {
		case isMetric && isFiniteNumber(str):
			return str, nil
		case isMetric && keySerial.MatchString(k) && str != "":
			// An identity metric derived from serials (nvmeSerialDigests,
			// #540): a stable digest no content rule recognises, so without
			// this it would link every shared result to the drive.
			return s.replace(CatSerial, str)
		case !isMetric && exemptLeafKeys[k]:
			return str, nil
		}
	}
	if !isMetric {
		switch {
		case keyEnv.MatchString(k):
			return s.maskEnv(val)
		case keySecret.MatchString(k):
			if str, ok := val.(string); ok && str != "" {
				return s.replace(CatSecret, str)
			}
		case keySerial.MatchString(k):
			if str, ok := val.(string); ok && str != "" {
				return s.replace(CatSerial, str)
			}
		case keyUsername.MatchString(k):
			if str, ok := val.(string); ok && str != "" {
				return s.replace(CatUsername, str)
			}
		case keySSID.MatchString(k):
			if str, ok := val.(string); ok && str != "" {
				return s.replace(CatWifiSSID, str)
			}
		case keyHostname.MatchString(k):
			if str, ok := val.(string); ok && str != "" && !isLocalhost(str) && !domainAllowed(str, s.allow) {
				return s.replace(CatHostname, str)
			}
		}
	}
	return s.walk(val, k)
}

func (s *Sanitizer) maskEnv(val any) (any, error) {
	switch t := val.(type) {
	case string:
		return s.replace(CatEnvDump, t)
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, v := range t {
			if envSafe[k] {
				out[k] = v
				continue
			}
			r, err := s.replace(CatEnvDump, fmt.Sprint(v))
			if err != nil {
				return nil, err
			}
			out[k] = r
		}
		return out, nil
	}
	return val, nil
}

var pseudonymShape = regexp.MustCompile(`^(serial|user|ssid|mid|uuid|mac|guid|ssh|host)-[0-9a-f]{12}$`)

// Residual re-runs every detector over sanitised output and reports what it
// finds. The caller must discard the output on any finding: fail-closed is
// the whole guarantee (SNT-9). It is not a proof — a secret shorter than the
// entropy rule's length that matches no shape is invisible to both passes.
func (s *Sanitizer) Residual(text string) []string {
	var out []string
	for _, sp := range s.detect(text) {
		v := text[sp.start:sp.end]
		if sp.cat == catKeep || pseudonymShape.MatchString(v) {
			continue
		}
		out = append(out, fmt.Sprintf("%s %q", sp.cat, v))
	}
	return out
}

// ResidualJSON is Residual over a sanitised JSON document, honouring the same
// exemptions the walk applied: a commit or digest is kept verbatim on purpose
// and would otherwise read as a high-entropy secret.
func (s *Sanitizer) ResidualJSON(data []byte) ([]string, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	var out []string
	var visit func(v any, key string, isMetric bool)
	visit = func(v any, key string, isMetric bool) {
		switch t := v.(type) {
		case map[string]any:
			if exemptObjects[key] {
				return
			}
			for k, val := range t {
				if key != "metrics" {
					out = append(out, s.Residual(k)...)
				}
				if str, ok := val.(string); ok {
					if (key == "metrics" && isFiniteNumber(str)) || (key != "metrics" && exemptLeafKeys[k]) ||
						str == maskEnv || str == maskSecret {
						continue
					}
				}
				visit(val, k, key == "metrics")
			}
		case []any:
			for _, e := range t {
				if _, ok := e.(string); ok && exemptStringArrays[key] {
					continue
				}
				visit(e, key, false)
			}
		case string:
			out = append(out, s.Residual(t)...)
		}
	}
	visit(v, "", false)
	return out, nil
}
