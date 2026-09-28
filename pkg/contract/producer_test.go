package contract

import "testing"

func TestProducerKeepsAStampedVersionAndCommit(t *testing.T) {
	p := ProducerFromBuild("burnin", "v0.9.1", "abc123")
	if p.Name != "burnin" || p.Version != "v0.9.1" || p.Commit != "abc123" {
		t.Errorf("got %+v", p)
	}
}

// Unstamped, it says so rather than inventing a version.
func TestAnUnstampedProducerSaysDevel(t *testing.T) {
	if p := ProducerFromBuild("burnin", "", ""); p.Version == "" {
		t.Errorf("unstamped producer has an empty version: %+v", p)
	}
}
