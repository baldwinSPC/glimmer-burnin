package main

import "testing"

func TestBuildRevision(t *testing.T) {
	for _, tc := range []struct {
		name   string
		cfg    map[string]any
		want   string
		wantOK bool
	}{
		{
			name: "present",
			cfg: map[string]any{
				"config": map[string]any{
					"Labels": map[string]any{
						"org.opencontainers.image.revision": "abc123",
					},
				},
			},
			want:   "abc123",
			wantOK: true,
		},
		{
			name:   "no config",
			cfg:    map[string]any{},
			wantOK: false,
		},
		{
			name: "no labels",
			cfg: map[string]any{
				"config": map[string]any{},
			},
			wantOK: false,
		},
		{
			name: "empty revision — absence is not a declaration",
			cfg: map[string]any{
				"config": map[string]any{
					"Labels": map[string]any{
						"org.opencontainers.image.revision": "",
					},
				},
			},
			wantOK: false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := buildRevision(tc.cfg)
			if ok != tc.wantOK || got != tc.want {
				t.Errorf("buildRevision(%v) = (%q, %v), want (%q, %v)", tc.cfg, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}
