//go:build !unix

package localrun

// DeviceGroupGaps has no device modes to read off a non-unix host.
func DeviceGroupGaps(RunSpec) []string { return nil }
