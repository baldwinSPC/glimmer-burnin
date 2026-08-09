// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

//go:build !linux

package main

import "fmt"

// O_DIRECT is a Linux flag. This file exists so the runner COMPILES on a
// developer's macOS laptop — where its own unit tests still run, since the
// arithmetic worth testing is platform-independent — while making it
// impossible to accidentally measure the page cache there: directSupported is
// false, and the runner skips rather than reporting a number.
const oDirect = 0

const directSupported = false

// freeBytes exists so the package compiles off Linux. The runner skips before
// reaching it — directSupported is false — so this is a compile-time stub and
// not a second implementation anybody could come to depend on.
func freeBytes(string) (uint64, error) {
	return 0, errNotLinux
}

var errNotLinux = fmt.Errorf("free space is only read on linux")
