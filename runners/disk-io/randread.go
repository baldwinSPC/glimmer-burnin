// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"math/rand/v2"
	"os"
	"sync"
	"time"
)

// The random-read pass (#545): 4 KiB reads at uniformly random aligned offsets
// in the file this run wrote, with randReadDepth reads in flight, direct I/O.
//
// Sequential bandwidth is where a healthy drive and a failing one look most
// alike. Small random reads are where a degrading drive, or a controller that
// throttles when warm, shows first — as lost IOPS and a long tail. The method
// is fio's randread job (bs=4k, iodepth 32, norandommap, direct=1), ported
// without fio, which is GPL and cannot ship here: goroutines each issuing one
// pread at a time stand in for the queue depth.
//
// The tail comes from a per-worker histogram at one-microsecond resolution, so
// memory stays bounded at half a million reads a second. Reads during an
// initial ramp are issued but not counted, the way fio's ramp_time discards
// them: the first moments measure the queue filling, not the device.
const (
	randBlock        = 4096
	randHistMaxUs    = 20000 // reads slower than 20 ms land in one overflow bucket
	defaultRandDepth = 32
)

type randResult struct {
	ops      uint64        // reads completed after the ramp
	elapsed  time.Duration // from the end of the ramp to the last completion
	hist     []uint64      // hist[us] = reads that took us microseconds
	overflow uint64
	errors   int
	reached  bool // at least one read returned data: the device was reached
}

func (r randResult) iops() float64 {
	if r.elapsed <= 0 {
		return 0
	}
	return float64(r.ops) / r.elapsed.Seconds()
}

// p99Us is the 99th-percentile latency from the histogram, at 1 µs
// resolution. ok is false when nothing was counted, or when the 99th
// percentile lies in the overflow bucket — a tail past 20 ms is reported as
// unmeasured rather than as 20 ms.
func (r randResult) p99Us() (float64, bool) {
	total := r.overflow
	for _, c := range r.hist {
		total += c
	}
	if total == 0 {
		return 0, false
	}
	want := uint64(float64(total)*0.99 + 0.5)
	if want == 0 {
		want = 1
	}
	var cum uint64
	for us, c := range r.hist {
		cum += c
		if cum >= want {
			return float64(us), true
		}
	}
	return 0, false
}

// splitReadWindow gives the sequential read at most half of what the write
// left, so the random pass always keeps a share. Sequential reads normally end
// at EOF long before that, and the random pass then gets the rest.
func splitReadWindow(now, deadline time.Time) time.Time {
	if !deadline.After(now) {
		return now
	}
	return now.Add(deadline.Sub(now) / 2)
}

func randReadPass(path string, fileBytes uint64, depth int, deadline time.Time) (randResult, error) {
	blocks := int64(fileBytes / randBlock)
	if blocks == 0 {
		return randResult{}, fmt.Errorf("the file holds no whole %d-byte block to read", randBlock)
	}
	window := time.Until(deadline)
	if window <= 0 {
		return randResult{}, nil
	}
	ramp := window / 10
	if ramp > 2*time.Second {
		ramp = 2 * time.Second
	}
	start := time.Now()
	measureFrom := start.Add(ramp)

	type workerOut struct {
		ops, overflow uint64
		hist          []uint64
		last          time.Time
		reached       bool
		err           error
	}
	outs := make([]workerOut, depth)
	var wg sync.WaitGroup
	for w := 0; w < depth; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			o := &outs[w]
			o.hist = make([]uint64, randHistMaxUs+1)
			f, err := openFile(path, os.O_RDONLY, 0)
			if err != nil {
				o.err = err
				return
			}
			defer f.Close()
			buf := alignedBuffer(randBlock)
			rng := rand.New(rand.NewPCG(uint64(w)+1, uint64(start.UnixNano())))
			for {
				t0 := time.Now()
				if !t0.Before(deadline) {
					return
				}
				off := rng.Int64N(blocks) * randBlock
				n, err := f.ReadAt(buf, off)
				t1 := time.Now()
				if err != nil || n != randBlock {
					if err == nil {
						err = fmt.Errorf("short read of %d bytes", n)
					}
					o.err = fmt.Errorf("random read at offset %d: %w", off, err)
					return
				}
				o.reached = true
				if t0.Before(measureFrom) {
					continue
				}
				o.ops++
				o.last = t1
				us := t1.Sub(t0).Microseconds()
				if us > randHistMaxUs {
					o.overflow++
				} else {
					o.hist[us]++
				}
			}
		}(w)
	}
	wg.Wait()

	res := randResult{hist: make([]uint64, randHistMaxUs+1)}
	var last time.Time
	var firstErr error
	for _, o := range outs {
		res.ops += o.ops
		res.overflow += o.overflow
		res.reached = res.reached || o.reached
		for i, c := range o.hist {
			res.hist[i] += c
		}
		if o.last.After(last) {
			last = o.last
		}
		if o.err != nil {
			res.errors++
			if firstErr == nil {
				firstErr = o.err
			}
		}
	}
	if last.After(measureFrom) {
		res.elapsed = last.Sub(measureFrom)
	}
	return res, firstErr
}
