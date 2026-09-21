#!/usr/bin/env bash
# hwverify.sh — the hardware verification capture for clockprobe-rocm (#320).
#
# SPDX-License-Identifier: Apache-2.0
# Copyright the Glimmer authors.
#
# WHY THIS IS A SCRIPT AND NOT A SESSION.
#
# clockprobe-rocm shipped unverified on purpose: its sysfs parsing and its
# judgement truth table are unit-tested, its HIP load path is compile-verified,
# and its floor defaults are conservative CHOICES rather than measurements (see
# README "Status — read before pinning"). The pass that changes that has to
# produce EVIDENCE THE REPOSITORY CAN HOLD, not an observation somebody made
# once over SSH: a fixture in sysfs_clocks_test.cc outlives the session, and a
# number in the README can be re-derived from the capture it came from.
#
# So this script's output is the deliverable. It reads files, it samples, and it
# writes a directory. It decides nothing that the runner decides.
#
# EACH CHECK NAMES WHAT WOULD FALSIFY IT, BEFORE THE RUN. That ordering is the
# point. A capture read after the fact will confirm whatever the reader already
# believes — the discriminating outcome has to be written down while it can
# still come out the other way.
#
# Usage:
#   ./hwverify.sh idle              # phase A: idle capture, no load, no container
#   ./hwverify.sh load [SECONDS]    # phase B: sample at 1 Hz (default 120) while
#                                   #          a load runs — start the load first
#   ./hwverify.sh report            # read the capture back and judge it
#
# Output lands in ./clockprobe-rocm-capture-<host>-<stamp>/ and is meant to be
# committed as fixture material, so it contains no credentials and no hostnames
# beyond the one in the directory name.

set -u

OUT_ROOT="${BURNIN_CAPTURE_DIR:-$PWD}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# dump copies a file verbatim into the capture, recording its absence as a fact
# rather than skipping it. An absent file is a finding: the parser's selection
# rule requires pp_dpm_sclk, and "no such file" is the answer to a question.
dump() {
	local src="$1" dst="$2"
	mkdir -p "$(dirname "$dst")"
	if [ -r "$src" ]; then
		cat "$src" > "$dst" 2>/dev/null || echo "<unreadable: $src>" > "$dst"
	elif [ -e "$src" ]; then
		echo "<exists but not readable as $(id -un): $src>" > "$dst"
	else
		echo "<absent: $src>" > "$dst"
	fi
}

say() { printf '%s\n' "$*"; }
hdr() { printf '\n=== %s ===\n' "$*"; }

# cards lists every amdgpu card sysfs directory, applying the SAME selection the
# runner applies (AllAmdgpuCards): cardN only, no connector entries.
cards() {
	local p name
	for p in /sys/class/drm/card*; do
		[ -d "$p" ] || continue
		name="$(basename "$p")"
		case "$name" in
		*-*) continue ;;  # card0-DP-1 style connector entries
		esac
		[ -d "$p/device" ] || continue
		printf '%s\n' "$p"
	done
}

# ---------------------------------------------------------------------------
# phase A — idle capture
# ---------------------------------------------------------------------------

phase_idle() {
	local dir="$OUT_ROOT/clockprobe-rocm-capture-$HOST-$STAMP"
	mkdir -p "$dir"
	say "capture -> $dir"

	{
		echo "capture: clockprobe-rocm hardware verification (#320)"
		echo "phase: idle"
		echo "host: $HOST"
		echo "utc: $STAMP"
		echo "user: $(id -un) uid=$(id -u) groups=$(id -Gn 2>/dev/null)"
		echo "kernel: $(uname -r)"
		echo "kernel-full: $(uname -a)"
	} > "$dir/capture.meta"

	# --- A0: the platform, for the record ---------------------------------
	dump /etc/os-release "$dir/host/os-release"
	dump /proc/cpuinfo   "$dir/host/cpuinfo"
	dump /proc/meminfo   "$dir/host/meminfo"
	{ lsmod 2>/dev/null | grep -i amdgpu; } > "$dir/host/lsmod-amdgpu" 2>/dev/null
	{ modinfo amdgpu 2>/dev/null | head -20; } > "$dir/host/modinfo-amdgpu" 2>/dev/null
	{ ls -la /sys/class/drm/ 2>/dev/null; } > "$dir/host/sysfs-class-drm.ls"

	# --- A1..A5: per card --------------------------------------------------
	local n=0 p name cdev hw
	while read -r p; do
		name="$(basename "$p")"
		cdev="$p/device"
		local cd="$dir/cards/$name"
		mkdir -p "$cd"
		n=$((n + 1))

		# A1 — IDENTITY. Is this an AMD card at all, and which one?
		#
		# Discriminating check: vendor must read 0x1002. AllAmdgpuCards gates on
		# exactly this string; anything else and the runner sees no card on a
		# machine that plainly has one.
		dump "$cdev/vendor"   "$cd/vendor"
		dump "$cdev/device"   "$cd/device"
		dump "$cdev/revision" "$cd/revision"
		dump "$cdev/uevent"   "$cd/uevent"
		( cd "$cdev" 2>/dev/null && readlink -f . ) > "$cd/pci-address.resolved" 2>/dev/null

		# A2 — THE DPM LADDER. The single most load-bearing parse in the runner.
		#
		# ParseDpmSclk expects "N: <mhz>Mhz" per line, " *" on the active level,
		# unit matched case-insensitively. The TOP level is the judgement
		# denominator for sustainedClockPct.
		#
		# Discriminating check, three ways it can come out:
		#   CONFIRMS   — lines like "0: 400Mhz" / "2: 2900Mhz *", top is a
		#                plausible gfx1151 ceiling, exactly one line starred.
		#   FALSIFIES  — a header line, a different unit spelling, a range form,
		#                or NO star at all. Any of those and ParseDpmSclk either
		#                errors (-> skip, exit 2, every node unjudged) or reads a
		#                ladder whose top is not the rated clock.
		#   WATCH FOR  — a SINGLE-LEVEL ladder. On an APU whose ladder collapses
		#                to one entry, current == rated == 100% always, and
		#                sustainedClockPct can never fail. That is not a pass,
		#                it is a denominator that cannot discriminate, and it
		#                would have to be reported rather than gated on.
		dump "$cdev/pp_dpm_sclk" "$cd/pp_dpm_sclk"
		dump "$cdev/pp_dpm_mclk" "$cd/pp_dpm_mclk"
		dump "$cdev/pp_dpm_fclk" "$cd/pp_dpm_fclk"
		dump "$cdev/power_dpm_force_performance_level" "$cd/power_dpm_force_performance_level"
		dump "$cdev/pp_od_clk_voltage" "$cd/pp_od_clk_voltage"

		# A3 — UTILIZATION. What makes "busy while slow" observable at all.
		#
		# Discriminating check: reads an integer 0..100 at idle. If the file is
		# absent, idle_clock_lock_suspected can never be `true` — it goes
		# `unknown`, which the runner is careful never to read as all-clear, but
		# it means the ROCm#5750 signature is undetectable on this part and the
		# README has to say so.
		dump "$cdev/gpu_busy_percent" "$cd/gpu_busy_percent.idle"
		dump "$cdev/mem_busy_percent" "$cd/mem_busy_percent.idle"

		# A4 — HWMON. Temperature buys the lenient floor; power is evidence.
		#
		# Discriminating check: temp1_input present and in MILLIDEGREES (a
		# plausible idle read is 30000-60000, i.e. 30-60 C). A value near 45
		# rather than 45000 means the driver changed units and the 90 C thermal
		# threshold would never trigger — healthy hot parts then fail the strict
		# 60% floor instead of getting the 40% they are owed.
		#
		# Also captured: EVERY *_input in the hwmon dir. The runner reads three
		# files; the capture takes the whole surface, because the next AMD
		# runner (#317's fault counters) reads from here too and a second trip
		# to the hardware to get a file we could have taken now is the waste
		# this script exists to prevent.
		if [ -d "$cdev/hwmon" ]; then
			for hw in "$cdev"/hwmon/hwmon*; do
				[ -d "$hw" ] || continue
				local hn
				hn="$(basename "$hw")"
				mkdir -p "$cd/hwmon/$hn"
				dump "$hw/name" "$cd/hwmon/$hn/name"
				local f
				for f in "$hw"/*_input "$hw"/*_label "$hw"/*_crit "$hw"/*_max "$hw"/power1_cap*; do
					[ -e "$f" ] || continue
					dump "$f" "$cd/hwmon/$hn/$(basename "$f")"
				done
				( ls -la "$hw" ) > "$cd/hwmon/$hn/.ls" 2>/dev/null
			done
		else
			echo "<absent: $cdev/hwmon>" > "$cd/hwmon.absent"
		fi

		# A6 — RAS, for #317/#259. Present on an APU? Almost certainly not, and
		# THAT IS THE POINT: #259 needs a part with no ras/ directory to prove
		# the n/a sentinel branch is reachable rather than dead reasoning. A
		# Halo is very likely that part, which makes this two lines of capture
		# that close a bullet on another ticket.
		if [ -d "$cdev/ras" ]; then
			mkdir -p "$cd/ras"
			local r
			for r in "$cdev"/ras/*; do
				[ -f "$r" ] || continue
				dump "$r" "$cd/ras/$(basename "$r")"
			done
		else
			echo "<absent: $cdev/ras>" > "$cd/ras.absent"
		fi
		dump "$cdev/pcie_replay_count" "$cd/pcie_replay_count"
		dump "$cdev/pp_features"       "$cd/pp_features"
		dump "$cdev/serial_number"     "$cd/serial_number"
		dump "$cdev/unique_id"         "$cd/unique_id"
		dump "$cdev/mem_info_vram_total" "$cd/mem_info_vram_total"
		dump "$cdev/mem_info_gtt_total"  "$cd/mem_info_gtt_total"
	done <<< "$(cards)"

	echo "$n" > "$dir/cards.count"

	# --- A5: DEVICE ACCESS. The uid 65532 question, answered as data. ------
	#
	# Discriminating check: the image runs as uid 65532 with no supplemental
	# groups. If /dev/kfd is rw for group `render` and mode 0660, that uid
	# CANNOT open it and the runner errors on every node until the pod gets
	# runAsUser: 0 or a supplementalGroup. The README currently says "may need"
	# — this capture is what replaces the hedge with the group number.
	{
		ls -la /dev/kfd 2>&1
		ls -la /dev/dri/ 2>&1
		echo "--- group ids ---"
		getent group render video 2>/dev/null
		echo "--- what uid 65532 would see ---"
		echo "kfd mode: $(stat -c '%a %U:%G' /dev/kfd 2>/dev/null || echo '<absent>')"
	} > "$dir/device-access.txt" 2>&1

	# --- ROCm presence, for context only -----------------------------------
	{
		echo "rocm-smi: $(command -v rocm-smi || echo '<not installed>')"
		echo "amd-smi:  $(command -v amd-smi  || echo '<not installed>')"
		echo "--- rocm-smi showclocks (if present) ---"
		rocm-smi --showclocks 2>&1 | head -40
		echo "--- amd-smi static (if present) ---"
		amd-smi static 2>&1 | head -40
	} > "$dir/rocm-tools.txt" 2>&1

	say ""
	say "idle capture complete: $n amdgpu card(s)"
	say "next: start a load, then run:  $0 load 120"
	printf '%s\n' "$dir" > "$OUT_ROOT/.last-capture"
}

# ---------------------------------------------------------------------------
# phase B — loaded sampling
# ---------------------------------------------------------------------------

phase_load() {
	local secs="${1:-120}"
	local dir
	dir="$(cat "$OUT_ROOT/.last-capture" 2>/dev/null)"
	if [ -z "$dir" ] || [ ! -d "$dir" ]; then
		say "no idle capture found — run '$0 idle' first" >&2
		exit 1
	fi
	say "sampling ${secs}s at 1 Hz -> $dir/samples.tsv"
	say "(start the load BEFORE this if you have not — the warmup claim depends on it)"

	# B1 — THE WARMUP WINDOW and B2 — THE SUSTAINED CLOCK.
	#
	# The runner discards a warmup window then takes the MEAN of the rest. Two
	# claims ride on this sample series and neither has been observed:
	#
	#   CONFIRMS   — clock climbs to a plateau within the warmup window and
	#                stays there. The plateau / ladder-top ratio is the number
	#                the README owes: clockprobe records 69.9% for GB10, and
	#                this is the AMD equivalent.
	#   FALSIFIES  — the ramp takes LONGER than the warmup window, which makes
	#                the mean include ramp samples and understate a healthy
	#                part. That is a floor that fails good hardware, and the
	#                fix is the warmup constant, not the floor.
	#   ALSO WATCH — a part that plateaus BELOW 60% while cool and busy is the
	#                ROCm#5750 signature itself. If a healthy Halo does that,
	#                the 60 default is simply wrong for this part and must be
	#                calibrated down from measurement, not defended.
	#
	# Sampling every file the runner samples, every second, with a timestamp:
	# the mean is recomputable from the capture, so a disagreement about the
	# number is settled by arithmetic instead of another trip to the hardware.
	{
		printf 'epoch_ms\tcard\tsclk_mhz\tsclk_rated_mhz\tstarred_level\tbusy_pct\ttemp_mc\tpower_uw\n'
	} > "$dir/samples.tsv"

	local end=$(( $(date +%s) + secs ))
	local p name cdev hw line cur rated star busy temp pw ms
	while [ "$(date +%s)" -lt "$end" ]; do
		while read -r p; do
			name="$(basename "$p")"
			cdev="$p/device"
			[ -r "$cdev/pp_dpm_sclk" ] || continue

			cur=""; rated=""; star=""
			while IFS= read -r line; do
				# "2: 2900Mhz *"  ->  level 2, 2900, starred
				local lvl val
				lvl="${line%%:*}"
				val="$(printf '%s' "${line#*:}" | tr -d ' ' | sed 's/[Mm][Hh][Zz].*//')"
				[ -n "$val" ] || continue
				rated="$val"                       # last line wins == top of ladder
				case "$line" in
				*\*) cur="$val"; star="$(printf '%s' "$lvl" | tr -d ' ')" ;;
				esac
			done < "$cdev/pp_dpm_sclk"

			busy="$(cat "$cdev/gpu_busy_percent" 2>/dev/null)"
			temp=""; pw=""
			for hw in "$cdev"/hwmon/hwmon*; do
				[ -d "$hw" ] || continue
				temp="$(cat "$hw/temp1_input" 2>/dev/null)"
				pw="$(cat "$hw/power1_average" 2>/dev/null)"
				[ -n "$pw" ] || pw="$(cat "$hw/power1_input" 2>/dev/null)"
				break
			done

			ms="$(date +%s%3N 2>/dev/null || echo "$(date +%s)000")"
			printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
				"$ms" "$name" "${cur:-NA}" "${rated:-NA}" "${star:-NA}" \
				"${busy:-NA}" "${temp:-NA}" "${pw:-NA}" >> "$dir/samples.tsv"
		done <<< "$(cards)"
		sleep 1
	done

	say "sampled $(( $(wc -l < "$dir/samples.tsv") - 1 )) rows -> $dir/samples.tsv"
	say "next: $0 report"
}

# ---------------------------------------------------------------------------
# report — read the capture back, judge nothing the runner judges
# ---------------------------------------------------------------------------

phase_report() {
	local dir
	dir="$(cat "$OUT_ROOT/.last-capture" 2>/dev/null)"
	if [ -z "$dir" ] || [ ! -d "$dir" ]; then
		say "no capture found — run '$0 idle' first" >&2
		exit 1
	fi

	hdr "capture"
	cat "$dir/capture.meta"

	hdr "A1 identity / A2 ladder"
	local cd name v
	for cd in "$dir"/cards/*; do
		[ -d "$cd" ] || continue
		name="$(basename "$cd")"
		v="$(cat "$cd/vendor" 2>/dev/null)"
		say "$name: vendor=$v device=$(cat "$cd/device" 2>/dev/null)"
		case "$v" in
		0x1002) say "  [OK]   vendor is AMD — AllAmdgpuCards selects this card" ;;
		*)      say "  [FAIL] vendor is not 0x1002 — the runner will not see this card" ;;
		esac
		say "  pp_dpm_sclk:"
		sed 's/^/    /' "$cd/pp_dpm_sclk" 2>/dev/null
		local levels starred
		levels="$(grep -c 'z' "$cd/pp_dpm_sclk" 2>/dev/null || echo 0)"
		starred="$(grep -c '\*' "$cd/pp_dpm_sclk" 2>/dev/null || echo 0)"
		say "  levels=$levels starred=$starred"
		[ "$levels" -le 1 ] && say "  [WATCH] single-level ladder: sustainedClockPct cannot discriminate"
		[ "$starred" -ne 1 ] && say "  [FAIL]  expected exactly one starred level; ParseDpmSclk needs it for current"
	done

	hdr "A4 hwmon units"
	for cd in "$dir"/cards/*; do
		[ -d "$cd" ] || continue
		local t
		t="$(cat "$cd"/hwmon/*/temp1_input 2>/dev/null | head -1)"
		if [ -z "$t" ]; then
			say "$(basename "$cd"): [FAIL] no temp1_input — thermal leniency unreachable, hot healthy parts fail the strict floor"
		elif [ "$t" -gt 1000 ] 2>/dev/null; then
			say "$(basename "$cd"): [OK]   temp1_input=$t millidegrees ($((t / 1000)) C)"
		else
			say "$(basename "$cd"): [FAIL] temp1_input=$t looks like DEGREES, not millidegrees — the 90 C threshold would never fire"
		fi
	done

	hdr "A5 device access as uid 65532"
	grep -E 'kfd mode|^crw' "$dir/device-access.txt" 2>/dev/null | sed 's/^/  /'
	say "  -> if the mode is 0660 and the group is not one uid 65532 holds,"
	say "     the pod needs runAsUser: 0 or that supplementalGroup. Record the number."

	hdr "A6 ras/ presence (closes a #259 bullet)"
	for cd in "$dir"/cards/*; do
		[ -d "$cd" ] || continue
		if [ -f "$cd/ras.absent" ]; then
			say "$(basename "$cd"): no ras/ — this part is the n/a-sentinel case #259 needs to prove reachable"
		else
			say "$(basename "$cd"): ras/ present — capture holds the real block files"
		fi
	done

	hdr "B sustained clock"
	if [ ! -f "$dir/samples.tsv" ]; then
		say "  no samples yet — run '$0 load 120' with a load running"
	else
		awk -F'\t' 'NR>1 && $3!="NA" && $4!="NA" && $4>0 {
			n++; pct=100*$3/$4; sum+=pct
			if (min=="" || pct<min) min=pct
			if (pct>max) max=pct
			if ($6!="NA") { bn++; bsum+=$6 }
			if ($7!="NA") { tn++; tsum+=$7; if ($7>tmax) tmax=$7 }
		} END {
			if (n==0) { print "  no usable samples"; exit }
			printf "  samples=%d  clock%% mean=%.1f min=%.1f max=%.1f\n", n, sum/n, min, max
			if (bn) printf "  mean busy=%.0f%%\n", bsum/bn
			if (tn) printf "  mean temp=%.1f C   peak=%.1f C\n", (tsum/tn)/1000, tmax/1000
			printf "\n  against the shipped defaults (floor 60, thermal 90 C -> 40):\n"
			m=sum/n
			if (m>=60) printf "    mean %.1f%% clears the 60%% floor\n", m
			else printf "    mean %.1f%% is BELOW the 60%% floor — calibrate from this, do not defend the default\n", m
			if (tn && tmax/1000>=90) printf "    peak %.1f C is at/over the 90 C thermal threshold: the lenient 40%% floor applies\n", tmax/1000
		}' "$dir/samples.tsv"
		say ""
		say "  warmup: first and last 10 rows (does the ramp finish inside the warmup window?)"
		{ head -11 "$dir/samples.tsv" | tail -10; echo "  ..."; tail -10 "$dir/samples.tsv"; } | sed 's/^/    /'
	fi

	hdr "capture directory"
	say "$dir"
	say "tar it back for fixtures:  tar czf capture.tgz -C $(dirname "$dir") $(basename "$dir")"
}

case "${1:-}" in
idle)   phase_idle ;;
load)   phase_load "${2:-120}" ;;
report) phase_report ;;
*)
	say "usage: $0 {idle|load [seconds]|report}"
	say ""
	say "  idle    capture the sysfs surface with no load (no container needed)"
	say "  load    sample at 1 Hz while a load runs — start the load first"
	say "  report  read the capture back and name what confirms or falsifies"
	exit 2
	;;
esac
