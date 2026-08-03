#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright the Glimmer authors.
#
# Remove linux-rdma/perftest's build dependency on pciutils (libpci).
#
# ── WHY THIS EXISTS ───────────────────────────────────────────────────────────
#
# perftest itself is dual-licensed GPL-2.0-only OR BSD-2-Clause and this project
# consumes it under the BSD option.  But since the PCIe relaxed-ordering check
# was added, its configure REQUIRES pciutils on every non-FreeBSD platform:
#
#     if [test $IS_FREEBSD = no]; then
#         AC_CHECK_HEADERS([pci/pci.h],,[AC_MSG_ERROR([pciutils header files not found...])])
#         AC_CHECK_LIB([pci], [pci_init], [LIBPCI=-lpci], AC_MSG_ERROR([libpci not found]))
#     fi
#
# and there is no --without-pciutils.  pciutils/libpci is GPL-2.0-or-later.
# Linking it would put a GPL library in a published runner image, which this
# project's licensing rules forbid outright — see CLAUDE.md.  So the choice is
# not "patch or don't patch", it is "patch or do not ship an RDMA runner".
#
# ── WHY REMOVING IT IS CORRECT, NOT MERELY CONVENIENT ─────────────────────────
#
# The only thing libpci is used for is check_pcie_relaxed_ordering_compliant(),
# which walks the PCI bus looking for ONE Intel CPU family with a known
# relaxed-ordering erratum (vendor 0x8086, device 0x2f01 and 0x6f01-0x6f0e — the
# Haswell/Broadwell-EP root complexes of
# https://lore.kernel.org/patchwork/patch/820922/).  It returns true for
# everything else.  On any non-x86 host — every aarch64 accelerator node this
# runner targets — the scan CANNOT find such a device, so "return true" is the
# same answer the scan would give, obtained without the GPL dependency.
#
# This deliberately does NOT disable HAVE_RO.  Doing so would be the easier
# patch, but it also drops IBV_ACCESS_RELAXED_ORDERING from the memory region,
# which changes the number being measured.  A burn-in runner may not quietly
# alter its own measurement to simplify its build.
#
# THE COST, STATED PLAINLY: on an affected Intel host this image loses perftest's
# "WARNING: CPU is not PCIe relaxed ordering compliant" advisory.  The workaround
# it advises (--disable_pcie_relaxed on both ends) is still available.  That is a
# real, if narrow, regression and it is why this file exists rather than a sed
# one-liner in the Dockerfile: the tradeoff should be readable by whoever bumps
# PERFTEST_REF.
#
# ── FAIL LOUD ─────────────────────────────────────────────────────────────────
# Every edit below is asserted.  If a future perftest restructures this code the
# build FAILS rather than silently producing an image that either links libpci
# or has a broken relaxed-ordering check.  Do not "fix" a failure here by
# installing pciutils-dev.

set -eu

cd "${1:?usage: remove-libpci.sh <perftest source dir>}"

test -f configure.ac || { echo "not a perftest source tree: no configure.ac" >&2; exit 1; }
test -f src/perftest_parameters.c || { echo "not a perftest source tree: no src/perftest_parameters.c" >&2; exit 1; }

# The patch must have something to do. If upstream has already dropped the
# dependency, say so loudly instead of pretending to have patched it.
grep -q 'pci/pci.h' configure.ac || { echo "configure.ac no longer requires pciutils — re-check whether this patch is still needed" >&2; exit 1; }

# 1. configure: stop requiring the pciutils headers and library. LIBPCI is left
#    unset, which is exactly the state a FreeBSD build is already in, so the
#    Makefile substitution already handles it.
sed -i '/AC_CHECK_HEADERS(\[pci\/pci\.h\]/d' configure.ac
sed -i '/AC_CHECK_LIB(\[pci\], \[pci_init\]/d' configure.ac

# 2. drop the include, which is guarded by HAVE_RO and not by the header check.
sed -i '/#include <pci\/pci\.h>/d' src/perftest_parameters.c

# 3. answer the compliance question without scanning the bus.
sed -i '/^static bool check_pcie_relaxed_ordering_compliant(void) {$/,/^\treturn cpu_is_RO_compliant;$/c\
static bool check_pcie_relaxed_ordering_compliant(void) {\
\t/* glimmer-burnin: the upstream implementation scans the PCI bus with\
\t * libpci (GPL-2.0-or-later) solely to detect one Intel root-complex\
\t * erratum. This image may not link a GPL library, and on a non-x86 host\
\t * the scan can only ever return true. See remove-libpci.sh. */\
\treturn true;' src/perftest_parameters.c

# ── assertions ────────────────────────────────────────────────────────────────
for f in configure.ac src/perftest_parameters.c; do
	if grep -nE 'pci/pci\.h|pci_init|pci_alloc|pci_scan_bus|pci_cleanup|pci_fill_info' "$f"; then
		echo "remove-libpci.sh: libpci is still referenced in $f — refusing to build" >&2
		exit 1
	fi
done
grep -q 'glimmer-burnin: the upstream implementation scans the PCI bus' src/perftest_parameters.c || {
	echo "remove-libpci.sh: the relaxed-ordering stub was not installed — refusing to build" >&2
	exit 1
}
# HAVE_RO must SURVIVE. If this check ever fails the image would be measuring
# without relaxed ordering and reporting the result as if nothing had changed.
grep -q 'IBV_ACCESS_RELAXED_ORDERING' configure.ac || {
	echo "remove-libpci.sh: HAVE_RO detection is gone from configure.ac — the measurement would change" >&2
	exit 1
}

echo "remove-libpci.sh: perftest patched; libpci (GPL-2.0-or-later) is no longer a dependency"
