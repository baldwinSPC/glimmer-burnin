#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright the Glimmer authors.
#
# Refuse to build a perftest that would drag a copyleft library into a published
# runner image.
#
# ── THE FINDING THIS GUARD EXISTS FOR ─────────────────────────────────────────
#
# linux-rdma/perftest itself is dual-licensed GPL-2.0-only OR BSD-2-Clause, and
# this project consumes it under the BSD option — that part is fine and is
# recorded in NOTICE and in the README.
#
# What is NOT fine, and is not obvious from perftest's own licence, is that from
# the v4.5 series onward its configure REQUIRES pciutils on every non-FreeBSD
# platform, with no --without-pciutils to turn it off:
#
#     if [test $IS_FREEBSD = no]; then
#         AC_CHECK_HEADERS([pci/pci.h],,[AC_MSG_ERROR([pciutils header files not found...])])
#         AC_CHECK_LIB([pci], [pci_init], [LIBPCI=-lpci], AC_MSG_ERROR([libpci not found]))
#     fi
#
# pciutils/libpci is GPL-2.0-or-later. Linking it would put a GPL library in an
# image this project publishes, which its licensing rules forbid outright (see
# CLAUDE.md: "a copyleft dependency would make the project unpublishable").
#
# The pinned version, v4.4-0.37, PREDATES that requirement and references libpci
# nowhere, so nothing has to be patched today. This script's job is to make sure
# that stays true: a well-meaning bump of PERFTEST_REF into the v4.5 series would
# otherwise silently produce an unpublishable image, and the symptom would be a
# licence audit months later rather than a red build now.
#
# ── IF THIS SCRIPT FAILS ──────────────────────────────────────────────────────
#
# DO NOT install pciutils-dev to make it pass. The options, in order of
# preference, are:
#
#   1. Ask upstream for a --without-pciutils (the code guarded by it is a single
#      PCI-bus scan for one Intel root-complex erratum — vendor 0x8086, devices
#      0x2f01 and 0x6f01-0x6f0e — which cannot match on any aarch64 node this
#      runner targets, so on those hosts it can only ever return "compliant").
#   2. Carry a patch here that removes the scan while KEEPING HAVE_RO, so
#      IBV_ACCESS_RELAXED_ORDERING stays on the memory region. Dropping HAVE_RO
#      is the easier patch and is the wrong one: it changes the number being
#      measured, and a burn-in runner may not quietly alter its own measurement
#      to simplify its build.
#   3. Leave the pin where it is.
#
# Note separately that the v4.5 series also fails to compile against stock
# rdma-core (50.0 on Ubuntu 24.04, 56.1 on Debian trixie): it needs the AES-XTS
# crypto mkey API from mlx5dv that only MLNX_OFED/DOCA ships. So the pin is held
# in place by two independent constraints, not one.

set -eu

cd "${1:?usage: assert-no-copyleft.sh <perftest source dir>}"

test -f configure.ac || { echo "not a perftest source tree: no configure.ac" >&2; exit 1; }
test -d src || { echo "not a perftest source tree: no src/" >&2; exit 1; }

if grep -rnE 'pci/pci\.h|pci_init|pci_alloc|pci_scan_bus|pciutils|-lpci|LIBPCI' configure.ac Makefile.am src/ 2>/dev/null; then
	cat >&2 <<'EOF'

REFUSING TO BUILD: this perftest wants pciutils/libpci, which is GPL-2.0-or-later.

A GPL library may not ship in a glimmer-burnin runner image. Read the header of
assert-no-copyleft.sh for the three ways out; installing pciutils-dev is not one
of them.
EOF
	exit 1
fi

echo "assert-no-copyleft.sh: no libpci (GPL-2.0-or-later) dependency in this perftest source: OK"
