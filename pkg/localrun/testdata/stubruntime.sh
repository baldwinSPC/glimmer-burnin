#!/bin/sh
# A stand-in for docker, podman or nerdctl whose containers belong to a daemon:
# a container outlives the client that started it, as theirs do.
#
#   STUB_STATE        directory holding argv (one line per call) and one
#                     running.<name> file per running container
#   STUB_TICKS        50 ms ticks a container runs before exiting 0 by itself;
#                     unset or 0 runs it until it is killed
#   STUB_DAEMON_DOWN  when 1, kill and rm fail as they do with no daemon
state=${STUB_STATE:?}
printf '%s\n' "$*" >>"$state/argv"

case "$1" in
run)
	name=
	prev=
	for a in "$@"; do
		if [ "$prev" = --name ]; then name=$a; fi
		prev=$a
	done
	: >"$state/running.$name"
	# The client relays SIGTERM to the container's PID 1, which has no handler
	# and drops it, so the client stays attached and the container runs on.
	trap '' TERM INT
	n=0
	while [ -f "$state/running.$name" ]; do
		if [ "${STUB_TICKS:-0}" -gt 0 ] && [ "$n" -ge "$STUB_TICKS" ]; then
			rm -f "$state/running.$name"
			echo tflops=101
			exit 0
		fi
		n=$((n + 1))
		sleep 0.05 >/dev/null 2>&1
	done
	exit 137
	;;
kill | rm)
	if [ "${STUB_DAEMON_DOWN:-}" = 1 ]; then
		echo "Cannot connect to the container daemon. Is it running?" >&2
		exit 1
	fi
	eval "name=\${$#}"
	if [ -f "$state/running.$name" ]; then
		rm -f "$state/running.$name"
		echo "$name"
		exit 0
	fi
	echo "Error response from daemon: No such container: $name" >&2
	# docker's rm --force exits 0 for a container that is already gone.
	if [ "$1" = rm ]; then exit 0; fi
	exit 1
	;;
esac
echo "stub runtime: unexpected call: $*" >&2
exit 2
