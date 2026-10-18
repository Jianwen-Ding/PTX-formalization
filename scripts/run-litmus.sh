#!/usr/bin/env bash
# Litmus tests for the relaxed memory mode (`.memmodel relaxed;`).  For every test, list the exit codes
# reachable over all schedules (0 = the weak outcome encoded by the .expect lines was observed) and compare
# with the outcome PTX allows.
set -u
cd "$(dirname "$0")/.."
DEF=${PTX_DEF:-build/ptx-kompiled}
export K_OPTS="${K_OPTS:--Xss512m}"
outcomes() {
  krun -d $DEF "$1" --search --pattern '<exit-code> X:Int </exit-code>' 2>/dev/null \
    | grep -A2 'X:Int' | grep -E '^\s+[0-9]+$' | sort -u | tr -d ' ' | paste -sd, -
}
fail=0
check() {   # file, expected set of exit codes
  got=$(outcomes "examples/$1.ptx")
  if [ "$got" = "$2" ]; then echo "PASS  $1  reachable exit codes: {$got}"; else echo "FAIL  $1  reachable {$got}, expected {$2}"; fail=1; fi
}
check litmus-sb        0,1     # weak outcome allowed
check litmus-sb-fence  1       # forbidden by fence.sc
check litmus-sb-sc     1       # default mode is sequentially consistent
check litmus-mp        0,1     # stores may be reordered
check litmus-mp-relacq 1       # forbidden by release/acquire
check litmus-lb        0,1     # load buffering: the load is performed after the later store
check litmus-lb-relacq 1       # forbidden by acquire/release
exit $fail
