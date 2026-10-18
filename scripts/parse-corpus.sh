#!/bin/bash
# parse-check every PTX file under the given directory (default corpora/) with the PTX grammar
D=${PTX_DEF:-build/ptx-kompiled}; fail=0
for f in $(find "${1:-corpora}" -name '*.ptx'); do
  if K_OPTS=-Xss512m kast -d "$D" "$f" -o none >/dev/null 2>&1; then echo "OK   $f"; else echo "FAIL $f"; fail=1; fi
done
exit $fail
