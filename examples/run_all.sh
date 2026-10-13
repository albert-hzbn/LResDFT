#!/bin/sh
# Run every example with both solvers; summary table at the end. Usage: examples/run_all.sh [extra lrft options]
cd "$(dirname "$0")/.."
mkdir -p examples/out
printf "%-12s %14s %10s %s\n" molecule "E_KS (Ha)" "dE/e (Ha)" status
for f in examples/*.xyz; do
  n=$(basename "$f" .xyz)
  python3 -m lrft run "$f" --solver both --json examples/out/$n.json "$@" > examples/out/$n.log 2>&1
  rc=$?
  python3 - "$n" "$rc" <<'P'
import json, sys
n, rc = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(f"examples/out/{n}.json"))["results"]
    print(f"{n:<12s} {r['ks']['energy_Ha']:14.6f} {r['comparison']['dE_per_electron_Ha']:10.1e} {'ok' if rc == '0' else 'NOT CONVERGED'}")
except Exception:
    print(f"{n:<12s} {'-':>14s} {'-':>10s} failed (see examples/out/{n}.log)")
P
done
