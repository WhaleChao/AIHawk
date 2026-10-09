#!/usr/bin/env bash
# Runs a benchmark on real Dots (tests/bench/README.md), inside the bench container, from /work/dots:
#
#   bash tests/bench/run.sh <agent> <harbor run arguments...>
#   bash tests/bench/run.sh longmemeval run <longmemeval.py run arguments...>
#
# <agent> is "dot" (the Dot does each task), "oracle" (each task's reference solution: proves the task, its
# grader and its replay in a Dot) or "nop" (does nothing: the grader must give 0). It starts the server,
# stores the key from E2E_OPENROUTER_KEY_FILE, runs `harbor run` with the Dot environment (or
# longmemeval.py) and stops the server. TEST HARNESS ONLY.
set -euo pipefail

agent=${1:?usage: run.sh dot|oracle|nop|longmemeval <arguments...>}
shift
case "$agent" in
  dot) agent=dots_harbor.agent:DotAgent ;;
  oracle | nop | longmemeval) ;;
  *) echo "run.sh: unknown agent $agent (dot, oracle, nop, longmemeval)" >&2; exit 2 ;;
esac

repo=$(cd "$(dirname "$0")/../.." && pwd)
logs=${BENCH_LOG_DIR:-/work/bench-logs}
mkdir -p "$logs"
cd "$repo"

node apps/cli/dist/invisible-dots.mjs server --no-web >> "$logs/server.log" 2>&1 &
server=$!
stop_server() {
  kill -TERM "$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
}
trap stop_server EXIT

node tests/bench/bridge.ts ready "${E2E_OPENROUTER_KEY_FILE:?set E2E_OPENROUTER_KEY_FILE}"
if [ "$agent" = longmemeval ]; then
  python3 tests/bench/longmemeval.py "$@"
  exit
fi
PYTHONPATH="$repo/tests/bench${PYTHONPATH:+:$PYTHONPATH}" \
  harbor run -e dots_harbor.environment:DotEnvironment -a "$agent" -o "${BENCH_JOBS_DIR:-/work/bench-jobs}" "$@"
