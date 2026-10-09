#!/bin/bash
# Run a list of prefill benchmarks serially and append machine-readable, VALIDATED results.
#
# Usage: run_suite.sh <manifest> <out.jsonl>
# Manifest lines: label|tokens-file|chunk|ENV1=VAL ENV2=VAL|extra engine args
# Blank lines and # comments are skipped.
#
# W0 rules enforced here:
#   * a run with nonzero exit, a missing final timing field, or a token-count mismatch is a FAILED record;
#   * failed records carry valid=false and invalid_reason and never enter an aggregate;
#   * every run writes its own uniquely named log and raw artifacts are kept even for failures;
#   * the JSONL is never overwritten silently (append to a fresh file, keep the old one as .prev).
set -u
cd /home/dwmcqueen/Strata
MAN="${1:?manifest}"; OUT="${2:?out.jsonl}"
RESULTS=bench/sycl-prefill/results
mkdir -p "$RESULTS"
# do not silently overwrite: move an existing output aside once
if [ -s "$OUT" ]; then mv "$OUT" "$OUT.prev.$(date +%s)"; fi
: > "$OUT.tmp"
STAMP="$(date +%Y%m%dT%H%M%S)"
SEQ=0
while IFS='|' read -r label tok chunk envs extra; do
    case "$label" in ''|'#'*) continue;; esac
    label="$(echo "$label" | xargs)"; tok="$(echo "$tok" | xargs)"; chunk="$(echo "$chunk" | xargs)"
    SEQ=$((SEQ+1))
    # unique log name: never overwrite a prior run
    log="$RESULTS/${label}.${STAMP}.$$.$SEQ.log"
    echo "=== $(date -Is) $label (chunk=$chunk) -> $log ===" >&2
    env $envs bench/sycl-prefill/run_prefill_bench.sh "$tok" "$chunk" $extra > "$log" 2>&1
    rc=$?
    echo "  exit=$rc" >&2
    # expected token count from the prompt ids (comma separated on one line)
    want=$(tr -cd ',' < "$tok" | wc -c)
    want=$((want + 1))
    python3 bench/sycl-prefill/parse_run.py "$log" 2>/dev/null \
        | python3 -c "
import sys, json
d = json.load(sys.stdin)
d['label'] = '$label'
d['chunk_arg'] = '$chunk'
d['exit'] = $rc
d['expected_tokens'] = $want
d['log'] = '$log'
bad = None
if $rc != 0: bad = 'nonzero exit'
elif 'final_prefill_ms' not in d and 'inner_prefill_ms' not in d: bad = 'no prefill timing field'
elif 'final_tokens' in d and d['final_tokens'] != $want: bad = 'token count %d != %d' % (d['final_tokens'], $want)
d['valid'] = bad is None
if bad: d['invalid_reason'] = bad
print(json.dumps(d))
" >> "$OUT.tmp"
done < "$MAN"
mv "$OUT.tmp" "$OUT"
python3 bench/sycl-prefill/summarize.py "$OUT" > "$OUT.summary.txt" 2>&1 || true
cat "$OUT.summary.txt" >&2
echo "=== suite done $(date -Is) ===" >&2
