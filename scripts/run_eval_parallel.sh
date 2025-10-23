#!/usr/bin/env bash
set -euo pipefail

# Configuration
SAMPLES="${SAMPLES:-samples.json}"
DOC_ROOT="${DOC_ROOT:-../../../data/users/yiming/dtagent/MinerU_25_MMLB}"
MODEL="${MODEL:-gpt-4o}"
CHUNK="${CHUNK:-200}"
PROMPT_FILE="${PROMPT_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_batches}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set."
  exit 1
fi

TOTAL=$(python - <<PY
import json
with open("${SAMPLES}") as f:
    data = json.load(f)
print(len(data))
PY
)

echo "Total samples: ${TOTAL}"

OFFSETS=$(python - <<PY
from math import ceil
chunk = int("${CHUNK}")
total = int("${TOTAL}")
batches = ceil(total / chunk)
for i in range(batches):
    print(i * chunk)
PY
)

for OFFSET in ${OFFSETS}; do
  echo "[INFO] Launching batch offset=${OFFSET}"
  PYTHONPATH=src:. python -m experiments.eval_pipeline \
    --samples "${SAMPLES}" \
    --doc-root "${DOC_ROOT}" \
    --answer-report "${OUTPUT_DIR}/answer_report_${OFFSET}.txt" \
    --dump-json "${OUTPUT_DIR}/details_${OFFSET}.json" \
    --model "${MODEL}" \
    --limit "${CHUNK}" \
    --offset "${OFFSET}" \
    ${PROMPT_FILE:+--answer-prompt "${PROMPT_FILE}"} \
    &
done

wait
echo "[INFO] All batches completed."
