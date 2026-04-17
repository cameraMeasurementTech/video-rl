#!/usr/bin/env bash
# Example SWE_REWARD_SCRIPT: prints a fixed score for wiring tests only.
# Replace with a script that runs Docker / SWE-Bench for your instance_id.
set -euo pipefail
INSTANCE_ID="${1:?instance_id}"
PRED="${2:?prediction file}"
# shellcheck disable=SC2181
if [[ ! -f "$PRED" ]]; then
  echo "0" >&2
  exit 1
fi
# Placeholder: non-empty response gets partial credit
if [[ -s "$PRED" ]]; then
  echo "0.1"
else
  echo "0"
fi
