#!/usr/bin/env bash
# Run the PM-SQL ladder on Bedrock, score it, and print the accuracy table.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 run.py "$@"
