#!/usr/bin/env bash
# Fetch the data PM-SQL cannot redistribute, into this directory. Idempotent.
#   dataset.tsv        benchmark questions + gold SQL
#   event_log.sqlite   BPIC 2020 International Declarations, SQLite form
#   --with-xes         also fetch the raw XES log, needed only to regenerate the
#                      process-mining artifacts for the Model-Full configuration.
set -euo pipefail
cd "$(dirname "$0")"
WITH_XES=0; [ "${1:-}" = "--with-xes" ] && WITH_XES=1

SRC="${BENCH_DIR:-.text2sql4pm-src}"
if [ ! -d "$SRC" ]; then
  echo "cloning Text-2-SQL-4-PM benchmark..."
  git clone --depth 1 --quiet https://github.com/pm-usp/text-2-sql.git "$SRC"
fi
cp "$SRC/data/dataset/text2sql4pm.tsv"          dataset.tsv
cp "$SRC/data/dataset/english/event_log.sqlite" event_log.sqlite
echo "placed dataset.tsv and event_log.sqlite"

if [ "$WITH_XES" = 1 ]; then
  XES=BPIC2020_InternationalDeclarations.xes
  if [ ! -s "$XES" ]; then
    echo "fetching the BPIC 2020 International Declarations XES log (~15 MB)..."
    curl -sSL --max-time 600 -o "$XES.gz" \
      "https://data.4tu.nl/file/91fd1fa8-4df4-4b1a-9a3f-0116c412378f/d45ee7dc-952c-4885-b950-4579a91ef426"
    gunzip -f "$XES.gz"
  fi
  echo "placed $XES  (now run: python3 architecture/process_mining/precompute_pm_artifacts.py)"
fi
