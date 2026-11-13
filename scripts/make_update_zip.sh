#!/bin/bash
# Run on your MAC (in the repo, after `git pull`): bundle everything changed since the commit the cluster is on.
#   bash scripts/make_update_zip.sh <cluster_commit> [output.zip]
# <cluster_commit> = the hash shown by `git log --oneline | head -1` on the cluster.
# Then upload the zip to the project folder on the cluster and run:  unzip -o update.zip && rm update.zip
set -e
FROM="${1:?give the commit hash that the cluster is on}"
OUT="${2:-$HOME/Desktop/update.zip}"
rm -f "$OUT"
git diff --name-only --diff-filter=d "$FROM" HEAD | xargs zip -q "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)), $(git diff --name-only --diff-filter=d "$FROM" HEAD | wc -l | tr -d ' ') files"
echo "upload it to the project folder on the cluster, then:  unzip -o update.zip && rm update.zip"
