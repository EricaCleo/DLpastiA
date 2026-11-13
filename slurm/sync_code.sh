#!/bin/bash
# Bring this checkout up to date with the branch on GitHub, trying several routes. Run on the LOGIN node:
#   bash slurm/sync_code.sh            (from the project folder)
# Route 1: shallow `git fetch` (small, retried).  Route 2: the branch as a zip from GitHub's download server.
# If both fail: on your Mac run `bash scripts/make_update_zip.sh <cluster_commit>` and upload the zip.
BRANCH="${BRANCH:-claude/upbeat-rubin-7at9xc}"
REPO="${REPO:-EricaCleo/DLpastiA}"
cd "$(dirname "$0")/.." || exit 1
echo "now at: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"

for i in 1 2 3 4 5; do
  echo "[git] attempt $i of 5"
  if timeout 240 git -c http.version=HTTP/1.1 fetch --depth 1 origin "$BRANCH"; then
    git reset --hard FETCH_HEAD
    echo "SYNCED via git: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
    exit 0
  fi
  sleep 20
done

echo "[zip] git failed; trying the branch zip"
T="$(mktemp -d)"
if curl -L --retry 3 --max-time 180 -o "$T/b.zip" "https://github.com/$REPO/archive/refs/heads/$BRANCH.zip" && unzip -q "$T/b.zip" -d "$T"; then
  D="$(ls -d "$T"/*/ | head -1)"
  cp -r "$D"/scripts "$D"/slurm "$D"/src "$D"/configs "$D"/notebooks "$D"/README.md "$D"/requirements.txt .
  echo "SYNCED via zip (the files are current, but git still shows the old commit)"
  exit 0
fi
echo "FAILED: could not reach GitHub. On your Mac: bash scripts/make_update_zip.sh $(git rev-parse --short HEAD) , upload the zip, unzip -o it here."
exit 1
