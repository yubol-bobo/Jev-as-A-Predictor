#!/bin/bash
# Publish this repository's committed state to github.com/yubol-bobo/Jev-as-A-Predictor (PUBLIC), at the root of
# its 032p clone, then push. The user decided on 2026-10-19 that the project now is this clinical study and that its
# progress is pushed there often. The earlier sequential-predictor study is kept at the tag
# archive/seqpred-2026-10-19. The full history stays in this local repository. The public repository gets one
# snapshot commit per sync, which lists the local commits it covers and ends in "Synced-From: <sha>".
# Only git-tracked files are copied (git archive HEAD), so data, logs and job directories, which are never tracked,
# cannot leave. Excluded on top: third_party/clinenv (no upstream license; see its PROVENANCE.md).
#   bash scripts/sync_to_jev.sh
set -euo pipefail
SRC=/ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents
JEV=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
DEST=$JEV
umask 022
cd "$JEV"
git pull -q --rebase origin main
head=$(git -C "$SRC" rev-parse HEAD)
last=$(git log --format=%B | sed -n 's/^Synced-From: //p' | head -1)
[ "$last" = "$head" ] && { echo "already synced at ${head:0:7}"; exit 0; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git -C "$SRC" archive HEAD | tar -x -C "$TMP"
rm -rf "$TMP/third_party/clinenv"
mkdir -p "$TMP/third_party/clinenv"
cat > "$TMP/third_party/clinenv/NOT_INCLUDED.md" <<'EOF'
# ClinEnv is not included here

The experiments use ClinEnv (https://github.com/ylin766/ClinEnv @ c033d74), vendored with local patches in the
private working repository. Upstream has no license file, so the code is not redistributed in this public
repository. Every local patch is described in `docs/clinenv_pipeline.md` ("Patches to ClinEnv").
EOF
rsync -a --delete --exclude=/.git "$TMP/" "$DEST/"
git add -A
if git diff --cached --quiet; then
  echo "no changes"; exit 0
fi
range=${last:+$last..}$head
subjects=$(git -C "$SRC" log --format='- %h %s' "$range" 2>/dev/null | head -40 || true)
git commit -q -m "Sync to ${head:0:7} ($(git -C "$SRC" log -1 --format=%s "$head" | cut -c1-80))

Local commits covered:
$subjects

Synced-From: $head
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q origin main
echo "$(date +%T) synced ${head:0:7} and pushed"
