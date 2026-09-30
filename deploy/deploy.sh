#!/usr/bin/env bash
#
# Build the NachtSim replay page and the training dashboard, and publish both to
# https://domalouf.com/zombies/ (the agent playing) and /zombies/training/ (how it learned).
#
# Same pattern as the site's other project pages: rsync a static directory into a
# sub-dir of the site's web root on the server (~/site/www/, which the MyWebsite repo's nginx serves).
# The page makes no external requests (fonts ship alongside it), so it holds up
# under the site's default-src 'self' CSP.
#
# Config via environment (optional):
#   DEPLOY_DEST     rsync destination  (default: lts:site/www/zombies/)
#   CHECKPOINT  trained policy     (default: runs/ppo-nacht-state-s1/checkpoint.pt)
#   SEED        which game to show (default: 10030, one of its round-4 games)
#
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${DEPLOY_DEST:-lts:site/www/zombies/}"
checkpoint="${CHECKPOINT:-runs/ppo-nacht-state-s1/checkpoint.pt}"
seed="${SEED:-10030}"

log() { printf '==> %s\n' "$*"; }

cd "$repo"

log "building the replay page (checkpoint=$checkpoint, seed=$seed)"
rm -rf site/zombies
uv run python scripts/build_site.py --checkpoint "$checkpoint" --seed "$seed" --out site/zombies

# The dashboard reads only the JSON each trainer wrote, so this step needs no checkpoint and no GPU.
# --site strips the local paths a config carries (clip filenames and the like) and ships the fonts as
# files rather than data: URIs, which the site's CSP refuses. --live makes it the live page: every checkout's
# runs baked in, then the machine and the runs kept current from training/live/*.json, which
# scripts/publish_live.py (deploy/zombiesai-live.service) pushes from the training PC every few seconds.
log "building the live training dashboard"
uv run python scripts/dashboard.py --site site/zombies/training --live

log "publishing site/zombies/ -> $dest"
# --delete so a rebuilt page doesn't leave stale files behind; trailing slashes matter. The filter protects
# training/live/ from it: those files are the training PC's, pushed there and not built here.
rsync -av --delete --filter='P training/live/' site/zombies/ "$dest"

log "done — https://domalouf.com/zombies/ and https://domalouf.com/zombies/training/"
