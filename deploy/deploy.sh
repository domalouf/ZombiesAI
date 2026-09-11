#!/usr/bin/env bash
#
# Build the NachtSim replay page and publish it to https://domalouf.com/zombies/.
#
# Same pattern as the site's other project pages: rsync a static directory into a
# sub-dir of the web root nginx serves on the Pi (~/HealthBoard/piStuff/website/).
# The page makes no external requests (fonts ship alongside it), so it holds up
# under the site's default-src 'self' CSP.
#
# Config via environment (optional):
#   PI_DEST     rsync destination  (default: pi:HealthBoard/piStuff/website/zombies/)
#   CHECKPOINT  trained policy     (default: runs/ppo-nacht-state-s1/checkpoint.pt)
#   SEED        which game to show (default: 10030, one of its round-4 games)
#
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${PI_DEST:-pi:HealthBoard/piStuff/website/zombies/}"
checkpoint="${CHECKPOINT:-runs/ppo-nacht-state-s1/checkpoint.pt}"
seed="${SEED:-10030}"

log() { printf '==> %s\n' "$*"; }

cd "$repo"

log "building the replay page (checkpoint=$checkpoint, seed=$seed)"
rm -rf site/zombies
uv run python scripts/build_site.py --checkpoint "$checkpoint" --seed "$seed" --out site/zombies

log "publishing site/zombies/ -> $dest"
# --delete so a rebuilt page doesn't leave stale files behind; trailing slashes matter.
rsync -av --delete site/zombies/ "$dest"

log "done — https://domalouf.com/zombies/"
