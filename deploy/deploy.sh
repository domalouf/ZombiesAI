#!/usr/bin/env bash
#
# Build the training dashboard and the stream's page, and publish them to https://domalouf.com/zombies/training/
# (how it is learning) and /zombies/live/ (the Twitch stream, with the overlay OBS draws on it at
# /zombies/live/overlay/). The bare /zombies/ sends visitors on to the Training Room.
#
# Same pattern as the site's other project pages: rsync a static directory into a
# sub-dir of the site's web root on the server (~/site/www/, which the MyWebsite repo's nginx serves).
# The page makes no external requests (fonts ship alongside it), so it holds up
# under the site's default-src 'self' CSP.
#
# Config via environment (optional):
#   DEPLOY_DEST     rsync destination  (default: lts:site/www/zombies/)
#   TWITCH_CHANNEL  the channel /zombies/live/ embeds (default: none, a placeholder until there is a stream).
#                   The site's CSP must let the player in: frame-src https://player.twitch.tv.
#   LIVE_MACHINES   the other gaming PCs' ids, space-separated (default: none): the Training Room shows each one's
#                   live/machine-<id>.json beside the training PC (publish_live.py --worker <id> on that PC).
#
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${DEPLOY_DEST:-lts:site/www/zombies/}"

log() { printf '==> %s\n' "$*"; }

cd "$repo"

rm -rf site/zombies
mkdir -p site/zombies
# /zombies/ has no page of its own; links to it land on the Training Room. A meta refresh, not a script: the
# site's CSP allows no inline script.
cat > site/zombies/index.html <<'HTML'
<!doctype html>
<meta charset="utf-8">
<title>ZombiesAI</title>
<meta http-equiv="refresh" content="0; url=training/">
<a href="training/">The Training Room</a>
HTML

# The dashboard reads only the JSON each trainer wrote, so this step needs no checkpoint and no GPU.
# --site strips the local paths a config carries (clip filenames and the like) and ships the fonts as
# files rather than data: URIs, which the site's CSP refuses. --live makes it the live page: every checkout's
# runs baked in, then the machine and the runs kept current from training/live/*.json, which
# scripts/publish_live.py (deploy/zombiesai-live.service) pushes from the training PC every few seconds.
log "building the live training dashboard"
uv run python scripts/dashboard.py --site site/zombies/training --live

# The stream's page and its overlay poll training/live/stream.json, which publish_live.py pushes beside the rest.
log "building the stream page (channel=${TWITCH_CHANNEL:-none yet})"
uv run python scripts/build_stream.py --out site/zombies/live

log "publishing site/zombies/ -> $dest"
# --delete so a rebuilt page doesn't leave stale files behind; trailing slashes matter. The filters protect two
# directories from it: training/live/ is the training PC's, pushed there and not built here; training/saved/ holds
# the saved games, which the site keeps once they are up, even when deployed from a checkout without their films
# (runs/saved/ is only on the training PC). Taking one down is done on the server.
rsync -av --delete --filter='P training/live/' --filter='P training/saved/' site/zombies/ "$dest"

log "done — https://domalouf.com/zombies/training/ and /zombies/live/"
