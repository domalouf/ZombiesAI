#!/usr/bin/env bash
# Record a World at War session with this machine's standard settings, so no session gets a flag wrong.
#
#   scripts/record_waw.sh 20 "camping the help room, bad positioning after round 8"
#   scripts/record_waw.sh 2 "smoke test" --no-hud          # anything after the notes goes to record_demo.py
#
# Run it from any workspace, then switch to the game: recording starts after a 3 s countdown once the game
# window is on screen. F8 marks menus, pauses, deaths and loading screens as not playing.
#
# The standing settings live here, overridable per run from the environment:
#   ZOMBIESAI_CPD      counts per degree for sensitivity 5 x m_yaw 0.022 (the recorder warns if the game's
#                      config.cfg disagrees; replace with calibrate_mouse.py's number once measured)
#   ZOMBIESAI_WINDOW   the game window's title
#   ZOMBIESAI_AUDIO    set to 0 to record without audio
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
minutes=$1
notes=$2
shift 2

cd "$(dirname "$0")/.."
args=(
  --source screen
  --window "${ZOMBIESAI_WINDOW:-Call of Duty}"
  --wait
  --counts-per-degree "${ZOMBIESAI_CPD:-9.09}"
  --bindings configs/waw_bindings.json
  --minutes "$minutes"
  --notes "$notes"
)
[[ "${ZOMBIESAI_AUDIO:-1}" != 0 ]] && args+=(--audio)
exec uv run python scripts/record_demo.py "${args[@]}" "$@"
