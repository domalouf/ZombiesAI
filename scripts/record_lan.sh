#!/usr/bin/env bash
# Record a demo on the fleet's own game: Plutonium T4 (LAN, offline) on your desktop.
#
#   scripts/record_lan.sh 20 "camping the help room, bad positioning after round 8"
#   scripts/record_lan.sh 2 "smoke test" --no-scene      # anything after the notes goes to record_demo.py
#
# Starts the game if it is not running (scripts/desktop_game.py up), then records it as record_waw.sh does.
# At the main menu press F9 to start Nacht, and again after each game over. F8 marks menus, pauses, deaths
# and loading screens as not playing. The game stays open when the recording ends: record again, or quit it
# with `uv run python scripts/desktop_game.py down`.
#
# The audio is the game's own sink only, not the whole desktop. The recording is stamped with the config.cfg
# the game actually played with (the fleet's, copied into runs/desktop/ at each launch).
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
minutes=$1
notes=$2
shift 2

cd "$(dirname "$0")/.."
uv run python scripts/desktop_game.py up
config=$(uv run python -c 'from zombiesai.realgame.desktop import desktop_config, profile_config; print(profile_config(desktop_config()))')
args=(--game-config "$config")
[[ "${ZOMBIESAI_AUDIO:-1}" != 0 ]] && args+=(--audio-device zai_desktop.monitor)
ZOMBIESAI_WINDOW="${ZOMBIESAI_WINDOW:-Plutonium T4}" exec scripts/record_waw.sh "$minutes" "$notes [plutonium lan]" "${args[@]}" "$@"
