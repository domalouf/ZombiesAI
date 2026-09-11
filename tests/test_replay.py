import json
import re

from zombiesai.agents.scripted import ScriptedAgent
from zombiesai.sim.nacht_sim import NachtSim, SimConfig
from zombiesai.viz.replay import record_replay, write_replay_html


def test_replay_records_every_step_and_embeds_it(tmp_path):
    env = NachtSim(SimConfig(max_steps=600))
    replay = record_replay(env, ScriptedAgent(), seed=0, agent_name="scripted")
    frames = replay["frames"]
    assert len(frames["t"]) == replay["summary"]["steps"] + 1
    assert all(len(col) == len(frames["t"]) for col in frames.values())
    assert replay["events"][0]["kind"] == "round_start"

    path = write_replay_html(replay, tmp_path / "replay.html")
    html = path.read_text()
    assert html.startswith("<!doctype html>") and "<title>NachtSim Film Room</title>" in html
    embedded = re.search(r"const REPLAY = (\{.*?\});</script>", html, re.S).group(1)
    assert json.loads(embedded)["summary"] == replay["summary"]
    fragment = write_replay_html(replay, tmp_path / "fragment.html", standalone=False).read_text()
    assert fragment.startswith("<title>")
