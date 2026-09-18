import json
import re

from zombiesai.agents.scripted import ScriptedAgent
from zombiesai.sim.nacht_sim import NachtSim, SimConfig
from zombiesai.viz.replay import record_replay, write_replay_html, write_site_page


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


def test_site_page_carries_its_nav_and_makes_no_external_request(tmp_path):
    env = NachtSim(SimConfig(max_steps=120))
    replay = record_replay(env, ScriptedAgent(), seed=0, agent_name="scripted")
    index = write_site_page(
        replay,
        tmp_path / "zombies",
        intro="One simulated game.",
        links=[("← domalouf.com", "/"), ("How it learned", "/zombies/training/")],
        description="Watch it play.",
    )
    html = index.read_text()
    # The nav is how a reader gets from the agent playing to how it learned, and back to the site.
    assert '<a href="/">← domalouf.com</a>' in html
    assert '<a href="/zombies/training/">How it learned</a>' in html
    assert "One simulated game." in html
    assert '<meta name="description" content="Watch it play.">' in html
    # Fonts ship alongside, so the page holds up under the site's default-src 'self' CSP.
    assert (tmp_path / "zombies" / "fonts" / "red-hat-mono.woff2").exists()
    assert "fonts.googleapis.com" not in html
    assert not re.search(r'(src|href)="(?!#)(https?:)?//', html)
