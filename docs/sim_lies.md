# What NachtSim doesn't model

NachtSim is a strategy and algorithm testbed, not a replica. **Visual sim-to-real
transfer is a non-goal:** no sim-trained weights ever load into the real agent's
encoder. Sim scores are a pass/fail smoke test, never a leaderboard. This file
lists what the sim gets wrong or leaves out, so nobody forgets. Keep it current.

## Ground truth vs guesses

Only `src/zombiesai/sim/mechanics.py` holds numbers read from
`_zombiemode_prototype.gsc`: the zombie health curve, solo zombie counts, spawn
pacing, move-speed formula, prices, and the scoring table. Everything in
`src/zombiesai/sim/params.py` is a guess, and the uncertain values are
domain-randomized per episode, scaled by `hardness`.

## Known lies

**World**
- **Geometry is uncalibrated.** `configs/env/nacht_geometry.yaml` has the right
  topology (start → help door → help room, start → debris → upstairs, help →
  upstairs door), but room sizes, window placement and count, and box and
  wall-weapon spots are schematic.
- **Two floors, one plane.** Upstairs is unfolded beside the ground floor and
  joined only by stair corridors. There's no looking down through the broken
  floor.
- **No props or clutter.** Rooms are empty polygons, so cover and body-blocking
  geometry are missing.

**Zombies**
- **Pathing is too clean.** Zombies follow a door-aware distance field over a
  0.5 m grid. There's no pathing jank, no getting stuck on props, and no
  crawlers (in WaW, explosives gib legs).
- **Speed tiers are borrowed.** Walk/run/sprint thresholds come from the later
  `_zombiemode_spawner` script, and the m/s speeds are invented.
- **Windows are simplified.** Six planks per window, torn one at a time per
  zombie, then a fixed-length climb. Zombies tearing at a window can hit a
  player standing right against it.
- **Attacks are a timer.** A fixed-range swing with a windup: stepping out of
  range before it lands dodges it. There are no animation lockouts.
- **No hellhounds, power-ups, or carpenter.** Max Ammo, Insta-Kill, Double
  Points and Nuke are all absent.

**Player and weapons**
- **Ballistics are an aim cone.** Gaussian angular error, a head disc, and a
  torso box. There's no bullet penetration, no real hitbox geometry, and damage
  falloff is a single step.
- **Weapon stats are approximate.** They weren't read from the WaW weapon
  files. Headshot multipliers, fire rates, reload times, and spreads are all
  guesses.
- **ADS is instant.** No aim-in time, and there's no reload canceling.
- **The box is idealized.** It grants instantly, from an invented table and
  weights, with no cycling animation, teddy bear, or pickup timer.
- **Grenades are crude.** They land at a fixed throw distance (shortened by
  walls) with linear damage falloff, no cooking, and no bounces.

**Perception and scoring**
- **The damage overlay is a model.** The red vignette is modelled as tracking
  missing health with exponential decay, plus detector noise and false
  positives. The real curve is unmeasured. The agent never sees HP directly,
  matching the real game.
- **HUD noise is synthetic.** It's digit drop/duplicate/substitution at a fixed
  rate, not the real parser's failure distribution. Reward still uses true
  Δpoints until `HudTracker` (M4) exists to filter misreads the same way in both
  backends.
- **Timing randomization is approximate.** Latency (0–3 decisions), action
  repeat (3–5 frames), and 5% action dropout are sampled per episode, not
  measured from S3.
- **No audio.**
- **The rendered view is crude.** `render()` is a flat-shaded raycast: untextured
  walls of one height (3 m), billboard zombies that always face the camera and
  never animate, a stand-in gun, and no lighting beyond distance fog. The HUD
  uses a made-up 3×5 pixel font, not the real glyph atlas. The 80° field of view
  assumes WaW widens `cg_fov 65` to Hor+ at 16:9; it hasn't been measured.
- **Pixels are an observation now, and they are still a stand-in.**
  `SimConfig(obs_profile="render")` feeds the raycast view to the policy, which
  is what lets a pixel network (behavioural cloning, M2's CNN check) be
  evaluated at all. It is a smoke test, not a transfer path: an inverse dynamics
  model or an encoder fit to these frames will not read real World at War
  footage, and no sim-trained encoder weights ever load into the real agent.
