# beyond-all-reason-reinforcement-learning — Deep reinforcement learning agent for Beyond All Reason

## Goal
Build a test harness + Gymnasium/PettingZoo wrapper around Beyond All Reason
(Recoil engine, fork of Spring) so a deep reinforcement learning agent can play the game,
then train on the local GPU. The user already has their own reinforcement learning library,
so the environment must match its interface (to confirm).

## Conventions
- Write names in full, no acronyms in project/file/folder names (the user asked for this).
- Discuss with the user in French.

## Machine
- NixOS 25.11, 20 CPU cores, 31 GiB RAM, NVIDIA RTX 5070 Laptop (8 GiB VRAM), Python 3.13.
- No docker, gcc or cmake installed globally. The user works in conda environments
  (`/media/SSD_1To/conda/envs/`); this project's is `Beyond_All_Reason_Reinforcement_Learning`
  (Python 3.13, set as VSCodium default in `.vscode/settings.json`); the package is `pip install -e .` there, no `nix develop` needed:
  `match.py` builds `flake.nix#graphics-libraries` on demand for graphical matches.
- nix-ld is enabled system-wide, so the prebuilt engine binaries run as is.

## Key technical choices (agreed so far)
- Run the game without rendering via `spring-headless` (Recoil engine), launched from
  Python with a generated start script (map, teams, AIs, seed, MinSpeed/MaxSpeed).
- Bridge: a thin native Skirmish AI ("BridgeAI", C++) that serialises observations
  and blocks in `update(frame)` until Python sends actions → synchronous `step()`.
  Simulation is deterministic (lockstep), so seeds give reproducible episodes.
- A Lua gadget for scenarios (spawn units, reset, end-of-episode, reward signals).
- Opponents/baselines: BARb (C++, CircuitAI fork, JSON config), STAI (Lua).
- Simulation is single-threaded → parallelise with one engine process per environment.
- Later: imitation pre-training on Beyond All Reason's public replays (.sdfz), AlphaStar-style.

## Roadmap (start small)
- [x] Step 0: `scripts/download_game_data.py`, `scripts/run.py configurations/runs/<file>.py` (Python file defining
      `configuration = RunConfiguration(...)`; one game or a parallel series, statistics and charts into
      `runs/<name>/`; `run.py`, `plots.py`; run_match.py and run_matches.py were merged into it on 2026-10-18).
      Default file barb_versus_barb.py; `--<field> <value>` replaces the simple fields of the run, scenario, engine
      and window (generated from the dataclass fields in `run.add_configuration_arguments`), `--graphical`,
      `--match_index`; the values used are saved in configuration.json.
- [ ] Micro environment (5 v 5), agreed design below. Phase 0: game extension + Lua <-> Python socket +
      synchronous step(), measure speed [done]; phase 1: scenario gadget [done]; phase 2: Python environment
      (no RLlib) [done: MicroEnvironment];
      phase 3: RLlib adapter in Maxalaar_Reinforcement_Learning; phase 4: several arenas per engine process.
- [ ] Later: 2D viewer, evaluation league vs BARb/STAI, hierarchical agents (group "captain" agents).

## Micro environment design (agreed with the user, 2026-10-15)
- One agent per unit, shared policy (like the user's Cubeball setup in RLlib). Not a centralized model.
- Observation per unit: tokens of the N nearest entities (N = `maximum_entities_seen`, configurable) + the
  unit's own token. Network (in the user's RLlib layer): M learned query vectors per unit type, attention over
  the N entity tokens (linear cost), concatenation, MLP.
- Positions are fractions of the map in [0, 1] (absolute, plus relative to the unit). The unit's own token
  gives the scale, unbounded: map size, map aspect ratio, distance covered per decision, weapon and sight
  ranges, all as fractions of the map.
- Action = (verb, x, y): verb among the commands allowed for the unit type (set in the environment
  configuration, masked): continue, move, fight (attack-move), attack, stop to start; (x, y) in [0, 1]^2 is
  a point of the map, (0, 0) one corner, (1, 1) the opposite one (minimap-like click). `attack` targets the
  enemy nearest to the click within a snap radius, otherwise it is ignored. The user rejected per-enemy
  actions and softmax pointers over entities (do not reintroduce them).
- Scenario: 5 v 5 Pawns (armpw) mirror, opponent does not move (first passive, then hold position and fire),
  episode ends when a side is eliminated or after a time limit; reset inside the match (no engine reload).
- Architecture: LuaSocket only exists in LuaUI and BAR removes it from user widgets, so our Lua is shipped as
  a small game extension (archive depending on the pinned BAR). A widget talks to Python (TCP, blocking in
  GameFrame = synchronous step) and reads unit states; a synced gadget spawns, resets and gives orders,
  receiving actions from the widget through Spring.SendLuaRulesMsg.

## Micro environment, how it works (learned while building it, 2026-10-16)
- Code (now ReinforcementLearningEnvironment, driven by the native BridgeAI; named CombatEnvironment until 2026-10-18): reinforcement_learning_environment_configuration.py,
  reinforcement_learning_environment.py, game.py, scenario.py, controllers.py,
  ray_multi_agent_environment.py (RLlib wrapper, extra "rllib"), engine_configuration.py, scenarios/, flat_map.py,
  game_extension.py, bridge_ai/, lua/game_extension/; scripts/try_reinforcement_learning_environment.py,
  tests/test_reinforcement_learning_environment.py (`pytest -m engine` for the end-to-end test).
- Configuration layout (agreed 2026-10-18, names chosen by the user): ScenarioConfiguration (scenario.py: the
  situation; map, time limit, teams = TeamConfiguration(side, units, allyteam), no controllers) -> GameConfiguration
  (game.py: scenario + controllers, one per team) -> RunConfiguration (run.py: game + matches, parallel, seeds,
  replays, engine; games without Python) or ReinforcementLearningEnvironmentConfiguration
  (reinforcement_learning_environment_configuration.py: game + observation + action + rewards + engine; for
  ReinforcementLearningEnvironment). Scenarios (scenarios/<name>.py, variable named like the file) define only the situation; the
  experiment (training configuration in Maxalaar_Reinforcement_Learning, configurations/beyond_all_reason/)
  chooses the controllers, observation, action and rewards. EngineConfiguration(speed None = 1000 headless, 1 with
  a window; graphics: GraphicsConfiguration) is shared; GraphicsConfiguration (show off, sound off, borderless
  fullscreen, camera_keys "zqsd", close_delay_seconds 5) goes to springsettings.cfg (Sound, WindowBorderless,
  resolution 0 = desktop) and to lua/graphics_widgets/graphics_settings.lua (copied only with a window: "bind
  <key> moveforward" and "bind Shift+<key> ..."; not Any+: with Alt the engine's moverotate turned the keys into a
  rotation that did not feel like the game's, which rotates with Alt + middle mouse button; quitforce after
  GameOver). GraphicsConfiguration.right_button_rotation (default True, user's choice: the 3D-editor convention):
  the widget takes the right button for spectators (MousePress returns true; layer -1000000 so it comes first in
  BAR's input order), switches to the Spring camera (viewspring) and changes its state rx (tilt, clamped to
  0.51-0.99 pi) and ry (turn) by camera_rotation_speed (default 0.003 rad per pixel; 0.005 felt too fast), around
  the point looked at; signs chosen by the user's test (ry + dx, rx + dy), each axis invertible
  (invert_horizontal_rotation, invert_vertical_rotation). The engine's CamSpringLockCardinalDirections (on by
  default) froze the rotation over about 18 degrees at each cardinal direction (GetRotationWithCardinalLock, lock
  width 0.2 x 90 degrees) while ry kept moving: off unless GraphicsConfiguration.lock_cardinal_directions. GraphicsConfiguration.show
  (default False) opens the window; engine.graphics is never None. BAR's
  keybind profiles may override a KeybindingFile, hence binds sent from a widget. Each team has a controller
  (controllers.py, agreed 2026-10-18: DoNothing, Passive, HoldAndFire, AttackNearest, BuiltInArtificialIntelligence,
  Agent) and a side; a controller gives the engine AI (NullAI for scripted ones), the placed units' states and an
  optional gadget script (mod option reinforcement_learning_controllers, "team,script,params/..."; SCRIPTS table in
  the gadget). GameConfiguration(scenario, [AttackNearest(), Passive()]) runs an agent scenario as games. Planned later: a PythonPolicy
  controller (a Python policy or a trained checkpoint through a BridgeAI, for the evaluation league).
  ArtificialIntelligencePlayer.label/display_name names players in charts and tables. Observation counts left to None follow
  the scenario. Armor types are the constant NUMBER_OF_ARMOR_TYPES (checked against the engine at start).
  Runs live in configurations/runs/ (one team per alliance: statistics count players as sides).
- Spawn zones (agreed 2026-10-18): each placed unit starts uniformly at random in its team's SpawnZone (fractions of
  the map); default from TeamConfiguration.spawn_depth = share of the alliance's band (map width split in equal
  bands, in alliance order) from the edge, whole height, SPAWN_MARGIN 0.02 from the border; or an explicit
  spawn_zone (a point for deterministic spawns: one_versus_one). Replaced group_separation/group_spread/spawn_jitter/
  spawn_x_range. scenarios/armada_versus_cortex_battle.py: 300 vs 300 tier 1 mixed units on Red Comet, run
  configuration of the same name with two AttackNearest (headless: Cortex won in 3.7 game minutes, 24 s).
- Teams and alliances (agreed 2026-10-18): TeamConfiguration.allyteam groups teams (None = alone);
  ScenarioConfiguration.allyteams numbers them 0.. in order of appearance. Agents are the teams with Agent(), on any
  team and on several (self-play); agents are named team<t>_unit_<k>. One BridgeAI (first agent team) talks to
  Python for all agent teams; the others run NullAI. A Skirmish AI can only order its own team, so every agent order
  goes through the gadget ("rl_actions|snap|unit,cmd,x,z;...", attack snaps to the nearest enemy of the unit's
  alliance), and current orders too ("rl_commands|<first>", chunked: responses are capped at MAX_RESPONSE_SIZE
  10240). The BridgeAI lists getFriendlyUnits + getEnemyUnits. Observations, rewards and info are per alliance and
  relative (alliance 0 = the observer's), enemy orders hidden; the visibility channel is computed in Python per
  alliance from sight ranges (the engine LOS map exists only for the BridgeAI's alliance). The reset observation
  still lists the units the reset destroyed (commanders, previous episode) for one frame: ReinforcementLearningEnvironment keeps
  only the spawned ones (before 2026-10-18 the initial health, hence the reward scale, counted the commanders).
- Placed units in a game without Python: the gadget reads mod option reinforcement_learning_scenario
  ("team,unit,x,z/...|states", "/" because ";" ends a start-script value) in GamePreload and spawns the units;
  deathmode=killall. No commander at all (since 2026-10-18, before they were spawned then destroyed, and their spawn
  effects still showed): with placed units, games and the environment set BAR's scenariooptions (game.py
  no_commander_mod_options: base64url JSON without padding, unitloadout = one placeholder entry naming no unit,
  disablefactionpicker), so game_initial_spawn.lua spawns the loadout instead of commanders and
  unit_scenario_loadout.lua creates nothing. Side effects of scenariooptions in BAR's interface: the Mission Info
  panel opens and hides the rest of the interface (graphics_settings.lua closes it), the top bar hides the changelog
  and team stats buttons. editor_sandbox=1 would also skip commanders but disables game_end. Games with placed units,
  full vision or scripted controllers use the game extension as GameType (start_engine writes it; replay.py too).
- BARb with only placed units (one_versus_one with BARb, 2026-10-18): it mostly leaves its Pawn idle (no damage in 6
  minutes, once 54 damage just before 2 minutes, same seed): not a useful baseline for unit-only scenarios.
- The game extension (archive "Beyond All Reason Reinforcement Learning 1", depends on the pinned BAR) is written
  into <run>/games/ and is the GameType; its widget keeps LuaSocket (BAR only sandboxes user widgets).
- Every unsynced -> synced message (LuaRulesMsg, AI commands too) goes through the server, which simulates ahead of
  the blocked client by about speed x block time, capped near 30 frames but up to ~60 at high speed. Pausing +
  `singlestep` gives exact frames but the engine freezes CommandAI while paused (units ignore orders): unusable.
  Solution in place: actions carry their decision frame, the gadget applies them at decision + action delay (32)
  or at once when late, publishes reinforcement_learning_applied_decision, and the widget observes only after it.
  Measured at speed 1000: ~160 steps/s, decisions every 32-40 frames (exact at speed <= 80, half the speed).
  The user refused to recompile the engine.
- Both teams are NullAI 0.1 (does nothing); BAR commanders are destroyed at the first reset; deathmode=neverend.
  Full vision (Spring.SetGlobalLos) is needed: the engine drops attacks on units the team does not see.
- Gotchas: Spring.GetGameRulesParam returns no value at all (not nil) for an absent parameter; destroyed units stay
  listed until the next frame (filter with Spring.GetUnitIsDead); BAR's cmd_paused_is_paused gadget blocks orders
  while paused unless modoption allowpausegameplay=1; within a frame, gadgets' GameFrame runs before widgets'.

## How the engine is driven (learned in step 0)
- Versions pinned in `src/beyond_all_reason_environment/installation.py` (engine = the one the
  official launcher uses, see BYAR-Chobby `dist_cfg/config.json`).
- `game_data/` is shared read-only through `SPRING_DATADIR`; each match gets its own write dir
  `runs/<name>/` (`--isolation --write-dir`), so matches can run in parallel.
- Speed: `MinSpeed`/`MaxSpeed` must be in `[MODOPTIONS]` (engine default max is 20), plus
  `debugcommands=1:setspeed 1000` (executed by the game's `cmd_dev_helpers.lua` gadget).
  Measured ≈ x30–45 per match on this machine, ~4.7 GB RAM per match.
- Our Lua user widgets are copied to `runs/<name>/LuaUI/Widgets/`. User widgets may only write
  in `LuaUI/Config/` or `LuaUI/Widgets/` (`checkUserWritePath` in `luaui/barwidgets.lua`).
- Graphical mode (EngineConfiguration.graphics or `--graphical`, binary `spring`): needs SDL2/X11/OpenAL from `flake.nix`
  (`BEYOND_ALL_REASON_GRAPHICS_LIBRARY_PATH`). Default renderer is the Intel iGPU (≈ x5 max);
  with NVIDIA PRIME offload variables + `SDL_VIDEODRIVER=x11` it uses the RTX (x10 reached).
- Checks: `ruff format`, `ruff check`, `basedpyright --pythonpath <env>/bin/python`, `pytest` (tests/ never start
  the engine). Charts: matplotlib, player colors fixed by position (blue, orange...).
- Replays: `RecordDemo=1` works headless (engine writes `demos/*.sdfz`, ~30 kB per 3 game minutes).
  Playback = graphical `spring <file.sdfz>`; the demo replays our `setspeed 1000` debug command, so
  `replay.py` adds `lua/replay_widgets/replay_speed.lua` to set the speed back at frame 2.
- After each match `_tidy_run_directory` keeps only script.txt, match_result.json, engine_output.log,
  replay.sdfz, charts/ (unless `keep_engine_files`). Run layout: runs/<name>/version_<n>/{overview.png,charts/,matches/<match>/{overview.png,charts/}};
  each run of a configuration adds a version (`new_version_directory`), nothing is overwritten.
- Charts (`plots.py`): `METRICS` registry (name, title, category, compute from all histories); every
  metric gets one file per match and one aggregated file; overviews pick 6 (`MATCH_OVERVIEW`, `RUN_OVERVIEW`).
  The user's key metrics are metal income, energy income and units alive (`KEY_METRICS`, top row of overviews).
- One color = one entry of `players` (the user wants to compare factions and, later, AIs). `side` is a faction or a
  list (drawn per match from the seed). No side swapping: start positions are already random per seed.
- Aggregated curves in game minutes suffer survivor bias (late values = long matches only); hence thin per-match
  lines, the "fewer than half running" note, and `over_time_relative/` (x = share of each match's duration).
- Analysis: `tables.py` writes runs/<name>/version_<n>/tables/{matches,players,timeseries}.parquet (match_id,
  experiment, version on every row; metrics from `metrics.py`, shared with `plots.py`); `analysis.py` (Wilson
  intervals, aggregation over time); web interface `results_browser.py` (Streamlit, launched by
  scripts/browse_results.py; tested headless with `streamlit.testing.v1.AppTest`). The widget records start
  positions (`Spring.GetTeamStartPosition`) and map size since this change; older matches have none.
- `map_name` may be a list (matches take maps in turn). Map sizes and 1v1 pools: lobby_maps.validated.json at
  maps-metadata.beyondallreason.dev. Downloaded: Red Comet Remake 1.8 (12x8), Comet Catcher Remake 1.8 (16x12),
  Archsimkats_Valley_V1 (16x16). The web interface has a map filter and wins per map.
- Ideas agreed but not done: mirrored matches (swap spawns between twin matches) to
  cancel map/spawn bias, logistic regression win ~ side + map + spawn.
- Legion needs mod option `experimentallegionfaction=1` (added automatically); BARb then plays it.
- The engine logs no winner; `match_result_reporter.lua` hooks `widget:GameOver` (and `Shutdown`
  for timeouts) and writes winner + full `Spring.GetTeamStatsHistory` per team (one entry per 15 s, cumulative).
- Parallel: 4 matches at once run at ≈ x20 each (one alone ≈ x35); the engine uses ~2.3 cores.
- BARb vs BARb is NOT reproducible with the same `FixedRNGSeed` (55, 33, 46 min, timeout).
  Probably BARb itself (threads/timing); to investigate before relying on seeds.
- The game repository has its own headless test tooling: `tools/headless_testing/` and
  `luaui/Tests/` (in `reference/Beyond-All-Reason`, a read-only clone; `reference/RecoilEngine`
  is the engine source at the pinned tag).

## Open questions
- Which interface does the user's reinforcement learning library expect
  (Gymnasium, PettingZoo, multi-agent, Dict spaces)?
- Target: full game vs SMAC-style scenarios first?
