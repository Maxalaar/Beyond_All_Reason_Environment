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
- [x] Step 0: `scripts/download_game_data.py`, `scripts/run_match.py` (BARb vs BARb headless → winner,
      `--graphical` to watch), `scripts/run_matches.py configurations/<file>.py`
      (Python file defining `configuration = BatchConfiguration(...)`; parallel batch, statistics and
      charts into `runs/<name>/`; `batch.py`, `plots.py`).
- [ ] Step 1: Lua widget/gadget dumps game state (units: pos, hp, team; team
      resources) to JSON every N frames.
- [ ] Step 2: 2D viewer (top-down map, team colours, time slider, resource curves).
- [ ] Step 3: BridgeAI + Python socket, synchronous step().
- [ ] Step 4: Gymnasium/PettingZoo environment on a small SMAC-like micro scenario.
- [ ] Step 5: train with the user's reinforcement learning library; evaluation league vs BARb/STAI.

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
- Graphical mode (`--graphical`, binary `spring`): needs SDL2/X11/OpenAL from `flake.nix`
  (`BEYOND_ALL_REASON_GRAPHICS_LIBRARY_PATH`). Default renderer is the Intel iGPU (≈ x5 max);
  with NVIDIA PRIME offload variables + `SDL_VIDEODRIVER=x11` it uses the RTX (x10 reached).
- Checks: `ruff format`, `ruff check`, `basedpyright --pythonpath <env>/bin/python`, `pytest` (tests/ never start
  the engine). Charts: matplotlib, player colors fixed by position (blue, orange...).
- Replays: `RecordDemo=1` works headless (engine writes `demos/*.sdfz`, ~30 kB per 3 game minutes).
  Playback = graphical `spring <file.sdfz>`; the demo replays our `setspeed 1000` debug command, so
  `replay.py` adds `lua/replay_widgets/replay_speed.lua` to set the speed back at frame 2.
- After each match `_tidy_run_directory` keeps only script.txt, match_result.json, engine_output.log,
  replay.sdfz, charts/ (unless `keep_engine_files`). Batch layout: runs/<name>/version_<n>/{overview.png,charts/,matches/<match>/{overview.png,charts/}};
  each run of a configuration adds a version (`new_version_directory`), nothing is overwritten.
- Charts (`plots.py`): `METRICS` registry (name, title, category, compute from all histories); every
  metric gets one file per match and one aggregated file; overviews pick 6 (`MATCH_OVERVIEW`, `BATCH_OVERVIEW`).
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
