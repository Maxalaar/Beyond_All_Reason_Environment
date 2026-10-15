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
- No docker, gcc or cmake installed globally → use a Nix flake / `nix develop` shell.
- Prebuilt Linux binaries (Recoil engine) need an FHS environment on NixOS
  (`buildFHSEnv`, `steam-run` or `nix-ld`).

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
- [ ] Step 0: download Recoil + Beyond All Reason + a small map, run BARb vs BARb headless,
      `run_match.py` returns the winner.
- [ ] Step 1: Lua widget/gadget dumps game state (units: pos, hp, team; team
      resources) to JSON every N frames.
- [ ] Step 2: 2D viewer (top-down map, team colours, time slider, resource curves).
- [ ] Step 3: BridgeAI + Python socket, synchronous step().
- [ ] Step 4: Gymnasium/PettingZoo environment on a small SMAC-like micro scenario.
- [ ] Step 5: train with the user's reinforcement learning library; evaluation league vs BARb/STAI.

## Open questions
- Which interface does the user's reinforcement learning library expect
  (Gymnasium, PettingZoo, multi-agent, Dict spaces)?
- Target: full game vs SMAC-style scenarios first?
