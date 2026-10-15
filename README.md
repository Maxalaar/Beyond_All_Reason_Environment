# Beyond All Reason reinforcement learning environment

Headless test harness (and, later, a Gymnasium/PettingZoo environment) for the real-time strategy
game [Beyond All Reason](https://www.beyondallreason.info/).

## Installation (once)

Any Python ≥ 3.11 environment works, for example a conda environment:

```bash
conda create -n Beyond_All_Reason_Reinforcement_Learning python=3.13   # same name as the folder
conda activate Beyond_All_Reason_Reinforcement_Learning
pip install -e .                      # editable: code changes apply without reinstalling
python scripts/download_game_data.py  # engine + game + map into game_data/ (~2 GB)
```

No `nix develop` needed: on NixOS the libraries of the graphical engine are built by Nix
from `flake.nix` the first time a window is opened. `nix develop` still works if preferred.

`.vscode/settings.json` makes VSCodium use this environment by default.

## Usage

```bash
python scripts/run_match.py                         # one match without graphics, prints the winner
python scripts/run_match.py --graphical --speed 3   # watch a match in a window (+ and - change the speed)
python scripts/run_matches.py configurations/barb_versus_barb.py   # several matches, statistics and charts
python scripts/watch_replay.py runs/barb_versus_barb/version_1/matches/match_000_seed1 --speed 5   # replay a recorded match
python scripts/plot_run.py runs/barb_versus_barb/version_1          # redraw all charts of a batch, no replay needed
python scripts/browse_results.py                                    # web interface to browse all results
```

`run_matches.py` reads a Python file in `configurations/` that defines
`configuration = BatchConfiguration(...)`: the name of the run, the players, the number of matches,
how many run in parallel, seeds, map, time limit. Each player keeps its color in every chart. Its `side` is one
faction (`"Armada"`, `"Cortex"` or `"Legion"`) or a list of them, in which case each match draws one
(reproducibly from the seed) and `summary.txt` gives its wins per faction. To compare factions, give
the players different factions; to compare artificial intelligences, give them different `short_name`.
`map_name` is one map or a list: matches then take the maps in turn, so each map gets the same number of
matches (see `configurations/factions_by_map_size.py`). Download maps first with
`python scripts/download_game_data.py --map "<name>"`. Copy `barb_versus_barb.py` to start a new experiment;
the editor completes and type-checks the fields. Results go to `runs/<name>/`:

```
runs/<name>/
├── version_1/                  first run of the configuration; running it again adds version_2/, ...
│   ├── overview.png            the six most telling charts, aggregated over all matches
│   ├── configuration.py        copy of the configuration file used
│   ├── summary.txt             statistics table: wins, durations, mean metal, damage, units per player
│   ├── summary.json            the same plus every match
│   ├── tables/                 matches.parquet, players.parquet, timeseries.parquet (see below)
│   ├── charts/                 all the aggregated charts
│   │   ├── outcomes/           wins.png, durations.png
│   │   ├── over_time/          per player over game time: median, interquartile band, one thin line per match
│   │   │   ├── economy/        metal and energy income, spending, totals, waste, share of metal
│   │   │   ├── military/       damage per second and totals, damage efficiency, units killed and lost
│   │   │   ├── units/          units alive, produced, production rate
│   │   │   └── matches_running.png
│   │   ├── over_time_relative/ the same with time as a share of each match's duration (no survivor bias)
│   │   └── end_of_match/       final statistics, one dot per match (all.png gathers them)
│   └── matches/
│       └── match_000_seed1/
│           ├── overview.png        the six most telling charts of this match
│           ├── script.txt          start script
│           ├── match_result.json   winner and full statistics history (every 15 s of game time)
│           ├── engine_output.log   engine logs
│           ├── replay.sdfz         replay, when record_replays=True (a few hundred kB)
│           └── charts/             economy/, military/, units/ (same metrics, this match only)
└── version_2/ ...
```

Everything else the engine writes in a match folder (cache, interface settings, about 5 MB) is
deleted after the match, unless `keep_engine_files=True`.

Earlier versions are never modified: a new run of the same configuration goes to the next `version_<n>/`.

Measured on a 20-core laptop: about x35 real time for one match alone, x20 each with 4 in parallel,
about 5 GB of memory per match.

## Micro environment (reinforcement learning)

`MicroEnvironment` is a small fight between two groups of units, one agent per unit of team 0, with the
parallel interface of PettingZoo (`reset`, `step`, `close`, one dictionary entry per agent) and gymnasium spaces.
Settings: `MicroScenarioConfiguration` (units of each side, opponent behaviour, episode length, decision
interval, number of entities seen, commands allowed per unit type, rewards).

```bash
python scripts/try_micro_environment.py --policy attack --episodes 5            # check and measure speed
python scripts/try_micro_environment.py --policy random --opponent hold_and_fire
python scripts/try_micro_environment.py --policy attack --graphical --speed 2   # watch it
```

- **Observation** of each agent: its own unit (`self`) and the `maximum_entities_seen` nearest units
  (`entities`, with `entity_mask`), positions as fractions of the map plus scale features (map size, distance
  covered per decision, weapon and sight ranges as fractions of the map), and `command_mask`.
- **Action**: `{"command": index in COMMANDS, "target": [x, z]}` with `(x, z)` a point of the map in `[0, 1]²`;
  commands: continue, move, fight (attack-move), attack (the enemy nearest to the point), stop.
- **How it runs**: one engine process; episodes reset inside the match. A small game extension adds a widget,
  which exchanges observations and actions with Python over a socket while the simulation waits, and a gadget,
  which spawns the units and gives the orders. Actions take effect about one decision later (32 frames, a little
  more when the engine server runs ahead); each observation shows the order the unit is executing.
- About 160 steps per second with 5 against 5 (one step is one decision of every agent, about 1 game second).

## Analysis

Each batch also writes Parquet tables in `runs/<name>/version_<n>/tables/`:

| Table | One row per | Main columns |
|---|---|---|
| `matches.parquet` | match | `match_id`, experiment, version, seed, map, outcome, winner, game minutes, engine and game versions |
| `players.parquet` | player in a match | artificial intelligence, side, start position, `won`, final statistics |
| `timeseries.parquet` | player and statistics entry (every 15 s) | raw engine statistics and the metrics of the charts (`metal_income`, `units_alive`...) |

Every row has `match_id`, `experiment` and `version`, so the tables of all batches can be analysed
together, with pandas (`tables.read_tables`) or DuckDB:

```sql
SELECT side, avg(won::int) FROM 'runs/*/version_*/tables/players.parquet' GROUP BY side
```

`python scripts/browse_results.py` opens a web interface (Streamlit) on these tables: choose experiments
and versions, compare by player, faction or artificial intelligence, see win rates with their 95 %
confidence interval, curves over time (in game minutes or share of each match's duration), end-of-match
statistics, start positions, and each match with its replay command. Batches played before the tables
existed need `python scripts/export_tables.py` once.

## Development

```bash
pip install -e ".[development]"   # ruff, basedpyright, pytest
ruff format . && ruff check .     # formatting and linting (also on save in VSCodium)
basedpyright                      # type checking
pytest                            # fast tests, the game engine is not started
```

VSCodium extensions: Python, basedpyright, Ruff, YAML, Even Better TOML and Lua, listed in
`.vscode/extensions.json`; settings in `.vscode/settings.json` and `pyproject.toml`.
