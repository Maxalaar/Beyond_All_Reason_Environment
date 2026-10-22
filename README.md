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
python scripts/run.py                                              # one game BARb against BARb, prints the winner
python scripts/run.py configurations/runs/one_versus_one_attack_nearest.py   # watch a game in a window
python scripts/run.py configurations/runs/barb_versus_barb.py --matches 20 --parallel 2   # a series, statistics and charts
python scripts/watch_replay.py runs/barb_versus_barb/version_1/matches/match_000_seed1 --speed 5   # replay a recorded game
python scripts/plot_run.py runs/barb_versus_barb/version_1          # redraw all charts of a run, no replay needed
python scripts/browse_results.py                                    # web interface to browse all results
```

The configuration is made of nested parts, from the situation to how it runs:

```
ScenarioConfiguration ──► GameConfiguration ──┬──► RunConfiguration                              ──► scripts/run.py
  (the situation)         (+ who plays        │      (+ number of games, seeds, replays, engine)      (1 or N games)
                           each team)         │
                                              └──► ReinforcementLearningEnvironmentConfiguration ──► ReinforcementLearningEnvironment
                                                     (+ observation, actions, rewards, engine)          (agents, RLlib)
```

- `ScenarioConfiguration` (predefined ones in `beyond_all_reason_environment.scenarios`): map, time limit, and the
  teams (`TeamConfiguration`): for each, its faction (`side`), its alliance (`allyteam`: teams with the same number
  are allies; by default each team is alone) and how it starts: normally (a commander), or with placed units
  (`units`). It says nothing of who plays. Placed units start at random in their team's spawn zone: by default
  `spawn_depth` (in [0, 1]), the share of the team's alliance band taken from the map edge (the alliances split the
  map width into equal bands: with two, the west and east halves; 1 on both sides gives 50 % / 50 %, 0.2 keeps each
  army in the fifth of its half nearest its edge), or an explicit `spawn_zone=SpawnZone(x_minimum, x_maximum,
  z_minimum, z_maximum)` in fractions of the map (a point when minimum equals maximum).
- `GameConfiguration(scenario, controllers)`: one controller per team, in the order of `scenario.teams`.
- `RunConfiguration`: games of a `GameConfiguration` without Python, one or a series in parallel (`matches`,
  `parallel`, `first_seed`, `record_replays`), and `engine`: speed and window. Runs count one team per alliance.
- `ReinforcementLearningEnvironmentConfiguration`: the environment for the agents (see below).

| Controller (`controllers.py`) | The team |
|---|---|
| `DoNothing()` | its units keep their default states and receive no order |
| `Passive()` | does not move, does not shoot |
| `HoldAndFire()` | does not move, shoots what comes in range |
| `AttackNearest(interval_seconds=1.0)` | every interval, each unit attacks the nearest enemy (scripted in the game extension's gadget, no Python) |
| `BuiltInArtificialIntelligence("BARb")` | an artificial intelligence of the game |
| `Agent()` | the agents of the reinforcement learning environment, played from Python (any number of teams: several for self-play) |

The window is configured in the file too, `engine=EngineConfiguration(graphics=GraphicsConfiguration(...))`:
`show` (open the window, off by default; the other settings stay in the file when it is off), `sound` (off by
default), `fullscreen` (a borderless window over the screen, else 1600x900), `camera_keys` (`"zqsd"` by default:
forward, left, back, right, Shift for faster), `right_button_rotation` (on by default: hold the right mouse button
and move the mouse to rotate the camera around the point it looks at, as in 3D editors; spectators give no orders,
so the button is free; the game's Alt + middle button keeps working), `camera_rotation_speed` (radians per pixel,
0.003 by default), `invert_horizontal_rotation` and `invert_vertical_rotation` (off by default: the view follows the
mouse), `lock_cardinal_directions` (off by default: the engine would hold the rotation for a moment at north, east,
south and west) and, for `camera_keys`, `None` keeps the
game's arrows) and `close_delay_seconds` (the window closes this long after the end, `None` leaves it open).
`engine.speed` left to `None` is real time with a window and as fast as possible without; games run one at a time
with a window.

A scenario made for the agents runs as games once another controller plays their team, for example
`GameConfiguration(one_versus_one, [AttackNearest(), Passive()])` (see
`configurations/runs/one_versus_one_attack_nearest.py`). Charts and tables name each player after its controller.
Each player keeps its color in every chart. Its `side` is one faction (`"Armada"`, `"Cortex"` or `"Legion"`) or a
list of them, in which case each game draws one (reproducibly from the seed) and `summary.txt` gives its wins per
faction. To compare factions, give the teams different factions; to compare controllers, give them different
controllers. `map_name` is one map or a list: games then take the maps in turn, so each map gets the same number of
games (see `configurations/runs/factions_by_map_size.py`). Download maps first with
`python scripts/download_game_data.py --map "<name>"`. Copy `barb_versus_barb.py` to start a new experiment;
the editor completes and type-checks the fields.

`scripts/run.py` takes a file of `configurations/runs/` (default: `barb_versus_barb.py`); the simple fields of the
run, of its scenario, of its engine and of its window can be replaced on the command line with `--<field> <value>`,
for example `--map_name "Comet Catcher Remake 1.8" --maximum_game_minutes 30 --first_seed 7 --no-record_replays`,
`--show` / `--no-show`, `--sound` (`--help` lists them). `--match_index <i>` runs only game i of the run
(seed `first_seed + i`, map `i % number of maps`). Results go to `runs/<name>/`:

```
runs/<name>/
├── version_1/                  first run of the configuration; running it again adds version_2/, ...
│   ├── overview.png            the six most telling charts, aggregated over all matches
│   ├── configuration.py        copy of the configuration file used
│   ├── configuration.json      the values actually used, after the command line, and the command line
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

## Reinforcement learning environment

`ReinforcementLearningEnvironment` is a fight between alliances of teams, one agent (`team<t>_unit_<k>`) per unit of every team
whose controller is `Agent()`, with the parallel interface
of PettingZoo (`reset`, `step`, `close`, one dictionary entry per agent) and gymnasium spaces;
`RayMultiAgentEnvironment` wraps it for RLlib (`pip install -e ".[rllib]"`). Its
`ReinforcementLearningEnvironmentConfiguration` is made
of separate parts, each answering one question:

| Part | Class | Question |
|---|---|---|
| `game` | `GameConfiguration` | the scenario (map, teams and alliances, placed units, start positions, episode length) and who plays each team: `Agent()` on the agents' teams, any other controller elsewhere, allies included; the same class as in runs |
| `observation` | `ObservationConfiguration` | what each agent sees: entity counts (left to `None`, they follow the scenario), weapon slots, terrain grid |
| `action` | `ActionConfiguration` | what each agent can do: commands per unit type, attack snap radius, decision interval |
| `rewards` | `RewardWeights` | how the agents are rewarded |
| `engine` | `EngineConfiguration` | how the engine runs (window, speed), without effect on the results |

Predefined scenarios are in `beyond_all_reason_environment.scenarios` (`one_versus_one`, `five_versus_five_arena`: the
1v1 arena with 5 Pawns per side, `five_versus_five`,
`armada_versus_cortex`, `armada_versus_cortex_battle`: 300 units per army, bots, vehicles, artillery and aircraft)
and only define the situation: an experiment, for example a training configuration of
the reinforcement learning library, chooses the scenario, the controllers and the other parts. Self-play is a game
with `Agent()`
on both sides, for example `GameConfiguration(five_versus_five, [Agent(), Agent()])`.

Every agent sees the game from its own alliance: allies are the units of its alliance, enemies those of the others
(alliance numbers in the observation are relative, 0 for its own), the current orders of enemy units are hidden, the
visibility channel of the terrain is its alliance's (sight ranges, without terrain occlusion), and the reward is its
alliance's. An episode ends when one alliance at most still has units.

```python
from beyond_all_reason_environment import (
    Agent, GameConfiguration, Passive, ReinforcementLearningEnvironmentConfiguration, RewardWeights,
)
from beyond_all_reason_environment.reinforcement_learning_environment import ReinforcementLearningEnvironment
from beyond_all_reason_environment.scenarios import one_versus_one

configuration = ReinforcementLearningEnvironmentConfiguration(
    game=GameConfiguration(one_versus_one, controllers=[Agent(), Passive()]), rewards=RewardWeights(kill=0.0)
)
environment = ReinforcementLearningEnvironment(configuration)
```

```bash
python scripts/try_reinforcement_learning_environment.py --policy attack --episodes 5            # check and measure speed
python scripts/try_reinforcement_learning_environment.py --policy random --opponent HoldAndFire
python scripts/try_reinforcement_learning_environment.py --scenario one_versus_one --opponent AttackNearest   # an opponent that attacks
python scripts/try_reinforcement_learning_environment.py --opponent Agent   # self-play: the same policy plays both sides
python scripts/try_reinforcement_learning_environment.py --policy attack --show --speed 2   # watch it
python scripts/try_ray_multi_agent_environment.py                              # through the RLlib wrapper
```

- **Action**: `{"command": index in COMMANDS, "target": [x, z]}` with `(x, z)` a point of the map in `[0, 1]²`;
  commands: continue, move, fight (attack-move), attack (the enemy nearest to the point), stop.
- **How it runs**: one engine process; episodes reset inside the match. A native Skirmish AI (BridgeAI) exchanges
  observations and actions with Python over a socket while the simulation waits, and a gadget of a small game
  extension spawns the units at each reset.

## Analysis

Each run also writes Parquet tables in `runs/<name>/version_<n>/tables/`:

| Table | One row per | Main columns |
|---|---|---|
| `matches.parquet` | match | `match_id`, experiment, version, seed, map, outcome, winner, game minutes, engine and game versions |
| `players.parquet` | player in a match | artificial intelligence, side, start position, `won`, final statistics |
| `timeseries.parquet` | player and statistics entry (every 15 s) | raw engine statistics and the metrics of the charts (`metal_income`, `units_alive`...) |

Every row has `match_id`, `experiment` and `version`, so the tables of all runs can be analysed
together, with pandas (`tables.read_tables`) or DuckDB:

```sql
SELECT side, avg(won::int) FROM 'runs/*/version_*/tables/players.parquet' GROUP BY side
```

`python scripts/browse_results.py` opens a web interface (Streamlit) on these tables: choose experiments
and versions, compare by player, faction or artificial intelligence, see win rates with their 95 %
confidence interval, curves over time (in game minutes or share of each match's duration), end-of-match
statistics, start positions, and each match with its replay command. Runs played before the tables
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
