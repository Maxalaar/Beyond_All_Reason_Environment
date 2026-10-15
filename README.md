# Beyond All Reason reinforcement learning environment

Headless test harness (and, later, a Gymnasium/PettingZoo environment) for the real-time strategy
game [Beyond All Reason](https://www.beyondallreason.info/).

## Installation (once)

Any Python ≥ 3.11 environment works, for example a conda environment:

```bash
conda create -n BeyondAllReasonReinforcementLearning python=3.13
conda activate BeyondAllReasonReinforcementLearning
pip install -e .                      # editable: code changes apply without reinstalling
python scripts/download_game_data.py  # engine + game + map into game_data/ (~2 GB)
```

No `nix develop` needed: on NixOS the libraries of the graphical engine are built by Nix
from `flake.nix` the first time a window is opened. `nix develop` still works if preferred.

## Usage

```bash
python scripts/run_match.py                         # one match without graphics, prints the winner
python scripts/run_match.py --graphical --speed 3   # watch a match in a window (+ and - change the speed)
python scripts/run_matches.py configurations/barb_versus_barb.yaml   # several matches, then statistics
```

`run_matches.py` reads everything from a YAML file in `configurations/`: the name of the run, the
players, the number of matches, how many run in parallel, seeds, map, time limit. Copy
`barb_versus_barb.yaml` to start a new experiment. Results go to `runs/<name>/`:

| File | Content |
|---|---|
| `configuration.yaml` | copy of the configuration file used |
| `summary.txt` | statistics table: wins, durations, mean metal, damage and units per player |
| `summary.json` | the same plus every match, for later analysis |
| `match_000_seed1/` … | start script, engine logs and result of each match |

An existing `runs/<name>/` is never overwritten unless `--overwrite` is given.

Measured on a 20-core laptop: about x35 real time for one match alone, x20 each with 4 in parallel,
about 5 GB of memory per match.
