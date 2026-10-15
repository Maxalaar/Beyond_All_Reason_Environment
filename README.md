# Beyond All Reason reinforcement learning environment

Headless test harness (and, later, a Gymnasium/PettingZoo environment) for the real-time strategy
game [Beyond All Reason](https://www.beyondallreason.info/).

## Quick start (Linux, Nix)

```bash
nix develop                          # Python, 7z, git
python scripts/download_game_data.py # engine + game + map into game_data/ (~2 GB)
python scripts/run_match.py --seed 1 # BARb vs BARb without graphics, prints the winner
python scripts/run_match.py --graphical --speed 3  # watch a match in a window
```

In the window, `+` and `-` change the game speed. Graphical matches render on the NVIDIA card when present.

Each match writes its start script, logs and result into `runs/<date>_seed<seed>/`.
