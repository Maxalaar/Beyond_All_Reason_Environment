"""Watch a recorded match in a window. + and - change the speed during the replay.

Matches are recorded when record_replays=True in the configuration file (or run_match.py --record-replay).

Examples:
    python scripts/watch_replay.py runs/barb_versus_barb/match_000_seed1/replay.sdfz
    python scripts/watch_replay.py runs/barb_versus_barb/match_000_seed1 --speed 5
"""

import argparse
from pathlib import Path

from beyond_all_reason_environment.match import REPLAY_FILE_NAME
from beyond_all_reason_environment.replay import watch_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path, help="replay.sdfz file, or the match folder that contains it")
    parser.add_argument("--speed", type=float, default=1.0, help="multiple of real time at the start")
    arguments = parser.parse_args()

    path = arguments.replay / REPLAY_FILE_NAME if arguments.replay.is_dir() else arguments.replay
    watch_replay(path, arguments.speed)


if __name__ == "__main__":
    main()
