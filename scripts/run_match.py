"""Run one match of a configuration file and print the winner.

The configuration is the one of run_matches.py (default: configurations/barb_versus_barb.py); this script
runs only one of its matches, the first one unless --match_index is given, and can show it in a window.
Any field of the file can be replaced on the command line.

Examples:
    python scripts/run_match.py
    python scripts/run_match.py configurations/factions_by_map_size.py --match_index 2
    python scripts/run_match.py --graphical --speed 3   # watch it; + and - change the speed in game
    python scripts/run_match.py --map_name "Comet Catcher Remake 1.8" --first_seed 7 --no-record_replays
"""

import argparse

from beyond_all_reason_environment import run_match
from beyond_all_reason_environment.batch import add_configuration_arguments, configuration_from_arguments
from beyond_all_reason_environment.plots import plot_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_configuration_arguments(parser)
    parser.add_argument(
        "--match_index",
        type=int,
        default=0,
        help="match of the series to run: seed first_seed + index, map index %% number of maps (default: 0)",
    )
    parser.add_argument("--graphical", action="store_true", help="open a window to watch the match")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="multiple of real time (default: 1 with --graphical, as fast as possible without)",
    )
    arguments = parser.parse_args()

    configuration = configuration_from_arguments(arguments).match_configuration(arguments.match_index)
    if arguments.speed is not None:
        configuration.speed = arguments.speed
    elif arguments.graphical:
        configuration.speed = 1.0
    result = run_match(configuration, graphical=arguments.graphical)

    if result.outcome == "crash":
        print(f"No result (crash or window closed). Logs: {result.run_directory / 'engine_output.log'}")
        return
    if result.outcome == "victory" and result.winner is not None:
        player = configuration.players[result.winner]
        print(f"Winner: player {result.winner} ({player.short_name}, {player.side})")
    else:
        print(f"No winner ({result.outcome}).")
    print(
        f"{result.game_minutes:.1f} game minutes in {result.wall_clock_seconds:.0f} s "
        f"(speed x{result.speed_factor:.0f}). Logs: {result.run_directory}"
    )
    if plot_match(result, configuration, result.run_directory):
        print(f"Charts: {result.run_directory / 'overview.png'}, details in {result.run_directory / 'charts'}")
    if result.replay_path is not None:
        print(f"Replay: python scripts/watch_replay.py {result.replay_path}")


if __name__ == "__main__":
    main()
