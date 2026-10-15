"""Run one match between two built-in artificial intelligences and print the winner.

Examples:
    python scripts/run_match.py --seed 3
    python scripts/run_match.py --graphical --speed 3   # watch it; + and - change the speed in game
"""

import argparse

from beyond_all_reason_environment import ArtificialIntelligencePlayer, MatchConfiguration, run_match
from beyond_all_reason_environment.installation import DEFAULT_MAP
from beyond_all_reason_environment.plots import plot_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--first", default="BARb", help="short name of the first artificial intelligence")
    parser.add_argument("--second", default="BARb", help="short name of the second artificial intelligence")
    parser.add_argument("--maximum-game-minutes", type=float, default=60.0)
    parser.add_argument("--graphical", action="store_true", help="open a window to watch the match")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="multiple of real time (default: 1 with --graphical, as fast as possible without)",
    )
    parser.add_argument("--record-replay", action="store_true", help="save replay.sdfz, see watch_replay.py")
    parser.add_argument(
        "--keep-engine-files", action="store_true", help="keep everything the engine writes, for debugging"
    )
    arguments = parser.parse_args()

    configuration = MatchConfiguration(
        players=[
            ArtificialIntelligencePlayer(short_name=arguments.first, side="Armada"),
            ArtificialIntelligencePlayer(short_name=arguments.second, side="Cortex"),
        ],
        map_name=arguments.map,
        seed=arguments.seed,
        maximum_game_minutes=arguments.maximum_game_minutes,
        record_replay=arguments.record_replay,
        keep_engine_files=arguments.keep_engine_files,
    )
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
