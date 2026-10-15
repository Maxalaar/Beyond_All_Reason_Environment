"""Run one headless match between two built-in artificial intelligences and print the winner.

Example: python scripts/run_match.py --seed 3
"""

import argparse

from beyond_all_reason_environment import ArtificialIntelligencePlayer, MatchConfiguration, run_match
from beyond_all_reason_environment.installation import DEFAULT_MAP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--first", default="BARb", help="short name of the first artificial intelligence")
    parser.add_argument("--second", default="BARb", help="short name of the second artificial intelligence")
    parser.add_argument("--maximum-game-minutes", type=float, default=60.0)
    arguments = parser.parse_args()

    configuration = MatchConfiguration(
        players=[
            ArtificialIntelligencePlayer(short_name=arguments.first, side="Armada"),
            ArtificialIntelligencePlayer(short_name=arguments.second, side="Cortex"),
        ],
        map_name=arguments.map,
        seed=arguments.seed,
        maximum_game_minutes=arguments.maximum_game_minutes,
    )
    result = run_match(configuration)

    if result.game_frames is None:
        print(f"No result (timeout or crash). Logs: {result.run_directory / 'engine_output.log'}")
        return
    if result.winner is None:
        print("Draw.")
    else:
        player = configuration.players[result.winner]
        print(f"Winner: player {result.winner} ({player.short_name}, {player.side})")
    print(
        f"{result.game_minutes:.1f} game minutes in {result.wall_clock_seconds:.0f} s "
        f"(speed x{result.speed_factor:.0f}). Logs: {result.run_directory}"
    )


if __name__ == "__main__":
    main()
