"""Run several matches without graphics in parallel and print statistics.

Examples:
    python scripts/run_matches.py --matches 5
    python scripts/run_matches.py --matches 20 --parallel 6 --alternate-sides
    python scripts/run_matches.py --first BARb --second SimpleAI --matches 10

Every match is saved under runs/<date>_batch/, with a summary.json for later analysis.
"""

import argparse
from datetime import datetime

from beyond_all_reason_environment import ArtificialIntelligencePlayer, MatchConfiguration
from beyond_all_reason_environment.batch import run_matches, save_batch, summarise
from beyond_all_reason_environment.installation import DEFAULT_MAP
from beyond_all_reason_environment.match import RUNS_DIRECTORY, MatchResult

SIDES = ("Armada", "Cortex")


def describe(result: MatchResult) -> str:
    if result.outcome == "crash":
        return f"crash, see {result.run_directory / 'engine_output.log'}"
    winner = f"player {result.winner} wins" if result.outcome == "victory" else result.outcome
    return f"{winner:<14} {result.game_minutes:5.1f} game min  {result.wall_clock_seconds:4.0f} s  x{result.speed_factor:.0f}"


def print_summary(summary: dict) -> None:
    outcomes = summary["outcomes"]
    print(f"\n{summary['matches']} matches: {outcomes['victory']} victories, {outcomes['draw']} draws, "
          f"{outcomes['timeout']} timeouts, {outcomes['crash']} crashes")
    durations = summary["game_minutes"]
    if durations["mean"] is not None:
        print(f"Game duration: mean {durations['mean']:.1f} min "
              f"(from {durations['minimum']:.1f} to {durations['maximum']:.1f}), "
              f"mean speed x{summary['mean_speed_factor']:.0f}")

    players = summary["players"]
    rows = [
        ("", [f"player {index}: {player['name']}" for index, player in enumerate(players)]),
        ("sides", [" / ".join(player["sides"]) for player in players]),
        ("wins", [f"{player['wins']} ({100 * player['wins'] / summary['matches']:.0f} %)" for player in players]),
    ]
    for key in ["metalProduced", "energyProduced", "damageDealt", "damageReceived",
                "unitsProduced", "unitsKilled", "unitsDied", "unitsAlive"]:
        rows.append((key, ["-" if player[key] is None else f"{player[key]:,.0f}" for player in players]))
    print("\nMean per match:")
    for label, values in rows:
        print(f"  {label:<16}" + "".join(f"{value:>24}" for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=4,
                        help="matches at the same time (each one uses about 5 GB of memory)")
    parser.add_argument("--first-seed", type=int, default=1, help="match i uses seed first-seed + i")
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--first", default="BARb", help="short name of the first artificial intelligence")
    parser.add_argument("--second", default="BARb", help="short name of the second artificial intelligence")
    parser.add_argument("--alternate-sides", action="store_true",
                        help="swap Armada and Cortex every other match, to cancel the faction advantage")
    parser.add_argument("--maximum-game-minutes", type=float, default=60.0)
    arguments = parser.parse_args()

    configurations = []
    for index in range(arguments.matches):
        sides = SIDES[::-1] if arguments.alternate_sides and index % 2 else SIDES
        configurations.append(MatchConfiguration(
            players=[
                ArtificialIntelligencePlayer(short_name=arguments.first, side=sides[0]),
                ArtificialIntelligencePlayer(short_name=arguments.second, side=sides[1]),
            ],
            map_name=arguments.map,
            seed=arguments.first_seed + index,
            maximum_game_minutes=arguments.maximum_game_minutes,
        ))

    batch_directory = RUNS_DIRECTORY / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_batch"
    print(f"{arguments.matches} matches, {arguments.parallel} at a time, in {batch_directory}")

    def on_result(index: int, result: MatchResult) -> None:
        print(f"  match {index:3d} (seed {configurations[index].seed}): {describe(result)}", flush=True)

    results = run_matches(configurations, batch_directory, arguments.parallel, on_result)
    summary = summarise(configurations, results)
    print_summary(summary)
    print(f"\nDetails: {save_batch(batch_directory, configurations, results, summary)}")


if __name__ == "__main__":
    main()
