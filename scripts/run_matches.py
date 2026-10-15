"""Run the matches described by a configuration file, in parallel and without graphics, then print statistics.

Examples:
    python scripts/run_matches.py configurations/barb_versus_barb.py
    python scripts/run_matches.py configurations/barb_versus_barb.py --overwrite

Results go to runs/<name>/, where <name> comes from the configuration file:
    configuration.py     copy of the configuration file used
    summary.txt          the statistics table printed at the end
    summary.json         the summary and every match, for later analysis
    charts/              charts aggregated over the matches: overview.png, then outcomes/, over_time/, end_of_match/
    matches/match_000_seed1/
        script.txt, match_result.json, engine_output.log, replay.sdfz
        charts/          overview.png, then economy/, military/, units/
"""

import argparse
import shutil
import sys
from pathlib import Path

from beyond_all_reason_environment.batch import (
    format_summary,
    load_batch_configuration,
    run_matches,
    save_batch,
    summarise,
)
from beyond_all_reason_environment.match import RUNS_DIRECTORY, MatchResult
from beyond_all_reason_environment.plots import plot_batch


def describe(result: MatchResult) -> str:
    if result.outcome == "crash":
        return f"crash, see {result.run_directory / 'engine_output.log'}"
    winner = f"player {result.winner} wins" if result.outcome == "victory" else result.outcome
    return (
        f"{winner:<14} {result.game_minutes:5.1f} game min  "
        f"{result.wall_clock_seconds:4.0f} s  x{result.speed_factor:.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configuration", type=Path, help="Python configuration file, see configurations/")
    parser.add_argument("--overwrite", action="store_true", help="delete runs/<name>/ if it already exists")
    arguments = parser.parse_args()

    batch = load_batch_configuration(arguments.configuration)
    batch_directory = RUNS_DIRECTORY / batch.name
    if batch_directory.exists():
        if not arguments.overwrite:
            sys.exit(f"{batch_directory} already exists: change 'name' in the configuration, or add --overwrite.")
        shutil.rmtree(batch_directory)
    batch_directory.mkdir(parents=True)
    shutil.copy(arguments.configuration, batch_directory / "configuration.py")

    configurations = batch.match_configurations()
    print(f"{batch.name}: {batch.matches} matches, {batch.parallel} at a time, in {batch_directory}")

    def on_result(index: int, result: MatchResult) -> None:
        print(f"  match {index:3d} (seed {configurations[index].seed}): {describe(result)}", flush=True)

    results = run_matches(configurations, batch_directory, batch.parallel, on_result)
    summary = summarise(configurations, results)
    save_batch(batch_directory, configurations, results, summary)
    print("\n" + format_summary(summary))
    plot_batch(configurations, results, batch_directory / "charts")
    print(f"Saved in {batch_directory}, overview: {batch_directory / 'charts' / 'overview.png'}")


if __name__ == "__main__":
    main()
