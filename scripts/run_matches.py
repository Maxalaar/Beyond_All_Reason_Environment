"""Run the matches described by a configuration file, in parallel and without graphics, then print statistics.

Example:
    python scripts/run_matches.py configurations/barb_versus_barb.py

Results go to runs/<name>/version_<n>/, where <name> comes from the configuration file; running it again
adds version_<n+1>/ next to the earlier ones:
    overview.png         the six most telling charts, aggregated over the matches
    configuration.py     copy of the configuration file used
    summary.txt          the statistics table printed at the end
    summary.json         the summary and every match
    tables/              matches.parquet, players.parquet, timeseries.parquet, for analysis (see tables.py)
    charts/              all the aggregated charts: outcomes/, over_time/, over_time_relative/, end_of_match/
    matches/match_000_seed1/
        overview.png, script.txt, match_result.json, engine_output.log, replay.sdfz
        charts/          economy/, military/, units/
"""

import argparse
import shutil
from pathlib import Path

from beyond_all_reason_environment.batch import (
    format_summary,
    load_batch_configuration,
    new_version_directory,
    run_matches,
    save_batch,
    summarise,
)
from beyond_all_reason_environment.match import RUNS_DIRECTORY, MatchResult
from beyond_all_reason_environment.plots import plot_batch
from beyond_all_reason_environment.tables import write_tables


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
    arguments = parser.parse_args()

    batch = load_batch_configuration(arguments.configuration)
    batch_directory = new_version_directory(RUNS_DIRECTORY / batch.name)
    shutil.copy(arguments.configuration, batch_directory / "configuration.py")

    configurations = batch.match_configurations()
    print(f"{batch.name}: {batch.matches} matches, {batch.parallel} at a time, in {batch_directory}")

    def on_result(index: int, result: MatchResult) -> None:
        print(f"  match {index:3d} (seed {configurations[index].seed}): {describe(result)}", flush=True)

    results = run_matches(configurations, batch_directory, batch.parallel, on_result)
    summary = summarise(configurations, results)
    save_batch(batch_directory, configurations, results, summary)
    write_tables(batch_directory, configurations, results)
    print("\n" + format_summary(summary))
    plot_batch(configurations, results, batch_directory)
    print(f"Saved in {batch_directory}, overview: {batch_directory / 'overview.png'}")


if __name__ == "__main__":
    main()
