"""Run the games described by a configuration file (one, or a series in parallel), then print statistics.

Examples:
    python scripts/run.py configurations/runs/one_versus_one_attack_nearest.py               # watch one game
    python scripts/run.py configurations/runs/barb_versus_barb.py --matches 20 --parallel 2
    python scripts/run.py configurations/runs/factions_by_map_size.py --match_index 2         # only game 2 of the run
    python scripts/run.py --map_name "Red Comet Remake 1.8" "Comet Catcher Remake 1.8" --maximum_game_minutes 30

Without a file, configurations/runs/barb_versus_barb.py is used. The fields of the run, of its scenario, of its engine
and of its window can be replaced on the command line (--<field> <value>, see --help); the teams and their
controllers stay in the file. With a window (--show, or engine.graphics.show in the file), games run one at a time.

Results go to runs/<name>/version_<n>/, where <name> comes from the configuration file; running it again
adds version_<n+1>/ next to the earlier ones:
    overview.png         the six most telling charts, aggregated over the games
    configuration.py     copy of the configuration file used
    configuration.json   the values actually used, after the command line, and the command line itself
    summary.txt          the statistics table printed at the end
    summary.json         the summary and every game
    tables/              matches.parquet, players.parquet, timeseries.parquet, for analysis (see tables.py)
    charts/              all the aggregated charts: outcomes/, over_time/, over_time_relative/, end_of_match/
    matches/match_000_seed1/
        overview.png, script.txt, match_result.json, engine_output.log, replay.sdfz
        charts/          economy/, military/, units/
"""

import argparse

from beyond_all_reason_environment.match import RUNS_DIRECTORY, MatchResult
from beyond_all_reason_environment.plots import plot_run
from beyond_all_reason_environment.run import (
    add_configuration_arguments,
    configuration_from_arguments,
    format_summary,
    new_version_directory,
    run_matches,
    save_configuration,
    save_run,
    summarise,
)
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
    add_configuration_arguments(parser)
    parser.add_argument(
        "--match_index",
        type=int,
        default=None,
        help="run only this game of the run: seed first_seed + index, map index %% number of maps",
    )
    arguments = parser.parse_args()

    run = configuration_from_arguments(arguments)
    if arguments.match_index is None:
        configurations = run.match_configurations()
    else:
        configurations = [run.match_configuration(arguments.match_index)]
    version_directory = new_version_directory(RUNS_DIRECTORY / run.name)
    save_configuration(version_directory, run, arguments)
    print(f"{run.name}: {len(configurations)} game(s), {run.games_at_once} at a time, in {version_directory}")

    def on_result(index: int, result: MatchResult) -> None:
        print(f"  game {index:3d} (seed {configurations[index].seed}): {describe(result)}", flush=True)

    results = run_matches(configurations, version_directory, run.games_at_once, on_result, run.engine.graphics)
    summary = summarise(configurations, results)
    save_run(version_directory, configurations, results, summary)
    write_tables(version_directory, configurations, results)
    print("\n" + format_summary(summary))
    plot_run(configurations, results, version_directory)
    print(f"Saved in {version_directory}, overview: {version_directory / 'overview.png'}")
    replays = [result.replay_path for result in results if result.replay_path is not None]
    if len(replays) == 1:
        print(f"Replay: python scripts/watch_replay.py {replays[0]}")


if __name__ == "__main__":
    main()
