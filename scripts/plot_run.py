"""Redraw all the charts of a finished run from its saved results, without replaying any match.

Useful after changing the charts (plots.py), or for a run copied from another machine.

Example:
    python scripts/plot_run.py runs/barb_versus_barb/version_1
"""

import argparse
import shutil
from pathlib import Path

from beyond_all_reason_environment.plots import CHARTS_FOLDER, OVERVIEW_FILE_NAME, plot_run
from beyond_all_reason_environment.run import load_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "version_directory", type=Path, help="runs/<name>/version_<n>, the folder that contains summary.json"
    )
    arguments = parser.parse_args()

    configurations, results = load_run(arguments.version_directory)
    # Start from empty chart folders, so charts that no longer exist do not linger.
    for directory in [arguments.version_directory, *(result.run_directory for result in results)]:
        shutil.rmtree(directory / CHARTS_FOLDER, ignore_errors=True)
        (directory / OVERVIEW_FILE_NAME).unlink(missing_ok=True)
    paths = plot_run(configurations, results, arguments.version_directory)
    print(f"{len(paths)} charts, overview: {arguments.version_directory / OVERVIEW_FILE_NAME}")


if __name__ == "__main__":
    main()
