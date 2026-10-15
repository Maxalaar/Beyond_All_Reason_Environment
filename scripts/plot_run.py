"""Redraw all the charts of a finished batch from its saved results, without replaying any match.

Useful after changing the charts (plots.py), or for a batch copied from another machine.

Example:
    python scripts/plot_run.py runs/barb_versus_barb
"""

import argparse
import shutil
from pathlib import Path

from beyond_all_reason_environment.batch import load_batch
from beyond_all_reason_environment.plots import plot_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch_directory", type=Path, help="runs/<name>, the folder that contains summary.json")
    arguments = parser.parse_args()

    configurations, results = load_batch(arguments.batch_directory)
    # Start from empty chart folders, so charts that no longer exist do not linger.
    shutil.rmtree(arguments.batch_directory / "charts", ignore_errors=True)
    for result in results:
        shutil.rmtree(result.run_directory / "charts", ignore_errors=True)
    paths = plot_batch(configurations, results, arguments.batch_directory / "charts")
    print(f"{len(paths)} charts, overview: {arguments.batch_directory / 'charts' / 'overview.png'}")


if __name__ == "__main__":
    main()
