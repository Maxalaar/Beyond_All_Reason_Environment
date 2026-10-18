"""Write (or rewrite) the Parquet tables of finished runs from their saved results.

New runs write their tables by themselves; this is for runs run before the tables existed,
or after a change of tables.py.

Examples:
    python scripts/export_tables.py                                   # every run in runs/
    python scripts/export_tables.py runs/armada_versus_cortex/version_1
"""

import argparse
from pathlib import Path

from beyond_all_reason_environment.match import RUNS_DIRECTORY
from beyond_all_reason_environment.run import VERSION_PREFIX, load_run
from beyond_all_reason_environment.tables import write_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version_directories", type=Path, nargs="*", help="runs/<name>/version_<n> folders")
    arguments = parser.parse_args()

    directories = arguments.version_directories or sorted(
        path.parent for path in RUNS_DIRECTORY.glob(f"*/{VERSION_PREFIX}*/summary.json")
    )
    for directory in directories:
        paths = write_tables(directory, *load_run(directory))
        print(f"{directory}: {', '.join(path.name for path in paths.values())}")


if __name__ == "__main__":
    main()
