"""Write (or rewrite) the Parquet tables of finished batches from their saved results.

New batches write their tables by themselves; this is for batches run before the tables existed,
or after a change of tables.py.

Examples:
    python scripts/export_tables.py                                   # every batch in runs/
    python scripts/export_tables.py runs/armada_versus_cortex/version_1
"""

import argparse
from pathlib import Path

from beyond_all_reason_environment.match import RUNS_DIRECTORY
from beyond_all_reason_environment.run import VERSION_PREFIX, load_batch
from beyond_all_reason_environment.tables import write_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch_directories", type=Path, nargs="*", help="runs/<name>/version_<n> folders")
    arguments = parser.parse_args()

    directories = arguments.batch_directories or sorted(
        path.parent for path in RUNS_DIRECTORY.glob(f"*/{VERSION_PREFIX}*/summary.json")
    )
    for directory in directories:
        paths = write_tables(directory, *load_batch(directory))
        print(f"{directory}: {', '.join(path.name for path in paths.values())}")


if __name__ == "__main__":
    main()
