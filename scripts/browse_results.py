"""Open the web interface to browse the results of all runs, in the browser.

Example:
    python scripts/browse_results.py            # then open http://localhost:8501 if no tab opens
    python scripts/browse_results.py --port 8600

Runs played before the Parquet tables existed need: python scripts/export_tables.py
"""

import argparse
import subprocess
import sys

from beyond_all_reason_environment import results_browser_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8501)
    arguments = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(results_browser_path()),
            "--server.port",
            str(arguments.port),
            "--browser.gatherUsageStats",
            "false",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
