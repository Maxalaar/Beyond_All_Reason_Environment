"""Make this project reachable from claude.ai/code and the Claude phone app, and keep it reachable.

Starts `claude remote-control` in the project directory, keeps the computer awake while it runs
(no idle sleep, no suspend, closing the lid does nothing) and restarts it if it stops unexpectedly.
Stop everything with Ctrl-C; the computer then sleeps normally again.

The first time, Claude asks "Enable Remote Control? (y/n)": answer y in this terminal.

Examples:
    python scripts/connect_remote_control.py
    python scripts/connect_remote_control.py --permission-mode acceptEdits
    python scripts/connect_remote_control.py --spawn worktree   # one git worktree per session
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
SLEEP_TO_INHIBIT = "idle:sleep:handle-lid-switch"
FIRST_RESTART_DELAY_SECONDS = 5.0
MAXIMUM_RESTART_DELAY_SECONDS = 300.0
# A run that lasted this long was healthy: the restart delay goes back to its first value.
HEALTHY_RUN_SECONDS = 600.0


def remote_control_command(arguments: argparse.Namespace) -> list[str]:
    command = ["claude", "remote-control", "--name", arguments.name, "--spawn", arguments.spawn]
    if arguments.permission_mode is not None:
        command += ["--permission-mode", arguments.permission_mode]
    if arguments.allow_sleep:
        return command
    return [
        "systemd-inhibit",
        f"--what={SLEEP_TO_INHIBIT}",
        "--who=Claude Remote Control",
        "--why=Keeps this project reachable from claude.ai/code",
        "--mode=block",
        *command,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="Beyond All Reason RL", help="session name shown in claude.ai/code")
    parser.add_argument(
        "--spawn",
        choices=["same-dir", "worktree", "session"],
        default="same-dir",
        help="where sessions opened from the phone work (default: this directory)",
    )
    parser.add_argument(
        "--permission-mode",
        choices=["default", "acceptEdits", "auto", "plan", "dontAsk", "bypassPermissions"],
        default=None,
        help="permission mode of the sessions opened from the phone (default: Claude's own)",
    )
    parser.add_argument("--allow-sleep", action="store_true", help="do not keep the computer awake")
    parser.add_argument("--no-restart", action="store_true", help="stop instead of restarting after a failure")
    arguments = parser.parse_args()

    required = ["claude"] if arguments.allow_sleep else ["claude", "systemd-inhibit"]
    for program in required:
        if shutil.which(program) is None:
            sys.exit(f"{program} not found in PATH")

    command = remote_control_command(arguments)
    restart_delay = FIRST_RESTART_DELAY_SECONDS
    while True:
        start = time.monotonic()
        try:
            return_code = subprocess.run(command, cwd=PROJECT_DIRECTORY).returncode
        except KeyboardInterrupt:
            print("\nRemote Control stopped, the computer may sleep again.")
            return
        if return_code == 0 or arguments.no_restart:
            print(f"Remote Control exited (code {return_code}).")
            return
        if time.monotonic() - start > HEALTHY_RUN_SECONDS:
            restart_delay = FIRST_RESTART_DELAY_SECONDS
        print(f"Remote Control stopped with code {return_code}, restarting in {restart_delay:.0f} s (Ctrl-C to quit).")
        try:
            time.sleep(restart_delay)
        except KeyboardInterrupt:
            print("\nRemote Control stopped, the computer may sleep again.")
            return
        restart_delay = min(2 * restart_delay, MAXIMUM_RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    main()
