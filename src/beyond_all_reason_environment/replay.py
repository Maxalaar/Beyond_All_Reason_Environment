"""Watch a recorded match (replay.sdfz) in a window."""

import subprocess
import tempfile
from pathlib import Path

from .installation import GRAPHICAL_EXECUTABLE
from .match import graphical_environment

REPLAY_SPEED_WIDGET = Path(__file__).parent / "lua" / "replay_widgets" / "replay_speed.lua"


def watch_replay(replay_path: Path, speed: float = 1.0) -> None:
    """Open a window that replays the match from the start; + and - change the speed, as in a live match.

    The replay needs the same engine, game and map versions as the recorded match (pinned in installation.py).
    """
    replay_path = Path(replay_path).resolve()
    if not replay_path.exists():
        raise FileNotFoundError(replay_path)
    # Cache and interface settings of the viewer are thrown away afterwards.
    with tempfile.TemporaryDirectory(prefix="beyond_all_reason_replay_") as directory:
        write_directory = Path(directory)
        widgets_directory = write_directory / "LuaUI" / "Widgets"
        widgets_directory.mkdir(parents=True)
        widget = REPLAY_SPEED_WIDGET.read_text().replace("local SPEED = 1\n", f"local SPEED = {speed:g}\n")
        (widgets_directory / REPLAY_SPEED_WIDGET.name).write_text(widget)
        subprocess.run(
            [str(GRAPHICAL_EXECUTABLE), "--isolation", "--write-dir", str(write_directory), str(replay_path)],
            cwd=write_directory,
            env=graphical_environment(write_directory),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
