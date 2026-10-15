"""Run one headless match and return its result."""

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .installation import GAME_DATA_DIRECTORY, HEADLESS_EXECUTABLE, REPOSITORY_ROOT
from .start_script import FRAMES_PER_SECOND, MatchConfiguration, build_start_script

RUNS_DIRECTORY = REPOSITORY_ROOT / "runs"
LUA_WIDGETS_DIRECTORY = Path(__file__).parent / "lua" / "widgets"


@dataclass
class MatchResult:
    # Index of the winning player in MatchConfiguration.players, or None for a draw or timeout.
    winner: int | None
    game_frames: int | None
    wall_clock_seconds: float
    run_directory: Path

    @property
    def game_minutes(self) -> float | None:
        return None if self.game_frames is None else self.game_frames / FRAMES_PER_SECOND / 60

    @property
    def speed_factor(self) -> float | None:
        return None if self.game_frames is None else self.game_frames / FRAMES_PER_SECOND / self.wall_clock_seconds


def run_match(configuration: MatchConfiguration, run_directory: Path | None = None) -> MatchResult:
    if not HEADLESS_EXECUTABLE.exists():
        raise FileNotFoundError("Engine missing: run scripts/download_game_data.py first.")

    if run_directory is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_directory = RUNS_DIRECTORY / f"{timestamp}_seed{configuration.seed}"
    run_directory.mkdir(parents=True, exist_ok=True)

    widgets_directory = run_directory / "LuaUI" / "Widgets"
    widgets_directory.mkdir(parents=True, exist_ok=True)
    for widget in LUA_WIDGETS_DIRECTORY.glob("*.lua"):
        shutil.copy(widget, widgets_directory)

    script_path = run_directory / "script.txt"
    script_path.write_text(build_start_script(configuration))

    # The run folder is the write directory (logs, results); the shared game_data folder is read-only.
    environment = {**os.environ, "SPRING_DATADIR": str(GAME_DATA_DIRECTORY)}
    start_time = time.monotonic()
    with open(run_directory / "engine_output.log", "w") as log_file:
        subprocess.run(
            [str(HEADLESS_EXECUTABLE), "--isolation", "--write-dir", str(run_directory), str(script_path)],
            cwd=run_directory,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_clock_seconds = time.monotonic() - start_time

    result_path = run_directory / "LuaUI" / "Config" / "match_result.json"
    if not result_path.exists():
        return MatchResult(None, None, wall_clock_seconds, run_directory)
    result = json.loads(result_path.read_text())
    winners = result["winning_ally_teams"]
    # Each player has its own ally team, numbered like the players.
    winner = winners[0] if len(winners) == 1 else None
    return MatchResult(winner, result["frame"], wall_clock_seconds, run_directory)
