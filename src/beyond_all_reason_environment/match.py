"""Run one match, with or without graphics, and return its result."""

import functools
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .installation import GAME_DATA_DIRECTORY, GRAPHICAL_EXECUTABLE, HEADLESS_EXECUTABLE, REPOSITORY_ROOT
from .start_script import FRAMES_PER_SECOND, MatchConfiguration, build_start_script

RUNS_DIRECTORY = REPOSITORY_ROOT / "runs"
LUA_WIDGETS_DIRECTORY = Path(__file__).parent / "lua" / "widgets"
# Written by lua/widgets/match_result_reporter.lua.
RESULT_RELATIVE_PATH = Path("LuaUI") / "Config" / "match_result.json"


@dataclass
class MatchResult:
    # "victory", "draw", "timeout" (maximum game time reached) or "crash" (no result written).
    outcome: str
    # Index of the winning player in MatchConfiguration.players, only for a victory.
    winner: int | None
    game_frames: int | None
    wall_clock_seconds: float
    run_directory: Path
    # One dictionary per player, in the order of MatchConfiguration.players: final engine statistics
    # (metalProduced, damageDealt, unitsKilled...) and units alive at the end (unitsAlive).
    team_statistics: list[dict] = field(default_factory=list)

    @property
    def game_minutes(self) -> float | None:
        return None if self.game_frames is None else self.game_frames / FRAMES_PER_SECOND / 60

    @property
    def speed_factor(self) -> float | None:
        return None if self.game_frames is None else self.game_frames / FRAMES_PER_SECOND / self.wall_clock_seconds


# Window settings for graphical matches, written to the run folder's engine configuration.
GRAPHICAL_SETTINGS = {
    "Fullscreen": "0",
    "XResolutionWindowed": "1600",
    "YResolutionWindowed": "900",
}

# On laptops with hybrid graphics, render on the NVIDIA card instead of the integrated one (about 2x faster).
# Offloading goes through GLX, hence X11 (XWayland under Wayland).
NVIDIA_OFFLOAD_ENVIRONMENT = {
    "__NV_PRIME_RENDER_OFFLOAD": "1",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "__VK_LAYER_NV_optimus": "NVIDIA_only",
    "SDL_VIDEODRIVER": "x11",
}

# OpenGL driver of NixOS (NVIDIA or Mesa).
SYSTEM_OPENGL_LIBRARY_DIRECTORY = "/run/opengl-driver/lib"


@functools.cache
def graphics_library_path() -> str:
    """SDL2, X11 and OpenAL for the graphical engine, which NixOS does not expose globally.

    Inside `nix develop` they come from an environment variable; anywhere else (conda, plain Python)
    they are built once by Nix from flake.nix and then reused from the Nix store.
    """
    path = os.environ.get("BEYOND_ALL_REASON_GRAPHICS_LIBRARY_PATH")
    if path is None:
        output = subprocess.run(
            ["nix", "build", "--no-link", "--print-out-paths", f"{REPOSITORY_ROOT}#graphics-libraries"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        path = f"{output}/lib"
    return f"{path}:{SYSTEM_OPENGL_LIBRARY_DIRECTORY}"


def run_match(
    configuration: MatchConfiguration,
    run_directory: Path | None = None,
    graphical: bool = False,
) -> MatchResult:
    """Run a match to the end. With ``graphical=True`` a window shows it, at ``configuration.speed``."""
    executable = GRAPHICAL_EXECUTABLE if graphical else HEADLESS_EXECUTABLE
    if not executable.exists():
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
    if graphical:
        (run_directory / "springsettings.cfg").write_text(
            "".join(f"{key} = {value}\n" for key, value in GRAPHICAL_SETTINGS.items())
        )
        environment["LD_LIBRARY_PATH"] = ":".join(
            filter(None, [graphics_library_path(), os.environ.get("LD_LIBRARY_PATH")])
        )
        if shutil.which("nvidia-smi") is not None:
            environment.update(NVIDIA_OFFLOAD_ENVIRONMENT)
    start_time = time.monotonic()
    with open(run_directory / "engine_output.log", "w") as log_file:
        subprocess.run(
            [str(executable), "--isolation", "--write-dir", str(run_directory), str(script_path)],
            cwd=run_directory,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_clock_seconds = time.monotonic() - start_time

    result_path = run_directory / RESULT_RELATIVE_PATH
    if not result_path.exists():
        return MatchResult("crash", None, None, wall_clock_seconds, run_directory)
    result = json.loads(result_path.read_text())

    # Each player has its own team and ally team, both numbered like the players.
    statistics_by_team = {int(team): values for team, values in result["teams"].items()}
    team_statistics = [statistics_by_team.get(index, {}) for index in range(len(configuration.players))]
    winners = result["winning_ally_teams"]
    if not result["game_over"]:
        outcome, winner = "timeout", None
    elif len(winners) == 1:
        outcome, winner = "victory", winners[0]
    else:
        outcome, winner = "draw", None
    return MatchResult(outcome, winner, result["frame"], wall_clock_seconds, run_directory, team_statistics)
