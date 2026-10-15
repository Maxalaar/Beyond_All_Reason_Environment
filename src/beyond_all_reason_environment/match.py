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
# Written there by lua/widgets/match_result_reporter.lua, then moved to the top of the run folder.
ENGINE_RESULT_PATH = Path("LuaUI") / "Config" / "match_result.json"
RESULT_FILE_NAME = "match_result.json"
# The engine records replays into demos/<date>_<map>_<engine>.sdfz; moved to the top of the run folder.
REPLAY_FILE_NAME = "replay.sdfz"
# Everything else the engine writes in the run folder is deleted after the match, unless keep_engine_files.
KEPT_FILE_NAMES = {"script.txt", "engine_output.log", RESULT_FILE_NAME, REPLAY_FILE_NAME, "charts", "overview.png"}


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
    # One list per player: engine statistics every 16 seconds of game time, cumulative values,
    # with "time" in seconds. Used by plots.py.
    team_history: list[list[dict]] = field(default_factory=list, repr=False)
    replay_path: Path | None = None
    # One {"x": ..., "z": ...} per player (map coordinates, None if unknown), and the map size {"sizeX", "sizeZ"}.
    start_positions: list[dict | None] = field(default_factory=list)
    map_size: dict | None = None

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


def graphical_environment(write_directory: Path) -> dict[str, str]:
    """Environment and window settings for the graphical engine binary, writing into ``write_directory``."""
    (write_directory / "springsettings.cfg").write_text(
        "".join(f"{key} = {value}\n" for key, value in GRAPHICAL_SETTINGS.items())
    )
    environment = {
        **os.environ,
        "SPRING_DATADIR": str(GAME_DATA_DIRECTORY),
        "LD_LIBRARY_PATH": ":".join(filter(None, [graphics_library_path(), os.environ.get("LD_LIBRARY_PATH")])),
    }
    if shutil.which("nvidia-smi") is not None:
        environment.update(NVIDIA_OFFLOAD_ENVIRONMENT)
    return environment


def _tidy_run_directory(run_directory: Path, keep_engine_files: bool) -> None:
    """Move the result and the replay to the top of the run folder, then delete the engine's other files."""
    engine_result = run_directory / ENGINE_RESULT_PATH
    if engine_result.exists():
        shutil.move(engine_result, run_directory / RESULT_FILE_NAME)
    replays = sorted((run_directory / "demos").glob("*.sdfz"))
    if replays:
        shutil.move(replays[-1], run_directory / REPLAY_FILE_NAME)
    if keep_engine_files:
        return
    for path in run_directory.iterdir():
        if path.name in KEPT_FILE_NAMES:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def start_engine(configuration: MatchConfiguration, run_directory: Path, graphical: bool = False) -> subprocess.Popen:
    """Prepare ``run_directory`` (widgets, start script) and start the engine on it without waiting.

    The engine output goes to engine_output.log in the run folder.
    """
    executable = GRAPHICAL_EXECUTABLE if graphical else HEADLESS_EXECUTABLE
    if not executable.exists():
        raise FileNotFoundError("Engine missing: run scripts/download_game_data.py first.")
    run_directory.mkdir(parents=True, exist_ok=True)

    widgets_directory = run_directory / "LuaUI" / "Widgets"
    widgets_directory.mkdir(parents=True, exist_ok=True)
    for widget in LUA_WIDGETS_DIRECTORY.glob("*.lua"):
        shutil.copy(widget, widgets_directory)

    script_path = run_directory / "script.txt"
    script_path.write_text(build_start_script(configuration))

    # The run folder is the write directory (logs, results); the shared game_data folder is read-only.
    if graphical:
        environment = graphical_environment(run_directory)
    else:
        environment = {**os.environ, "SPRING_DATADIR": str(GAME_DATA_DIRECTORY)}
    with open(run_directory / "engine_output.log", "w") as log_file:
        # The child process keeps its own handle on the log file.
        return subprocess.Popen(
            [str(executable), "--isolation", "--write-dir", str(run_directory), str(script_path)],
            cwd=run_directory,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def run_match(
    configuration: MatchConfiguration,
    run_directory: Path | None = None,
    graphical: bool = False,
) -> MatchResult:
    """Run a match to the end. With ``graphical=True`` a window shows it, at ``configuration.speed``."""
    if run_directory is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_directory = RUNS_DIRECTORY / f"{timestamp}_seed{configuration.seed}"
    start_time = time.monotonic()
    start_engine(configuration, run_directory, graphical).wait()
    wall_clock_seconds = time.monotonic() - start_time
    _tidy_run_directory(run_directory, configuration.keep_engine_files)
    return read_match_result(run_directory, len(configuration.players), wall_clock_seconds)


def read_match_result(run_directory: Path, number_of_players: int, wall_clock_seconds: float) -> MatchResult:
    """Build the result of a finished match from its match_result.json ("crash" when it is missing)."""
    replay_path = run_directory / REPLAY_FILE_NAME
    replay = replay_path if replay_path.exists() else None
    result_path = run_directory / RESULT_FILE_NAME
    if not result_path.exists():
        return MatchResult("crash", None, None, wall_clock_seconds, run_directory, replay_path=replay)
    result = json.loads(result_path.read_text())

    # Each player has its own team and ally team, both numbered like the players.
    teams = {int(team): values for team, values in result["teams"].items()}
    team_history = [teams[index]["history"] if index in teams else [] for index in range(number_of_players)]
    team_statistics = [
        {**(history[-1] if history else {}), "unitsAlive": teams[index]["unitsAlive"]} if index in teams else {}
        for index, history in enumerate(team_history)
    ]
    winners = result["winning_ally_teams"]
    if not result["game_over"]:
        outcome, winner = "timeout", None
    elif len(winners) == 1:
        outcome, winner = "victory", winners[0]
    else:
        outcome, winner = "draw", None
    map_information = result.get("map")  # absent from matches played before it was recorded
    return MatchResult(
        outcome,
        winner,
        result["frame"],
        wall_clock_seconds,
        run_directory,
        team_statistics,
        team_history,
        replay,
        start_positions=[teams.get(index, {}).get("startPosition") for index in range(number_of_players)],
        map_size={key: map_information[key] for key in ("sizeX", "sizeZ")} if map_information else None,
    )
