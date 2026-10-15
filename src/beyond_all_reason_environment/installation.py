"""Locate and download the pinned engine, game and maps.

Everything lives in ``game_data/`` at the repository root. That folder is shared read-only
by every match; each match writes into its own run folder instead.
"""

import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

# Pinned versions: change them together and re-run scripts/download_game_data.py.
# The engine version is the one the official launcher uses (BYAR-Chobby dist_cfg/config.json).
ENGINE_VERSION = "2026.07.04"
GAME_VERSION = "Beyond All Reason test-31630-d0ec39b"
DEFAULT_MAP = "Red Comet Remake 1.8"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GAME_DATA_DIRECTORY = REPOSITORY_ROOT / "game_data"
ENGINE_DIRECTORY = GAME_DATA_DIRECTORY / "engine" / f"recoil_{ENGINE_VERSION}"
HEADLESS_EXECUTABLE = ENGINE_DIRECTORY / "spring-headless"
GRAPHICAL_EXECUTABLE = ENGINE_DIRECTORY / "spring"

ENGINE_URL = (
    "https://github.com/beyond-all-reason/RecoilEngine/releases/download/"
    f"{ENGINE_VERSION}/recoil_{ENGINE_VERSION}_amd64-linux.7z"
)

# Download servers of Beyond All Reason, as configured in its launcher.
PR_DOWNLOADER_ENVIRONMENT = {
    "PRD_RAPID_USE_STREAMER": "false",
    "PRD_RAPID_REPO_MASTER": "https://repos-cdn.beyondallreason.dev/repos.gz",
    "PRD_HTTP_SEARCH_URL": "https://files-cdn.beyondallreason.dev/find",
}


def download_engine() -> None:
    if HEADLESS_EXECUTABLE.exists():
        print(f"Engine {ENGINE_VERSION} already present.")
        return
    # Outside `nix develop`, borrow 7z from Nix for this one command.
    seven_zip = ["7z"] if shutil.which("7z") else ["nix", "shell", "nixpkgs#p7zip", "--command", "7z"]
    print(f"Downloading engine {ENGINE_VERSION}...")
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive = Path(temporary_directory) / "engine.7z"
        urllib.request.urlretrieve(ENGINE_URL, archive)
        ENGINE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [*seven_zip, "x", "-y", f"-o{ENGINE_DIRECTORY}", str(archive)],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def _pr_downloader(*arguments: str) -> None:
    subprocess.run(
        [str(ENGINE_DIRECTORY / "pr-downloader"), "--filesystem-writepath", str(GAME_DATA_DIRECTORY), *arguments],
        check=True,
        env={**os.environ, **PR_DOWNLOADER_ENVIRONMENT},
    )


def download_game() -> None:
    print(f"Downloading game {GAME_VERSION} (about 3 GB the first time)...")
    _pr_downloader("--download-game", GAME_VERSION)


def download_map(map_name: str = DEFAULT_MAP) -> None:
    print(f"Downloading map {map_name}...")
    _pr_downloader("--download-map", map_name)


def download_all(map_names: list[str] | None = None) -> None:
    download_engine()
    download_game()
    for map_name in map_names or [DEFAULT_MAP]:
        download_map(map_name)
