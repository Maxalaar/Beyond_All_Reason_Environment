"""A small game archive that depends on Beyond All Reason and adds the Lua files of the reinforcement learning
environments (lua/game_extension/).

Beyond All Reason removes LuaSocket from user widgets, so the bridge to Python must be a game file. The engine
loads an archive's dependencies first, so the game is Beyond All Reason plus these files.
"""

import shutil
from pathlib import Path

from .installation import GAME_VERSION

EXTENSION_NAME = "Beyond All Reason Reinforcement Learning"
EXTENSION_VERSION = "1"
EXTENSION_SOURCE_DIRECTORY = Path(__file__).parent / "lua" / "game_extension"
ARCHIVE_FOLDER_NAME = "beyond_all_reason_reinforcement_learning.sdd"


def extension_game_name() -> str:
    """The value of GameType in the start script: the archive name, then its version."""
    return f"{EXTENSION_NAME} {EXTENSION_VERSION}"


def write_game_extension(write_directory: Path) -> str:
    """Write the archive into <write_directory>/games/, where the engine finds it, and return its game name."""
    archive = write_directory / "games" / ARCHIVE_FOLDER_NAME
    if archive.exists():
        shutil.rmtree(archive)
    shutil.copytree(EXTENSION_SOURCE_DIRECTORY, archive)
    (archive / "modinfo.lua").write_text(
        "return {\n"
        f'\tname = "{EXTENSION_NAME}",\n'
        '\tshortName = "BeyondAllReasonReinforcementLearning",\n'
        f'\tversion = "{EXTENSION_VERSION}",\n'
        f'\tgame = "{EXTENSION_NAME}",\n'
        '\tshortGame = "BeyondAllReasonReinforcementLearning",\n'
        "\tmodtype = 1,\n"
        f'\tdepend = {{ "{GAME_VERSION}" }},\n'
        "}\n"
    )
    return extension_game_name()
