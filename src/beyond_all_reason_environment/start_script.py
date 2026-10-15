"""Generate the engine start script ("script.txt") that describes a match."""

from dataclasses import dataclass, field
from typing import Literal

from .installation import DEFAULT_MAP, GAME_VERSION

FRAMES_PER_SECOND = 30
MAXIMUM_ENGINE_SPEED = 1000.0

Side = Literal["Armada", "Cortex"]


@dataclass
class ArtificialIntelligencePlayer:
    """A built-in artificial intelligence, for example BARb (``short_name="BARb"``).

    ``short_name`` is the folder name in game_data/engine/<version>/AI/Skirmish.
    """

    short_name: str = "BARb"
    version: str = "stable"
    side: Side = "Armada"


@dataclass
class MatchConfiguration:
    players: list[ArtificialIntelligencePlayer] = field(
        default_factory=lambda: [
            ArtificialIntelligencePlayer(side="Armada"),
            ArtificialIntelligencePlayer(side="Cortex"),
        ]
    )
    map_name: str = DEFAULT_MAP
    game_version: str = GAME_VERSION
    seed: int = 1
    # Multiple of real time. Without graphics, a high value means "as fast as the processor allows".
    speed: float = 1000.0
    # Hard stop after this much game time, in case nobody wins.
    maximum_game_minutes: float = 60.0
    # Save replay.sdfz in the run folder, to watch the match later with scripts/watch_replay.py.
    record_replay: bool = False
    # Keep everything the engine writes in the run folder (cache, interface settings...), for debugging.
    # Otherwise only script.txt, match_result.json, engine_output.log, replay.sdfz and match.png remain.
    keep_engine_files: bool = False
    extra_mod_options: dict[str, str] = field(default_factory=dict)


def build_start_script(configuration: MatchConfiguration) -> str:
    """One spectator hosts the match; each artificial intelligence gets its own team and ally team."""
    maximum_frames = int(configuration.maximum_game_minutes * 60 * FRAMES_PER_SECOND)
    # debugcommands is read by the game's cmd_dev_helpers gadget: "frame:command|frame:command".
    # MaxSpeed only lifts the engine limit (20 by default); setspeed actually applies the speed.
    mod_options = {
        "MaxSpeed": f"{MAXIMUM_ENGINE_SPEED:g}",
        "debugcommands": f"1:setspeed {configuration.speed:g}|{maximum_frames}:quitforce",
        **configuration.extra_mod_options,
    }

    lines = [
        "[GAME]",
        "{",
        f"\tGameType={configuration.game_version};",
        f"\tMapName={configuration.map_name};",
        "\tIsHost=1;",
        "\tHostIP=127.0.0.1;",
        "\tHostPort=0;",
        "\tMyPlayerName=observer;",
        "\tStartPosType=1;",  # random start positions, deterministic for a given seed
        "\tGameStartDelay=0;",
        f"\tRecordDemo={int(configuration.record_replay)};",
        f"\tFixedRNGSeed={configuration.seed};",
        "\tNumPlayers=1;",
        f"\tNumTeams={len(configuration.players)};",
        f"\tNumAllyTeams={len(configuration.players)};",
        "\t[PLAYER0] { Name=observer; Spectator=1; Team=0; }",
    ]
    for index, player in enumerate(configuration.players):
        lines += [
            f"\t[AI{index}] {{ Name={player.short_name}_{index}; ShortName={player.short_name}; "
            f"Version={player.version}; Team={index}; Host=0; }}",
            f"\t[TEAM{index}] {{ TeamLeader=0; AllyTeam={index}; Side={player.side}; }}",
            f"\t[ALLYTEAM{index}] {{ NumAllies=0; }}",
        ]
    lines.append("\t[MODOPTIONS]")
    lines.append("\t{")
    lines += [f"\t\t{key}={value};" for key, value in mod_options.items()]
    lines.append("\t}")
    lines.append("}")
    return "\n".join(lines) + "\n"
