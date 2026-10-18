"""Controllers: who plays a team of a scenario, and how. Give one to each TeamConfiguration.

    DoNothing()                       the units keep their default states and receive no order
    Passive()                         the units do not move and do not shoot
    HoldAndFire()                     the units do not move and shoot what comes in range
    AttackNearest(interval_seconds)   every interval, each unit attacks the nearest enemy
    BuiltInArtificialIntelligence()   an artificial intelligence of the game: BARb by default
    Agent()                           the agents of the combat environment, played from Python

Each controller says which artificial intelligence the engine runs for the team (``player``), the fire and move
states of its placed units (``unit_states``) and the script the game extension's gadget runs for it, if any
(``gadget_script``). The scripted controllers run in the gadget, in Lua: they need no Python and also play in
matches. Adding a controller means a subclass here, and for a scripted one a function in the gadget
(lua/game_extension/luarules/gadgets/reinforcement_learning_scenario.lua).
"""

from dataclasses import dataclass, field

from .start_script import FRAMES_PER_SECOND, ArtificialIntelligencePlayer, Side

# Fire states: 0 hold fire, 1 return fire, 2 fire at will. Move states: 0 hold position, 1 manoeuvre, 2 roam.
FIRE_AT_WILL, HOLD_FIRE = 2, 0
HOLD_POSITION, MANOEUVRE = 0, 1
# Keeps the unit's own state (the game's default, or what its artificial intelligence sets).
DEFAULT_STATE = -1

NULL_ARTIFICIAL_INTELLIGENCE = ("NullAI", "0.1")


@dataclass(frozen=True)
class Controller:
    """Base class: a team whose engine artificial intelligence does nothing (NullAI)."""

    @property
    def label(self) -> str:
        """Name of the team's player in charts and tables."""
        return type(self).__name__

    def player(self, side: Side | list[Side]) -> ArtificialIntelligencePlayer:
        """The artificial intelligence the engine runs for the team."""
        short_name, version = NULL_ARTIFICIAL_INTELLIGENCE
        return ArtificialIntelligencePlayer(short_name=short_name, version=version, side=side, label=self.label)

    def unit_states(self) -> tuple[int, int]:
        """Fire and move states given to the team's placed units (DEFAULT_STATE keeps the unit's own)."""
        return (DEFAULT_STATE, DEFAULT_STATE)

    def gadget_script(self) -> str | None:
        """ "<name>,<parameter>,..." for a behaviour the gadget runs, or None."""
        return None


@dataclass(frozen=True)
class DoNothing(Controller):
    """The units keep their default states and receive no order: they only shoot back."""


@dataclass(frozen=True)
class Passive(Controller):
    """The units do not move and do not shoot."""

    def unit_states(self) -> tuple[int, int]:
        return (HOLD_FIRE, HOLD_POSITION)


@dataclass(frozen=True)
class HoldAndFire(Controller):
    """The units do not move and shoot what comes in range."""

    def unit_states(self) -> tuple[int, int]:
        return (FIRE_AT_WILL, HOLD_POSITION)


@dataclass(frozen=True)
class AttackNearest(Controller):
    """Every ``interval_seconds`` of game time, each unit attacks the nearest enemy it sees, or moves towards the
    nearest enemy while fighting (attack-move) when it sees none."""

    interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.interval_frames < 1:
            raise ValueError("interval_seconds must be at least one frame (1/30 s)")

    @property
    def interval_frames(self) -> int:
        return round(self.interval_seconds * FRAMES_PER_SECOND)

    def unit_states(self) -> tuple[int, int]:
        return (FIRE_AT_WILL, MANOEUVRE)

    def gadget_script(self) -> str | None:
        return f"attack_nearest,{self.interval_frames}"


@dataclass(frozen=True)
class BuiltInArtificialIntelligence(Controller):
    """An artificial intelligence of the game, BARb by default; ``short_name`` is its folder name in
    game_data/engine/<version>/AI/Skirmish. It sets its units' states itself."""

    short_name: str = "BARb"
    version: str = "stable"
    # Written as an [OPTIONS] block in the start script, readable by the artificial intelligence.
    options: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.short_name

    def player(self, side: Side | list[Side]) -> ArtificialIntelligencePlayer:
        return ArtificialIntelligencePlayer(
            short_name=self.short_name, version=self.version, side=side, options=dict(self.options)
        )


@dataclass(frozen=True)
class Agent(Controller):
    """The agents of the combat environment: the BridgeAI forwards their observations and actions to Python.

    Only the combat environment can run it, on any number of teams (several for self-play); their units' states come
    from ActionConfiguration.
    """

    def player(self, side: Side | list[Side]) -> ArtificialIntelligencePlayer:
        return ArtificialIntelligencePlayer(short_name="BridgeAI", version="0.1", side=side, label=self.label)
