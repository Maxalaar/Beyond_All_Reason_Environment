"""How the engine runs, shared by runs of games and the reinforcement learning environment: not part of the experiment.

Changing these does not change what the teams see or do, only how fast and how visibly it runs.
"""

from dataclasses import dataclass, field

# Without graphics, "as fast as the processor allows"; with graphics, real time.
HEADLESS_SPEED = 1000.0
GRAPHICAL_SPEED = 1.0


@dataclass
class GraphicsConfiguration:
    """A window to watch the game (the graphical engine binary)."""

    sound: bool = False
    # A borderless window covering the screen; False: a 1600x900 window.
    fullscreen: bool = True
    # Four keys moving the camera forward, left, back and right, for example "zqsd" on an AZERTY keyboard ("wasd" on
    # QWERTY); with Alt held, the same keys rotate and tilt the camera. None keeps the game's keys (arrows; Alt and
    # the middle mouse button rotate).
    camera_keys: str | None = "zqsd"
    # Close the window this many seconds after the end of the game; None leaves it open.
    close_delay_seconds: float | None = 5.0

    def __post_init__(self) -> None:
        if self.camera_keys is not None and len(self.camera_keys) != 4:
            raise ValueError(f"camera_keys must be four keys (forward, left, back, right), not {self.camera_keys!r}")


@dataclass
class EngineConfiguration:
    """Speed, window, and what the engine leaves in the run folder."""

    # Multiple of real time; None: HEADLESS_SPEED without graphics, GRAPHICAL_SPEED with (+ and - change it in game).
    speed: float | None = None
    # None: no window (the headless engine binary).
    graphics: GraphicsConfiguration | None = field(default=None)
    # Keep runs/reinforcement_learning_environments/<run>/ (start script, engine log) after the environment closes,
    # for debugging.
    keep_run_directory: bool = False

    @property
    def effective_speed(self) -> float:
        if self.speed is not None:
            return self.speed
        return HEADLESS_SPEED if self.graphics is None else GRAPHICAL_SPEED
