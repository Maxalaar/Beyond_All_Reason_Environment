"""How the engine runs, shared by runs of games and the reinforcement learning environment: not part of the experiment.

Changing these does not change what the teams see or do, only how fast and how visibly it runs.
"""

from dataclasses import dataclass, field

# Without graphics, "as fast as the processor allows"; with graphics, real time.
HEADLESS_SPEED = 1000.0
GRAPHICAL_SPEED = 1.0


@dataclass
class GraphicsConfiguration:
    """The window to watch the game (the graphical engine binary): opened only when ``show`` is True, so the other
    settings stay written in the configuration when the window is off."""

    show: bool = False
    sound: bool = False
    # A borderless window covering the screen; False: a 1600x900 window.
    fullscreen: bool = True
    # Four keys moving the camera forward, left, back and right (Shift: faster), for example "zqsd" on an AZERTY
    # keyboard ("wasd" on QWERTY); the arrows keep working. None leaves the game's keys alone.
    camera_keys: str | None = "zqsd"
    # Hold the right mouse button and move the mouse to rotate the camera around the point it looks at, as in 3D
    # editors (Unity, Unreal); for spectators only, who give no orders. The game's own rotation (Alt and the middle
    # button) keeps working; the wheel zooms.
    right_button_rotation: bool = True
    # Radians the camera turns per pixel the mouse moves, with right_button_rotation.
    camera_rotation_speed: float = 0.003
    # By default the view follows the mouse (right turns the view right, up tilts it up); these invert each axis.
    invert_horizontal_rotation: bool = False
    invert_vertical_rotation: bool = False
    # Hold the rotation for a moment at north, east, south and west (about 18 degrees around each), to line the view
    # up with the map: the engine's CamSpringLockCardinalDirections, on in the game, off here for a free rotation.
    lock_cardinal_directions: bool = False
    # Close the window this many seconds after the end of the game; None leaves it open.
    close_delay_seconds: float | None = 0.0

    def __post_init__(self) -> None:
        if self.camera_rotation_speed <= 0:
            raise ValueError(f"camera_rotation_speed must be positive, not {self.camera_rotation_speed}")
        if self.camera_keys is not None and len(self.camera_keys) != 4:
            raise ValueError(f"camera_keys must be four keys (forward, left, back, right), not {self.camera_keys!r}")


@dataclass
class EngineConfiguration:
    """Speed, window, and what the engine leaves in the run folder."""

    # Multiple of real time; None: HEADLESS_SPEED without a window, GRAPHICAL_SPEED with (Alt+= and Alt+- change it in
    # game).
    speed: float | None = None
    # The window; graphics.show False: none (the headless engine binary).
    graphics: GraphicsConfiguration = field(default_factory=GraphicsConfiguration)
    # Keep runs/reinforcement_learning_environments/<run>/ (start script, engine log) after the environment closes,
    # for debugging.
    keep_run_directory: bool = False

    @property
    def effective_speed(self) -> float:
        if self.speed is not None:
            return self.speed
        return GRAPHICAL_SPEED if self.graphics.show else HEADLESS_SPEED
