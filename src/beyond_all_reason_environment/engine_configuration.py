"""How the engine runs, shared by runs of games and the reinforcement learning environment: not part of the experiment.

Changing these does not change what the teams see or do, only how fast and how visibly it runs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Without graphics, "as fast as the processor allows"; with graphics, real time.
HEADLESS_SPEED = 1000.0
GRAPHICAL_SPEED = 1.0


# Widgets of the game hidden by each interface preset (GraphicsConfiguration.interface), by their names (the game's log
# lists them: "Loading widget from mod: <name> <file>").
# reduced: the panels and the screen texts (minimap, resources, players, menus, console, notifications), the range
# circles; the health bars and the names above the units stay.
REDUCED_INTERFACE_HIDDEN_WIDGETS = (
    "Top Bar",
    "Top Bar Buttons",
    "Minimap",
    "Picture-in-Picture Minimap",
    "AdvPlayersList",
    "AdvPlayersList Game Info",
    "AdvPlayersList Music Player New",
    "Ecostats",
    "Chat",
    "Messages",
    "Death Messages",
    "Info",
    "Order menu",
    "Grid menu",
    "Unit Stats",
    "Tooltip",
    "Player-TV",
    "Game info",
    "GameTypeInfo",
    "Screen Mode Info",
    "Replay buttons",
    "Pause Screen",
    "Keybind/Mouse Info",
    "Changelog Info",
    "Vote interface",
    "TeamStats",
    "Awards",
    "Idle Builders",
    "Unit Groups",
    "Rejoin progress",
    "Converter Usage",
    "Commander Hurt Vignette",
    "Attack Range GL4",
    "Defense Range GL4",
)
# cinematic: everything drawn over the game too (health bars, names, icons, selections, orders, ranges): only the
# rendering of the game (terrain, units, shots, explosions, their lights and scars) stays.
CINEMATIC_INTERFACE_HIDDEN_WIDGETS = REDUCED_INTERFACE_HIDDEN_WIDGETS + (
    "Health Bars GL4",
    "Commander Name Tags",
    "Rank Icons GL4",
    "Selected Units GL4",
    "Selectionbox",
    "Show Orders",
    "Commands FX",
    "Self-Destruct Icons",
    "Unit Energy Icons",
    "Unit Fire State Icons",
    "Unit Wait Icons",
    "Unit Repeat Icons",
    "Unit Idle Builder Icons",
    "Unit Group Number",
    "BuildETA",
    "AllyCursors",
    "Mouse FX",
    "Point Tracker",
    "Mapmarks FX",
)
INTERFACE_HIDDEN_WIDGETS = {
    "complete": (),
    "reduced": REDUCED_INTERFACE_HIDDEN_WIDGETS,
    "cinematic": CINEMATIC_INTERFACE_HIDDEN_WIDGETS,
}
Interface = Literal["complete", "reduced", "cinematic"]


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
    # What the window shows over the game: "complete" (the game's whole interface), "reduced" (no panels nor screen
    # texts nor range circles, the health bars and the names above the units stay) or "cinematic" (only the rendering
    # of the game); see INTERFACE_HIDDEN_WIDGETS. Except "complete", the mouse cursor is the system's (the engine's
    # HardwareCursor): on the screen, not in the window's image, so neither in the videos.
    interface: Interface = "complete"
    # Widgets of the game hidden on top of the preset's, by their names.
    hidden_widgets: tuple[str, ...] = ()
    # Record the window into this video file (.mp4, H.264) from the first game frame to the end of the engine
    # (video_recording.py: from the engine itself, whatever screen the window is on). None: no video.
    video_path: Path | str | None = None
    # Images per second of the video.
    video_frame_rate: int = 30
    # Game seconds per second of video: 2 plays the game twice as fast (chosen with the user, 2026-10-25), one image
    # every 30 * video_speed / video_frame_rate game frames.
    video_speed: float = 2.0

    @property
    def all_hidden_widgets(self) -> tuple[str, ...]:
        """The widgets of the game the window hides: the preset's and hidden_widgets."""
        return INTERFACE_HIDDEN_WIDGETS[self.interface] + tuple(self.hidden_widgets)

    def __post_init__(self) -> None:
        if self.camera_rotation_speed <= 0:
            raise ValueError(f"camera_rotation_speed must be positive, not {self.camera_rotation_speed}")
        if self.camera_keys is not None and len(self.camera_keys) != 4:
            raise ValueError(f"camera_keys must be four keys (forward, left, back, right), not {self.camera_keys!r}")
        if self.interface not in INTERFACE_HIDDEN_WIDGETS:
            raise ValueError(f"interface must be one of {tuple(INTERFACE_HIDDEN_WIDGETS)}, not {self.interface!r}")
        if self.video_frame_rate <= 0:
            raise ValueError(f"video_frame_rate must be positive, not {self.video_frame_rate}")
        if self.video_speed <= 0:
            raise ValueError(f"video_speed must be positive, not {self.video_speed}")


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
