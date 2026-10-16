"""Base configuration shared by all environment types."""

from dataclasses import dataclass

from .installation import DEFAULT_MAP

FRAMES_PER_SECOND = 30


@dataclass
class EnvironmentConfiguration:
    map_name: str = DEFAULT_MAP
    maximum_episode_minutes: float = 1.0
    decision_interval_frames: int = 32
    # Observation: the nearest entities each agent sees.
    maximum_entities_seen: int = 16
    # Both teams see the whole map; required so the engine accepts attack orders on distant units.
    full_vision: bool = True
    # Terrain observation: the heightmap resampled to this fixed (width, height) grid.
    # Channels: normalised height, slope in [0,1), water mask. The real map dimensions are
    # already in the self token (map_size_x_thousands, map_size_z_thousands, map_aspect_ratio).
    terrain_resolution: tuple[int, int] = (64, 64)
    # Engine: as fast as possible without graphics; with graphical=True a window shows the episodes at speed.
    graphical: bool = False
    speed: float = 1000.0
    # Keep runs/combat_environments/<run>/ (start script, engine log) after close(), for debugging.
    keep_run_directory: bool = False

    def __post_init__(self) -> None:
        if self.decision_interval_frames < 1:
            raise ValueError("decision_interval_frames must be at least 1")
        if self.terrain_resolution[0] < 1 or self.terrain_resolution[1] < 1:
            raise ValueError("terrain_resolution dimensions must be at least 1")

    @property
    def maximum_episode_frames(self) -> int:
        return int(self.maximum_episode_minutes * 60 * FRAMES_PER_SECOND)
