"""How the engine of an environment runs: shared by all environment types, not part of the experiment."""

from dataclasses import dataclass


@dataclass
class EngineConfiguration:
    """Changing these does not change what the agents see or do, only how fast and how visibly it runs."""

    # As fast as possible without graphics; with graphical=True a window shows the episodes at speed.
    graphical: bool = False
    speed: float = 1000.0
    # Keep runs/combat_environments/<run>/ (start script, engine log) after close(), for debugging.
    keep_run_directory: bool = False
