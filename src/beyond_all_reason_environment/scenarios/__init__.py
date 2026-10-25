"""Predefined scenarios (scenario.ScenarioConfiguration): the situation of a game, whoever plays it.

A GameConfiguration adds who plays each team (controllers.py); it then runs as games without Python
(RunConfiguration, scripts/run.py) or as the reinforcement learning environment, where Agent() teams are played
from Python. Each file defines a scenario named like the file.
"""

from .armada_versus_cortex import armada_versus_cortex
from .armada_versus_cortex_battle import armada_versus_cortex_battle
from .combined_arms_arena import combined_arms_arena
from .combined_arms_commander_arena import combined_arms_commander_arena
from .combined_arms_symmetric_arena import combined_arms_symmetric_arena
from .five_versus_five import five_versus_five
from .five_versus_five_arena import five_versus_five_arena
from .five_versus_one_to_five_arena import five_versus_one_to_five_arena
from .one_versus_one import one_versus_one

__all__ = [
    "armada_versus_cortex",
    "armada_versus_cortex_battle",
    "combined_arms_arena",
    "combined_arms_commander_arena",
    "combined_arms_symmetric_arena",
    "five_versus_five",
    "five_versus_five_arena",
    "five_versus_one_to_five_arena",
    "one_versus_one",
]
