"""Predefined scenarios (scenario.ScenarioConfiguration): the situation of a game, whoever plays it.

A GameConfiguration adds who plays each team (controllers.py); it then runs as games without Python
(RunConfiguration, scripts/run.py) or as the reinforcement learning environment, where Agent() teams are played
from Python. Each file defines a scenario named like the file.
"""

from .armada_versus_cortex import armada_versus_cortex
from .five_versus_five import five_versus_five
from .one_versus_one import one_versus_one

__all__ = ["armada_versus_cortex", "five_versus_five", "one_versus_one"]
