"""Predefined scenarios: what happens in the game, ready to use as CombatConfiguration(scenario=...).

A scenario says nothing of the agents (observation, actions, rewards): an experiment chooses those.
Each file defines a ScenarioConfiguration named like the file.
"""

from .five_versus_five_minimal import five_versus_five_minimal
from .one_versus_one import one_versus_one

__all__ = ["five_versus_five_minimal", "one_versus_one"]
