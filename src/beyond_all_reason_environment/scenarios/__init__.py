"""Predefined scenarios: what happens in the game, ready to use as CombatConfiguration(scenario=...).

A scenario says nothing of the agents (observation, actions, rewards): an experiment chooses those.
Each file defines a ScenarioConfiguration named like the file.
"""

from .five_versus_five import five_versus_five
from .one_versus_one import one_versus_one

__all__ = ["five_versus_five", "one_versus_one"]
