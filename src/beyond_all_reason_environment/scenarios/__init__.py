"""Predefined scenarios (scenario.ScenarioConfiguration): what happens in the game, whatever runs it.

A scenario with agents (controllers.Agent) runs in the combat environment, CombatConfiguration(scenario=...), or in
matches once another controller plays the agents' allyteam (ScenarioConfiguration.with_controller); a scenario
without agents runs in matches, BatchConfiguration(scenario=...). A scenario says nothing of how the
agents see, act or are rewarded: an experiment chooses those. Each file defines a scenario named like the file.
"""

from .armada_versus_cortex import armada_versus_cortex
from .five_versus_five import five_versus_five
from .one_versus_one import one_versus_one

__all__ = ["armada_versus_cortex", "five_versus_five", "one_versus_one"]
