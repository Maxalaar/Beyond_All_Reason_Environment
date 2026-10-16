"""5v5 Pawns on a small map: the baseline scenario for combat environment tests and early training."""

from ..combat_configuration import CombatConfiguration
from ..units import units

five_versus_five_minimal = CombatConfiguration(
    agent_units={units.armada.bot.t1.pawn: 5},
    opponent_units={units.armada.bot.t1.pawn: 5},
    opponent_behaviour="hold_and_fire",
)
