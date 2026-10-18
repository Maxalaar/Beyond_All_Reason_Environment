"""5v5 Pawns on a small map: the baseline scenario for combat environment tests and early training."""

from ..combat_configuration import AllyteamConfiguration, ScenarioConfiguration
from ..units import units

five_versus_five = ScenarioConfiguration(
    allyteams=[
        AllyteamConfiguration(units={units.armada.bot.t1.pawn: 5}),
        AllyteamConfiguration(units={units.armada.bot.t1.pawn: 5}, behaviour="hold_and_fire"),
    ],
)
