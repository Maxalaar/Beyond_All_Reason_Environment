"""5v5 Pawns on a small map: the baseline scenario for combat environment tests and early training."""

from ..controllers import Agent, HoldAndFire
from ..scenario import AllyteamConfiguration, ScenarioConfiguration
from ..units import units

five_versus_five = ScenarioConfiguration(
    allyteams=[
        AllyteamConfiguration(controller=Agent(), units={units.armada.bot.t1.pawn: 5}),
        AllyteamConfiguration(controller=HoldAndFire(), units={units.armada.bot.t1.pawn: 5}),
    ],
    maximum_game_minutes=1.0,
    full_vision=True,
)
