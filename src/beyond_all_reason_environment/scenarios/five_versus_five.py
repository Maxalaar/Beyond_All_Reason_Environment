"""5v5 Pawns on a small map: the baseline scenario for combat environment tests and early training."""

from ..scenario import AGENT, NULL_ARTIFICIAL_INTELLIGENCE, AllyteamConfiguration, ScenarioConfiguration
from ..units import units

five_versus_five = ScenarioConfiguration(
    allyteams=[
        AllyteamConfiguration(player=AGENT, units={units.armada.bot.t1.pawn: 5}),
        AllyteamConfiguration(
            player=NULL_ARTIFICIAL_INTELLIGENCE, units={units.armada.bot.t1.pawn: 5}, behaviour="hold_and_fire"
        ),
    ],
    maximum_game_minutes=1.0,
    full_vision=True,
)
