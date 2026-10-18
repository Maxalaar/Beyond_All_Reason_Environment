"""5v5 Pawns on a small map: the baseline scenario for reinforcement learning environment tests and early training.

Typical game: GameConfiguration(five_versus_five, controllers=[Agent(), HoldAndFire()]); [Agent(), Agent()] for
self-play.
"""

from ..scenario import ScenarioConfiguration, TeamConfiguration
from ..units import units

five_versus_five = ScenarioConfiguration(
    teams=[
        TeamConfiguration(units={units.armada.bot.t1.pawn: 5}),
        TeamConfiguration(units={units.armada.bot.t1.pawn: 5}),
    ],
    maximum_game_minutes=1.0,
    full_vision=True,
)
