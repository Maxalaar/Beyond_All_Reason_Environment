"""5v5 Pawns on a small map: the baseline scenario for reinforcement learning environment tests and early training.

Typical game: GameConfiguration(five_versus_five, controllers=[Agent(), HoldAndFire()]); [Agent(), Agent()] for
self-play.
"""

from ..scenario import ScenarioConfiguration, SpawnZone, TeamConfiguration
from ..units import units

five_versus_five = ScenarioConfiguration(
    teams=[
        # Two small groups facing each other around the map centre.
        TeamConfiguration(units={units.armada.bot.t1.pawn: 5}, spawn_zone=SpawnZone(0.41, 0.44, 0.45, 0.55)),
        TeamConfiguration(units={units.armada.bot.t1.pawn: 5}, spawn_zone=SpawnZone(0.56, 0.59, 0.45, 0.55)),
    ],
    maximum_game_minutes=1.0,
    full_vision=True,
)
