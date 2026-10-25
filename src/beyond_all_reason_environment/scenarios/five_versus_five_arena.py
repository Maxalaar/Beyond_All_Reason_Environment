"""Curriculum step 2: the 1v1 arena (one_versus_one: flat 1024×1024 elmos, 2 minutes) with 5 Pawns per side.

Spawn: each team in a band of 0.05 × 0.2 of the map (~50 × 200 elmos) at the middle of its edge, west and east as in
one_versus_one; positions drawn at random in it.

Typical games:
    GameConfiguration(five_versus_five_arena, controllers=[Agent(), Passive()])      # learn to attack still targets
    GameConfiguration(five_versus_five_arena, controllers=[Agent(), HoldAndFire()])  # then targets that shoot back
"""

from dataclasses import replace

from ..scenario import BLUE, RED, SpawnZone, TeamConfiguration
from ..units import units
from .one_versus_one import one_versus_one

five_versus_five_arena = replace(
    one_versus_one,
    teams=[
        TeamConfiguration(units={units.armada.bot.t1.pawn: 5}, spawn_zone=SpawnZone(0.0, 0.05, 0.4, 0.6), color=BLUE),
        TeamConfiguration(units={units.armada.bot.t1.pawn: 5}, spawn_zone=SpawnZone(0.95, 1.0, 0.4, 0.6), color=RED),
    ],
)
