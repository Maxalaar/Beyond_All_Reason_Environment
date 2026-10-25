"""Curriculum step 3: the 5v5 arena with 5 Pawns on the west side and 1 to 5 on the east side, drawn for each game.

Against opponents that come to fight (AttackNearest), the agents cannot avoid the fight; the drawn number of opponents
gives easy games (1 or 2) as well as hard ones (5) from the start.

Typical game:
    GameConfiguration(five_versus_one_to_five_arena, controllers=[Agent(), AttackNearest()])
"""

from dataclasses import replace

from ..scenario import BLUE, RED, SpawnZone, TeamConfiguration
from ..units import units
from .five_versus_five_arena import five_versus_five_arena

five_versus_one_to_five_arena = replace(
    five_versus_five_arena,
    teams=[
        TeamConfiguration(units={units.armada.bot.t1.pawn: 5}, spawn_zone=SpawnZone(0.0, 0.05, 0.4, 0.6), color=BLUE),
        TeamConfiguration(
            units={units.armada.bot.t1.pawn: (1, 5)}, spawn_zone=SpawnZone(0.95, 1.0, 0.4, 0.6), color=RED
        ),
    ],
)
