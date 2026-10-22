"""Between the two 5v5 steps of the curriculum: the 5v5 arena with only 2 Pawns on the east side.

Against opponents that shoot back (HoldAndFire), 5 against 5 starts with more defeats than victories, and a terminal
reward then teaches to avoid the fight (a draw is worth more than a defeat). Against 2, the 5 agents win by attacking
together: they learn that attacking pays despite the return fire, before facing 5.

Typical game:
    GameConfiguration(five_versus_two_arena, controllers=[Agent(), HoldAndFire()])
"""

from dataclasses import replace

from ..scenario import SpawnZone, TeamConfiguration
from ..units import units
from .five_versus_five_arena import five_versus_five_arena

five_versus_two_arena = replace(
    five_versus_five_arena,
    teams=[
        TeamConfiguration(units={units.armada.bot.t1.pawn: 5}, spawn_zone=SpawnZone(0.0, 0.05, 0.4, 0.6)),
        TeamConfiguration(units={units.armada.bot.t1.pawn: 2}, spawn_zone=SpawnZone(0.95, 1.0, 0.4, 0.6)),
    ],
)
