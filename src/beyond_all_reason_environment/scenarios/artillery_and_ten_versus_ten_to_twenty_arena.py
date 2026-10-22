"""An artillery to protect: a Shellshocker and 10 Pawns on the west side against 10 to 20 Pawns on the east side, drawn
for each game, in the 1v1 arena (flat 1024×1024 elmos, 2 minutes).

The Shellshocker (armart, light artillery vehicle) outranges the Pawns but dies quickly at close range: against
opponents that come to fight (AttackNearest), the west side wins the uneven games by screening it while it fires,
not by charging.

Spawn: each team in a band of 0.1 × 0.4 of the map (~100 × 400 elmos) at the middle of its edge, wider than
five_versus_five_arena for up to 20 units; positions drawn at random in it (the Shellshocker among the Pawns).

Typical game:
    GameConfiguration(artillery_and_ten_versus_ten_to_twenty_arena, controllers=[Agent(), AttackNearest()])
"""

from dataclasses import replace

from ..scenario import SpawnZone, TeamConfiguration
from ..units import units
from .one_versus_one import one_versus_one

artillery_and_ten_versus_ten_to_twenty_arena = replace(
    one_versus_one,
    teams=[
        TeamConfiguration(
            units={units.armada.vehicle.t1.shellshocker: 1, units.armada.bot.t1.pawn: 10},
            spawn_zone=SpawnZone(0.0, 0.1, 0.3, 0.7),
        ),
        TeamConfiguration(units={units.armada.bot.t1.pawn: (10, 20)}, spawn_zone=SpawnZone(0.9, 1.0, 0.3, 0.7)),
    ],
)
