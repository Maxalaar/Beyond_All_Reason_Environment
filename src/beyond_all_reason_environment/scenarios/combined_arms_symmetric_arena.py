"""Combined arms on both sides: 3 Shellshockers, 5 Stouts and 10 Pawns on each side, in the 1v1 arena (flat 1024×1024
elmos, 2 minutes); the armies of combined_arms_arena's west side, mirrored.

Made for self-play (Agent() on both sides, one policy against itself): the numbers are fixed, as agent teams need.

Spawn: each team in a band of 0.15 × 0.5 of the map (~150 × 500 elmos) at the middle of its edge; positions drawn at
random in it (the units of a side mixed).

Typical game:
    GameConfiguration(combined_arms_symmetric_arena, controllers=[Agent(), Agent()])
"""

from dataclasses import replace

from ..scenario import SpawnZone, TeamConfiguration
from ..units import units
from .one_versus_one import one_versus_one

ARMY = {units.armada.vehicle.t1.shellshocker: 3, units.armada.vehicle.t1.stout: 5, units.armada.bot.t1.pawn: 10}

combined_arms_symmetric_arena = replace(
    one_versus_one,
    teams=[
        TeamConfiguration(units=dict(ARMY), spawn_zone=SpawnZone(0.0, 0.15, 0.25, 0.75)),
        TeamConfiguration(units=dict(ARMY), spawn_zone=SpawnZone(0.85, 1.0, 0.25, 0.75)),
    ],
)
