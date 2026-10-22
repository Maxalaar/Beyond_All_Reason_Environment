"""Combined arms: 3 Shellshockers, 5 Stouts and 10 Pawns on the west side against 15 to 30 Pawns on the east side,
drawn for each game, in the 1v1 arena (flat 1024×1024 elmos, 2 minutes).

Units (from the engine, against the Pawns' armor):
    Pawn (armpw)          370 health, speed 87, range 180, 9 damage every 0.3 s
    Stout (armstump)      1800 health, speed 75, range 350, 97 damage every 1.2 s (light tank: the defence line)
    Shellshocker (armart) 620 health, speed 54, range 710, 182 damage every 4.3 s, area 38 (artillery, slow shells)
Against opponents that come to fight (AttackNearest: each attacks the nearest enemy), the west side gains by holding
a line of Stouts in front of the Pawns and the Shellshockers, which then fire on opponents busy with the Stouts.

Spawn: each team in a band of 0.15 × 0.5 of the map (~150 × 500 elmos) at the middle of its edge, for up to 30 units;
positions drawn at random in it (the west side's units mixed).

Typical game:
    GameConfiguration(combined_arms_arena, controllers=[Agent(), AttackNearest()])
"""

from dataclasses import replace

from ..scenario import SpawnZone, TeamConfiguration
from ..units import units
from .one_versus_one import one_versus_one

combined_arms_arena = replace(
    one_versus_one,
    teams=[
        TeamConfiguration(
            units={
                units.armada.vehicle.t1.shellshocker: 3,
                units.armada.vehicle.t1.stout: 5,
                units.armada.bot.t1.pawn: 10,
            },
            spawn_zone=SpawnZone(0.0, 0.15, 0.25, 0.75),
        ),
        TeamConfiguration(units={units.armada.bot.t1.pawn: (15, 30)}, spawn_zone=SpawnZone(0.85, 1.0, 0.25, 0.75)),
    ],
)
