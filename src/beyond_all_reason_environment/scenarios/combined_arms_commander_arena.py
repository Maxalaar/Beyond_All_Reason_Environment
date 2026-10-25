"""Combined arms around a commander, on both sides: a commander, 3 Shellshockers, 5 Stouts and 10 Pawns each, in the
1v1 arena (flat 1024×1024 elmos, 2 minutes); the game ends when a commander dies, its side loses.

Units (from the engine, against the Pawns' armor):
    Pawn (armpw)          370 health, speed 87, range 180, 9 damage every 0.3 s
    Stout (armstump)      1800 health, speed 75, range 350, 97 damage every 1.2 s (light tank)
    Shellshocker (armart) 620 health, speed 54, range 710, 182 damage every 4.3 s, area 38 (artillery, slow shells)
    Commander (armcom)    3700 health, speed 37.5, range 300, 75 damage every 0.4 s; its D-gun needs a manual fire
                          command that no controller gives, and it explodes when it dies
The commander is a builder. Under a "fight" order (CBuilderCAI::ExecuteFight), the engine looks around it, within
300 elmos × its move state + its build distance (~450 elmos on manoeuvre): it attacks an enemy there first, otherwise
repairs a damaged ally, otherwise reclaims a wreck, otherwise goes on to the point. Left without orders, it does
nothing on its own.

Spawn: each side anywhere in its half of the map (spawn_depth 1.0: the whole half, SPAWN_MARGIN from the borders),
positions drawn uniformly: armies scattered, sometimes in contact at once, which leaves no corner to wait in for a
draw; a formation has to be built by moving.

Made for self-play (Agent() on both sides): the numbers are fixed, as agent teams need.

Typical game:
    GameConfiguration(combined_arms_commander_arena, controllers=[Agent(), Agent()])
"""

from dataclasses import replace

from ..scenario import TeamConfiguration
from ..units import units
from .one_versus_one import one_versus_one

ARMY = {
    units.armada.commander: 1,
    units.armada.vehicle.t1.shellshocker: 3,
    units.armada.vehicle.t1.stout: 5,
    units.armada.bot.t1.pawn: 10,
}

combined_arms_commander_arena = replace(
    one_versus_one,
    teams=[TeamConfiguration(units=dict(ARMY), spawn_depth=1.0), TeamConfiguration(units=dict(ARMY), spawn_depth=1.0)],
    commander_death_ends_game=True,
)
