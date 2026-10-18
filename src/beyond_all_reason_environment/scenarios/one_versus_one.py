"""Curriculum step 1: 1v1 Pawn fight on the smallest possible flat arena.

Map: 128×128 squares = 1024×1024 elmos, flat terrain generated the first time it is needed (no water, no metal).
Spawn: team 0 on the west edge, team 1 on the east edge, both at mid-height; deterministic (point spawn zones).
  Give the teams a spawn_depth instead of a spawn_zone for random positions in their half of the map.
Episode length: 2 minutes leaves room for a pawn to cross the map and fight.

Typical games:
    GameConfiguration(one_versus_one, controllers=[Agent(), Passive()])        # learn against a still target
    GameConfiguration(one_versus_one, controllers=[AttackNearest(), Passive()])  # a scripted baseline, no Python
"""

from ..scenario import ScenarioConfiguration, SpawnZone, TeamConfiguration
from ..units import units

one_versus_one = ScenarioConfiguration(
    map_name="flat_arena_1024x1024",
    flat_map_squares=(128, 128),  # 128 squares × 8 elmos = 1024 elmos
    teams=[
        TeamConfiguration(units={units.armada.bot.t1.pawn: 1}, spawn_zone=SpawnZone(0.0, 0.0, 0.5, 0.5)),
        TeamConfiguration(units={units.armada.bot.t1.pawn: 1}, spawn_zone=SpawnZone(1.0, 1.0, 0.5, 0.5)),
    ],
    maximum_game_minutes=2.0,
    full_vision=True,
)
