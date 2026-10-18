"""Curriculum step 1: 1v1 Pawn fight on the smallest possible flat arena.

Map: 128×128 squares = 1024×1024 elmos, flat terrain generated the first time it is needed (no water, no metal).
Spawn: team 0 at x≈0 (left edge), team 1 at x≈1 (right edge); deterministic start position.
  spawn_x_range controls the width of the zone: 0 = edges only, 1 = full half-map randomness.
Episode length: 2 minutes leaves room for a pawn to cross the map and fight.

Typical games:
    GameConfiguration(one_versus_one, controllers=[Agent(), Passive()])        # learn against a still target
    GameConfiguration(one_versus_one, controllers=[AttackNearest(), Passive()])  # a scripted baseline, no Python
"""

from ..scenario import ScenarioConfiguration, TeamConfiguration
from ..units import units

one_versus_one = ScenarioConfiguration(
    map_name="flat_arena_1024x1024",
    flat_map_squares=(128, 128),  # 128 squares × 8 elmos = 1024 elmos
    teams=[
        TeamConfiguration(units={units.armada.bot.t1.pawn: 1}),
        TeamConfiguration(units={units.armada.bot.t1.pawn: 1}),
    ],
    maximum_game_minutes=2.0,
    full_vision=True,
    # Deterministic at opposite edges, no z jitter.
    spawn_x_range=0.0,
    spawn_jitter=0.0,
)
