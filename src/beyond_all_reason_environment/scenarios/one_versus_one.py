"""Curriculum step 1: 1v1 Pawn fight on the smallest possible flat arena.

Map: 128×128 squares = 1024×1024 elmos, flat terrain generated at the first start (no water, no metal).
Spawn: agent at x≈0 (left edge), opponent at x≈1 (right edge); deterministic start position.
  spawn_x_range controls the width of the zone: 0 = edges only, 1 = full half-map randomness.
Opponent: passive (hold fire, hold position) — stands still, does not shoot.
Episode length: 2 minutes leaves room for the pawn to cross the map and fight.

Typical usage:
    from beyond_all_reason_environment.combat_configuration import CombatConfiguration
    from beyond_all_reason_environment.scenarios import one_versus_one
    environment = CombatEnvironment(CombatConfiguration(scenario=one_versus_one))
"""

from ..combat_configuration import AllyteamConfiguration, ScenarioConfiguration
from ..units import units

one_versus_one = ScenarioConfiguration(
    map_name="flat_arena_1024x1024",
    flat_map_squares=(128, 128),  # 128 squares × 8 elmos = 1024 elmos
    allyteams=[
        AllyteamConfiguration(units={units.armada.bot.t1.pawn: 1}),
        AllyteamConfiguration(units={units.armada.bot.t1.pawn: 1}, behaviour="passive"),
    ],
    maximum_episode_minutes=2.0,
    # Deterministic at opposite edges, no z jitter.
    spawn_x_range=0.0,
    spawn_jitter=0.0,
)
