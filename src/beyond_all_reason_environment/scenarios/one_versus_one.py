"""Curriculum step 1: 1v1 Pawn fight on the smallest possible flat arena.

Map: 128×128 squares = 1024×1024 elmos, flat terrain generated the first time it is needed (no water, no metal).
Spawn: agent at x≈0 (left edge), opponent at x≈1 (right edge); deterministic start position.
  spawn_x_range controls the width of the zone: 0 = edges only, 1 = full half-map randomness.
Opponent: NullAI, passive (hold fire, hold position) — stands still, does not shoot.
Episode length: 2 minutes leaves room for the pawn to cross the map and fight.

Typical usage:
    environment = CombatEnvironment(CombatConfiguration(scenario=one_versus_one))
    # or watch BARb play the agent's Pawn: configurations/matches/one_versus_one_barb.py
"""

from ..scenario import AGENT, NULL_ARTIFICIAL_INTELLIGENCE, AllyteamConfiguration, ScenarioConfiguration
from ..units import units

one_versus_one = ScenarioConfiguration(
    map_name="flat_arena_1024x1024",
    flat_map_squares=(128, 128),  # 128 squares × 8 elmos = 1024 elmos
    allyteams=[
        AllyteamConfiguration(player=AGENT, units={units.armada.bot.t1.pawn: 1}),
        AllyteamConfiguration(
            player=NULL_ARTIFICIAL_INTELLIGENCE, units={units.armada.bot.t1.pawn: 1}, behaviour="passive"
        ),
    ],
    maximum_game_minutes=2.0,
    full_vision=True,
    # Deterministic at opposite edges, no z jitter.
    spawn_x_range=0.0,
    spawn_jitter=0.0,
)
