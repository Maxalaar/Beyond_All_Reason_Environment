"""Curriculum step 1: 1v1 Pawn fight on the smallest possible flat arena.

Map: 128×128 squares = 1024×1024 elmos, auto-generated flat terrain (no water, no metal).
Reward: +1 victory, -1 defeat, 0 otherwise (no intermediate shaping).
Spawn: agent at x≈0 (left edge), opponent at x≈1 (right edge); deterministic start position.
  spawn_x_range controls the width of the zone: 0 = edges only, 1 = full half-map randomness.
Opponent: passive (hold fire, hold position) — stands still, does not shoot.

Typical usage:
    from beyond_all_reason_environment.scenarios.one_versus_one import CONFIGURATION
    env = CombatEnvironment(CONFIGURATION)
"""

from ..combat_configuration import COMMANDS, AllyteamConfiguration, CombatConfiguration, RewardWeights
from ..flat_map import ensure_flat_map

_MAP_NAME = "flat_arena_1024x1024"
_MAP_WIDTH_SQ = 128  # squares; 128 × 8 elmos = 1024 elmos
_MAP_HEIGHT_SQ = 128

ensure_flat_map(_MAP_NAME, _MAP_WIDTH_SQ, _MAP_HEIGHT_SQ)

CONFIGURATION = CombatConfiguration(
    map_name=_MAP_NAME,
    # Allyteams: agents (allyteam 0) vs passive opponent (allyteam 1)
    allyteams=[
        AllyteamConfiguration(units={"armpw": 1}),
        AllyteamConfiguration(units={"armpw": 1}, behaviour="passive"),
    ],
    # Spawn: deterministic at opposite edges, no z jitter
    spawn_x_range=0.0,
    spawn_jitter=0.0,
    # Actor observation: only 1 enemy, no allies, no buildings
    maximum_ally_units=0,
    maximum_enemy_units=1,
    maximum_ally_buildings=0,
    maximum_enemy_buildings=0,
    maximum_weapons_per_unit=1,
    # Critic observation: 1 unit per enemy allyteam
    maximum_units_per_enemy_allyteam=1,
    maximum_buildings_per_enemy_allyteam=0,
    # Reward: terminal only
    rewards=RewardWeights(
        damage_dealt=0.0,
        damage_received=0.0,
        kill=0.0,
        victory=1.0,
        defeat=-1.0,
    ),
    # Episode length: 2 min leaves room for the pawn to cross the 1024-elmo map and fight.
    maximum_episode_minutes=2.0,
    # Commands available to the agent
    commands={"default": list(COMMANDS)},
)
