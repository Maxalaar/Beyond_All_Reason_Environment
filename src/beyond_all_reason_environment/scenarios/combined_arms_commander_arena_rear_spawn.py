"""combined_arms_commander_arena with each side spawned in the rear of its half: the half of it nearest its own edge
(spawn_depth 0.5), the same armies.

Why: with the whole half (spawn_depth 1.0), armies often start in contact and a commander can be lost in the first
seconds; the draw of the positions weighs much on who wins. From the rear, the armies have room to form up before the
fight. combined_arms_commander_self_play_ppo version_0 (trained on combined_arms_commander_arena, checkpoint 110)
against AttackNearest: 59 % victories over 300 games there, 75 % (23 % defeats) over 450 games here (2026-10-25).

Typical game:
    GameConfiguration(combined_arms_commander_arena_rear_spawn, controllers=[Agent(), AttackNearest()])
"""

from dataclasses import replace

from .combined_arms_commander_arena import combined_arms_commander_arena

combined_arms_commander_arena_rear_spawn = replace(
    combined_arms_commander_arena,
    teams=[replace(team, spawn_depth=0.5) for team in combined_arms_commander_arena.teams],
)
