"""A large battle, watched in a window: 300 Armada units against 300 Cortex units, both played by AttackNearest.

Scenario armada_versus_cortex_battle (scenarios/armada_versus_cortex_battle.py): bots, vehicles, artillery and
aircraft on Red Comet Remake 1.8, random start positions in the western and eastern parts of the map, no commander.

Watch it with: python scripts/run.py configurations/runs/armada_versus_cortex_battle.py
Measure it without a window: python scripts/run.py configurations/runs/armada_versus_cortex_battle.py \
    --no-show --matches 10
Results go to runs/armada_versus_cortex_battle/.
"""

from beyond_all_reason_environment import (
    AttackNearest,
    EngineConfiguration,
    GameConfiguration,
    GraphicsConfiguration,
    RunConfiguration,
)
from beyond_all_reason_environment.scenarios import armada_versus_cortex_battle

configuration = RunConfiguration(
    name="armada_versus_cortex_battle",
    game=GameConfiguration(
        scenario=armada_versus_cortex_battle,
        controllers=[AttackNearest(interval_seconds=1.0), AttackNearest(interval_seconds=1.0)],
    ),
    matches=1,
    parallel=4,
    first_seed=1,
    record_replays=True,
    engine=EngineConfiguration(
        # speed None: real time with the window (+ and - change it in game), as fast as possible without.
        speed=None,
        graphics=GraphicsConfiguration(show=True, sound=False, fullscreen=True, camera_keys="zqsd"),
    ),
)
