"""The one_versus_one scenario, AttackNearest against a Passive Pawn, watched in a window.

AttackNearest (controllers.py) attacks the nearest enemy every game second, without Python: a baseline for the
agents, which should do at least as well. To try another controller, change it below, for example
BuiltInArtificialIntelligence("BARb").

Watch it with: python scripts/run.py configurations/runs/one_versus_one_attack_nearest.py
Measure it without a window: python scripts/run.py configurations/runs/one_versus_one_attack_nearest.py \
    --no-graphical --matches 10
Results go to runs/one_versus_one_attack_nearest/.
"""

from beyond_all_reason_environment import (
    AttackNearest,
    EngineConfiguration,
    GameConfiguration,
    GraphicsConfiguration,
    Passive,
    RunConfiguration,
)
from beyond_all_reason_environment.scenarios import one_versus_one

configuration = RunConfiguration(
    name="one_versus_one_attack_nearest",
    game=GameConfiguration(scenario=one_versus_one, controllers=[AttackNearest(interval_seconds=1.0), Passive()]),
    matches=1,
    first_seed=1,
    record_replays=True,
    engine=EngineConfiguration(
        # speed None: real time with the window (+ and - change it in game), as fast as possible without.
        speed=None,
        graphics=GraphicsConfiguration(sound=False, fullscreen=True, camera_keys="zqsd", close_delay_seconds=5.0),
    ),
)
