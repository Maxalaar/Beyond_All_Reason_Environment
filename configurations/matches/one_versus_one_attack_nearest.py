"""The one_versus_one scenario of the combat environment, with a scripted controller playing the agent's Pawn.

AttackNearest (controllers.py) attacks the nearest enemy every game second, without Python: a baseline for the
agents, which should do at least as well. The opponent is the scenario's own (a Passive Pawn). To try another
controller, change it below, for example BuiltInArtificialIntelligence("BARb").

Watch one match with: python scripts/run_match.py configurations/matches/one_versus_one_attack_nearest.py --graphical
Run the series with: python scripts/run_matches.py configurations/matches/one_versus_one_attack_nearest.py
Results go to runs/one_versus_one_attack_nearest/.
"""

from beyond_all_reason_environment import AttackNearest, BatchConfiguration
from beyond_all_reason_environment.scenarios import one_versus_one

configuration = BatchConfiguration(
    name="one_versus_one_attack_nearest",
    scenario=one_versus_one.with_controller(0, AttackNearest(interval_seconds=1.0)),
    matches=10,
    parallel=4,
    first_seed=1,
    record_replays=True,
    keep_engine_files=False,
)
