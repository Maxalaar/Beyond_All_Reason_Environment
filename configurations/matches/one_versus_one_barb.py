"""The one_versus_one scenario of the combat environment, with BARb playing the agent's Pawn instead of Python.

To see what a built-in artificial intelligence does in the scenario, and to measure it as a baseline for the agents:
the opponent is the scenario's own (a passive Pawn played by NullAI).

Watch one match with: python scripts/run_match.py configurations/matches/one_versus_one_barb.py --graphical
Run the series with: python scripts/run_matches.py configurations/matches/one_versus_one_barb.py
Results go to runs/one_versus_one_barb/.
"""

from beyond_all_reason_environment import ArtificialIntelligencePlayer, BatchConfiguration
from beyond_all_reason_environment.scenarios import one_versus_one

configuration = BatchConfiguration(
    name="one_versus_one_barb",
    scenario=one_versus_one.with_agents_played_by(ArtificialIntelligencePlayer(short_name="BARb", version="stable")),
    matches=10,
    parallel=4,
    first_seed=1,
    record_replays=True,
    keep_engine_files=False,
)
