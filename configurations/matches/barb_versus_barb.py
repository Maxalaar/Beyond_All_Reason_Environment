"""BARb against itself, Armada against Cortex: baseline to compare the factions and measure the noise.

Scenario armada_versus_cortex (scenarios/armada_versus_cortex.py): each player keeps its faction in every match;
start positions are drawn at random from the seed. A player can also get a list of factions, for example
side=["Cortex", "Legion"]: each match then draws one of them, and summary.txt gives the wins per faction.

Run with: python scripts/run_matches.py configurations/matches/barb_versus_barb.py
Results go to runs/barb_versus_barb/.
"""

from beyond_all_reason_environment import BatchConfiguration
from beyond_all_reason_environment.scenarios import armada_versus_cortex

configuration = BatchConfiguration(
    name="barb_versus_barb",
    scenario=armada_versus_cortex,
    matches=10,
    parallel=4,
    first_seed=1,
    record_replays=True,
    keep_engine_files=False,
)
