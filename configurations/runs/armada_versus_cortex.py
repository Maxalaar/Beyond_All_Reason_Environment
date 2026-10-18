"""BARb against itself, Armada against Cortex: baseline to compare the factions and measure the noise.

Scenario armada_versus_cortex (scenarios/armada_versus_cortex.py): each team keeps its faction in every game;
start positions are drawn at random from the seed. A team can also get a list of factions, for example
side=["Cortex", "Legion"]: each game then draws one of them, and summary.txt gives the wins per faction.

Run with: python scripts/run.py configurations/runs/armada_versus_cortex.py
Results go to runs/armada_versus_cortex/.
"""

from beyond_all_reason_environment import BuiltInArtificialIntelligence, GameConfiguration, RunConfiguration
from beyond_all_reason_environment.scenarios import armada_versus_cortex

configuration = RunConfiguration(
    name="armada_versus_cortex",
    game=GameConfiguration(
        scenario=armada_versus_cortex,
        controllers=[BuiltInArtificialIntelligence("BARb", version="stable")] * 2,
    ),
    matches=20,
    parallel=4,
    first_seed=1,
    record_replays=True,
    keep_engine_files=False,
)
