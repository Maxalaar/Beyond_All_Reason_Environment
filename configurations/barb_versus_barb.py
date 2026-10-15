"""BARb against itself, Armada against Cortex: baseline to compare the factions and measure the noise.

Each player keeps its faction in every match; start positions are drawn at random from the seed.
A player can also get a list of factions, for example side=["Cortex", "Legion"]: each match then
draws one of them, and summary.txt gives the wins per faction.

Run with: python scripts/run_matches.py configurations/barb_versus_barb.py
Results go to runs/barb_versus_barb/.
"""

from beyond_all_reason_environment import ArtificialIntelligencePlayer, BatchConfiguration

configuration = BatchConfiguration(
    name="barb_versus_barb",
    matches=10,
    parallel=4,
    first_seed=1,
    maximum_game_minutes=60,
    map_name="Red Comet Remake 1.8",
    record_replays=True,
    keep_engine_files=False,
    players=[
        ArtificialIntelligencePlayer(short_name="BARb", version="stable", side="Armada"),
        ArtificialIntelligencePlayer(short_name="BARb", version="stable", side="Cortex"),
    ],
)
