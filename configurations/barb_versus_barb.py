"""BARb against itself: baseline to measure the noise of the results.

Run with: python scripts/run_matches.py configurations/barb_versus_barb.py
Results go to runs/barb_versus_barb/.
"""

from beyond_all_reason_environment import ArtificialIntelligencePlayer, BatchConfiguration

configuration = BatchConfiguration(
    name="barb_versus_barb",
    matches=10,
    parallel=4,
    first_seed=1,
    alternate_sides=True,
    maximum_game_minutes=60,
    map_name="Red Comet Remake 1.8",
    players=[
        ArtificialIntelligencePlayer(short_name="BARb", version="stable", side="Armada"),
        ArtificialIntelligencePlayer(short_name="BARb", version="stable", side="Cortex"),
    ],
)
