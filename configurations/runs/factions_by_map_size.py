"""Armada against Cortex on three map sizes: does the faction advantage depend on the map?

60 matches, taking the maps in turn: 20 on each. Both players are BARb, so the only difference
between them is the faction; start positions are drawn at random from the seed.

    small   Red Comet Remake 1.8        12 x 8   (tourney and popular 1v1 pool)
    medium  Comet Catcher Remake 1.8    16 x 12  (tourney, competitive and popular 1v1 pool)
    large   Archsimkats_Valley_V1       16 x 16  (tourney, competitive and popular 1v1 pool)

Sizes in map units (512 game units each), from
https://maps-metadata.beyondallreason.dev/latest/lobby_maps.validated.json. Download the maps first:
    python scripts/download_game_data.py --map "Comet Catcher Remake 1.8" --map "Archsimkats_Valley_V1"
Run with: python scripts/run_matches.py configurations/matches/factions_by_map_size.py (about an hour).
"""

from dataclasses import replace

from beyond_all_reason_environment import BuiltInArtificialIntelligence, GameConfiguration, RunConfiguration
from beyond_all_reason_environment.scenarios import armada_versus_cortex

configuration = RunConfiguration(
    name="factions_by_map_size",
    game=GameConfiguration(
        # The armada_versus_cortex scenario, on three maps and with a longer time limit.
        scenario=replace(
            armada_versus_cortex,
            map_name=["Red Comet Remake 1.8", "Comet Catcher Remake 1.8", "Archsimkats_Valley_V1"],
            # Larger maps make longer games; 60 minutes already stopped 8 games out of 20 on the small map.
            maximum_game_minutes=90,
        ),
        controllers=[BuiltInArtificialIntelligence("BARb", version="stable")] * 2,
    ),
    matches=60,
    parallel=4,
    first_seed=1,
    record_replays=True,
    keep_engine_files=False,
)
