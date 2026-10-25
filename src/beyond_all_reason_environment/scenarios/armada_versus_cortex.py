"""A full game, Armada against Cortex on Red Comet Remake 1.8 (small 1v1 map, 12 x 8).

Normal start (commander and economy); start positions are drawn at random from the seed of each game.
Typical game: GameConfiguration(armada_versus_cortex, controllers=[BuiltInArtificialIntelligence("BARb")] * 2).
"""

from ..scenario import BLUE, RED, ScenarioConfiguration, TeamConfiguration

armada_versus_cortex = ScenarioConfiguration(
    map_name="Red Comet Remake 1.8",
    teams=[TeamConfiguration(side="Armada", color=BLUE), TeamConfiguration(side="Cortex", color=RED)],
    maximum_game_minutes=60,
)
