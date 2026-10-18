"""A full game, BARb as Armada against BARb as Cortex on Red Comet Remake 1.8 (small 1v1 map, 12 x 8).

Normal start (commander and economy); start positions are drawn at random from the seed of each match.
"""

from ..controllers import BuiltInArtificialIntelligence
from ..scenario import ScenarioConfiguration, TeamConfiguration

armada_versus_cortex = ScenarioConfiguration(
    map_name="Red Comet Remake 1.8",
    teams=[
        TeamConfiguration(controller=BuiltInArtificialIntelligence("BARb", version="stable"), side="Armada"),
        TeamConfiguration(controller=BuiltInArtificialIntelligence("BARb", version="stable"), side="Cortex"),
    ],
    maximum_game_minutes=60,
)
