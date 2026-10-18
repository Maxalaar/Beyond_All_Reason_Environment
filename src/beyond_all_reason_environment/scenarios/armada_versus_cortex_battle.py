"""A large battle on a small map: 300 Armada units (team 0, west) against 300 Cortex units (team 1, east).

Each army mixes bots, vehicles, artillery and aircraft of tier 1, the same number in each role on both sides.
Map: Red Comet Remake 1.8 (12 x 8, a small 1v1 map). Units start at random in the western and eastern thirds of the
map (spawn_depth 0.6 of each half), with no commander; the game ends when an army is destroyed, or after 10 minutes.

Typical game: GameConfiguration(armada_versus_cortex_battle, controllers=[AttackNearest(), AttackNearest()]).
"""

from ..scenario import ScenarioConfiguration, TeamConfiguration
from ..units import units

armada, cortex = units.armada, units.cortex

armada_versus_cortex_battle = ScenarioConfiguration(
    map_name="Red Comet Remake 1.8",
    teams=[
        TeamConfiguration(
            side="Armada",
            spawn_depth=0.6,
            units={
                # Bots: 150
                armada.bot.t1.pawn: 60,
                armada.bot.t1.rocketeer: 30,
                armada.bot.t1.mace: 20,
                armada.bot.t1.centurion: 20,
                armada.bot.t1.crossbow: 20,  # anti-air
                # Vehicles: 85
                armada.vehicle.t1.blitz: 30,
                armada.vehicle.t1.stout: 30,
                armada.vehicle.t1.janus: 15,
                armada.vehicle.t1.whistler: 10,  # anti-air
                # Artillery: 25
                armada.vehicle.t1.shellshocker: 25,
                # Aircraft: 40
                armada.aircraft.t1.falcon: 15,  # fighter
                armada.aircraft.t1.stormbringer: 15,  # bomber
                armada.aircraft.t1.banshee: 10,  # gunship
            },
        ),
        TeamConfiguration(
            side="Cortex",
            spawn_depth=0.6,
            units={
                # Bots: 150
                cortex.bot.t1.grunt: 60,
                cortex.bot.t1.aggravator: 30,
                cortex.bot.t1.thug: 40,
                cortex.bot.t1.trasher: 20,  # anti-air
                # Vehicles: 85
                cortex.vehicle.t1.incisor: 30,
                cortex.vehicle.t1.brute: 30,
                cortex.vehicle.t1.pounder: 15,
                cortex.vehicle.t1.lasher: 10,  # anti-air
                # Artillery: 25
                cortex.vehicle.t1.wolverine: 25,
                # Aircraft: 40
                cortex.aircraft.t1.valiant: 15,  # fighter
                cortex.aircraft.t1.whirlwind: 15,  # bomber
                cortex.aircraft.t1.shuriken: 10,  # gunship
            },
        ),
    ],
    maximum_game_minutes=10,
)
