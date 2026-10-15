"""Download the pinned engine, game and maps into game_data/.

Examples:
    python scripts/download_game_data.py                                  # engine, game and the default map
    python scripts/download_game_data.py --map "Comet Catcher Remake 1.8" --map "Archsimkats_Valley_V1"

Map names are the ones of the lobby (springName in https://maps-metadata.beyondallreason.dev/latest/lobby_maps.validated.json).
"""

import argparse

from beyond_all_reason_environment.installation import DEFAULT_MAP, download_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", action="append", dest="maps", help="map to download, can be repeated")
    arguments = parser.parse_args()
    download_all(arguments.maps or [DEFAULT_MAP])


if __name__ == "__main__":
    main()
