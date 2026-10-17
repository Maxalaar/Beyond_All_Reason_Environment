"""Generate a flat, featureless Spring map (SMF/SMT format) written as a .sdz archive.

Spring requires mapx and mapy to be multiples of 128 squares (each square = 8 elmos).
The smallest valid map is therefore 128×128 squares = 1024×1024 elmos.

The archive layout:
    mapinfo.lua
    maps/<name>.smf   Spring Map File: heightmap, typemap, tilemap, minimap, metalmap
    maps/<name>.smt   Spring Map Tiles: single solid-colour 32×32 tile
"""

import struct
import zipfile
from pathlib import Path

from .installation import GAME_DATA_DIRECTORY

# Fixed Spring constraints
_SQUARE_SIZE = 8  # elmos per map square
_TILE_SIZE = 32  # texels per tile side
_TEXELS_PER_SQUARE = 8  # fixed by Spring

# Terrain colour: grey-green, neutral — RGB(96, 120, 64)
# R5 = 96>>3 = 12,  G6 = 120>>2 = 30,  B5 = 64>>3 = 8
_TERRAIN_COLOR = (12 << 11) | (30 << 5) | 8  # 25544 in RGB565

# Flat terrain height (elmos above sea level); must be > 0 to avoid water.
_TERRAIN_HEIGHT = 100.0
_HEIGHT_MIN = -10.0
_HEIGHT_MAX = 210.0
# uint16 value encoding _TERRAIN_HEIGHT between _HEIGHT_MIN and _HEIGHT_MAX:
_HEIGHT_VALUE = round((_TERRAIN_HEIGHT - _HEIGHT_MIN) / (_HEIGHT_MAX - _HEIGHT_MIN) * 65535)


def _dxt1_block(rgb565: int) -> bytes:
    """8-byte DXT1 block: all 16 pixels set to rgb565 (index 0 in 4-color mode)."""
    # color0 > color1 → 4-color mode; all 2-bit indices = 0 → all pixels = color0
    return struct.pack("<HHI", rgb565, 0, 0)


def _solid_dxt1(width_px: int, height_px: int, rgb565: int) -> bytes:
    """DXT1-compressed solid-color image; width and height must be multiples of 4."""
    return _dxt1_block(rgb565) * ((width_px // 4) * (height_px // 4))


def _smt_bytes(name: str) -> bytes:
    """SMT file content: one solid-colour 32×32 DXT1 tile."""
    # Header: magic(16) + version(4) + numTiles(4) + tileSize(4) + compressionType(4)
    header = b"spring tilefile\0" + struct.pack("<4i", 1, 1, _TILE_SIZE, 1)
    tile = _solid_dxt1(_TILE_SIZE, _TILE_SIZE, _TERRAIN_COLOR)  # 512 bytes
    return header + tile


def _smf_bytes(name: str, width_sq: int, height_sq: int) -> bytes:
    """SMF file content for a flat map at _TERRAIN_HEIGHT elmos."""
    # Heightmap: (w+1)*(h+1) uint16, row-major
    n_verts = (width_sq + 1) * (height_sq + 1)
    heightmap = struct.pack(f"<{n_verts}H", *([_HEIGHT_VALUE] * n_verts))

    # Typemap: (w/2)*(h/2) uint8, all terrain type 0
    typemap = bytes((width_sq // 2) * (height_sq // 2))

    # Tiles section: one .smt file, tile index 0 repeated everywhere
    smt_filename = f"maps/{name}.smt\0".encode()
    tiles_x = width_sq // 4  # tiles per row  (32 texels per tile, 8 per square → 4 sq per tile)
    tiles_z = height_sq // 4
    num_tiles = tiles_x * tiles_z
    tile_header = struct.pack("<ii", 1, num_tiles)  # numFiles=1, numTiles
    tile_file = struct.pack("<i", num_tiles) + smt_filename  # all tiles from this file
    tile_indices = struct.pack(f"<{num_tiles}i", *([0] * num_tiles))
    tiles_section = tile_header + tile_file + tile_indices

    # Minimap: always 1024×1024 DXT1
    minimap = _solid_dxt1(1024, 1024, _TERRAIN_COLOR)

    # Metalmap: (w/2)*(h/2) uint8, no metal
    metalmap = bytes((width_sq // 2) * (height_sq // 2))

    # Feature header: 0 types, 0 features
    feature_header = struct.pack("<ii", 0, 0)

    # Compute offsets
    header_size = 80
    heightmap_offset = header_size
    typemap_offset = heightmap_offset + len(heightmap)
    tiles_offset = typemap_offset + len(typemap)
    minimap_offset = tiles_offset + len(tiles_section)
    metalmap_offset = minimap_offset + len(minimap)
    feature_offset = metalmap_offset + len(metalmap)

    # SMF header: magic(16) + version,id,mapx,mapy(4×4) + squareSize,texelPerSquare,tileSize(3×4)
    #             + minHeight,maxHeight(2×4f) + 7 offsets(7×4) = 80 bytes
    header = (
        b"spring map file\0"
        + struct.pack("<4i", 1, 0x52_4C_4D_41, width_sq, height_sq)
        + struct.pack("<3i", _SQUARE_SIZE, _TEXELS_PER_SQUARE, _TILE_SIZE)
        + struct.pack("<2f", _HEIGHT_MIN, _HEIGHT_MAX)
        + struct.pack(
            "<7i",
            heightmap_offset,
            typemap_offset,
            tiles_offset,
            minimap_offset,
            metalmap_offset,
            feature_offset,
            0,  # numExtraHeaders
        )
    )
    assert len(header) == 80, f"SMF header is {len(header)} bytes, expected 80"

    return header + heightmap + typemap + tiles_section + minimap + metalmap + feature_header


def _mapinfo_lua(name: str, width_sq: int, height_sq: int) -> str:
    width_elmos = width_sq * _SQUARE_SIZE
    height_elmos = height_sq * _SQUARE_SIZE
    mid_z = height_elmos // 2
    margin = width_elmos // 8  # 12.5% from each edge for start positions
    return (
        f"return {{\n"
        f'    name        = "{name}",\n'
        f'    shortname   = "{name}",\n'
        f'    description = "Auto-generated flat arena for RL training",\n'
        f'    author      = "flat_map.py",\n'
        f'    version     = "1",\n'
        f'    mapfile     = "maps/{name}.smf",\n'
        f"    maphardness = 100.0,\n"
        f"    gravity     = 130.0,\n"
        f"    tidalstrength = 0.0,\n"
        f"    maxmetal    = 0.0,\n"
        f"    extractorradius = 500.0,\n"
        f"    notDeformable   = true,\n"
        f"    smf = {{\n"
        f"        minheight = {_HEIGHT_MIN},\n"
        f"        maxheight = {_HEIGHT_MAX},\n"
        f"    }},\n"
        f"    teams = {{\n"
        f"        [0] = {{ startpos = {{ x = {margin}, z = {mid_z} }} }},\n"
        f"        [1] = {{ startpos = {{ x = {width_elmos - margin}, z = {mid_z} }} }},\n"
        f"    }},\n"
        f"}}\n"
    )


def generate_flat_map(name: str, width_sq: int, height_sq: int) -> Path:
    """Generate a flat Spring map and write it to game_data/maps/<name>.sdz.

    Args:
        name: map name (used as archive name, .smf/.smt filenames, and mapinfo name)
        width_sq: map width in squares; must be a multiple of 128
        height_sq: map height in squares; must be a multiple of 128

    Returns:
        Path to the written .sdz archive.
    """
    if width_sq % 128 != 0 or height_sq % 128 != 0:
        raise ValueError(f"width_sq and height_sq must be multiples of 128, got {width_sq}×{height_sq}")

    output = GAME_DATA_DIRECTORY / "maps" / f"{name}.sdz"
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mapinfo.lua", _mapinfo_lua(name, width_sq, height_sq))
        zf.writestr(f"maps/{name}.smf", _smf_bytes(name, width_sq, height_sq))
        zf.writestr(f"maps/{name}.smt", _smt_bytes(name))

    return output


def ensure_flat_map(name: str, width_sq: int, height_sq: int) -> Path:
    """Generate the flat map only if the archive does not already exist."""
    output = GAME_DATA_DIRECTORY / "maps" / f"{name}.sdz"
    if not output.exists():
        generate_flat_map(name, width_sq, height_sq)
    return output
