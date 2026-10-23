"""DPL's HUD in Driv3r's colours for the Driv3r city levels: DPL's layout and graphics, Driv3r's monochrome look.

usage: d3_hud.py <DPL dir> [--level | --restore-level]

Writes Overlays/HUD-MIAMI.gfx and .bin, copies of DPL's HUD-now pair; DPL's own files are never written. --level lets
the first LEVEL of Cities/levels.txt (the Driv3r city level) load them (Hud), --restore-level points it back.
- .bin: the HUD's theme colour, an RGB float triple (now: cyan 0.094 0.773 0.937, 67 times; then: orange), becomes
  Driv3r's white; every other value stays.
- .gfx: a texture pack of whole DDS files (DXT1/3/5). Blue pixels (speedometer, weapon icons, shield) turn grey of
  the same mean brightness, as Driv3r's grey icons; red, green and white stay. Only the two 565 end colours of each
  4x4 block change, so nothing is recompressed; a DXT1 block keeps its colour mode (c0 > c1: four colours, else
  three and transparent): end colours that would swap order are swapped back with their indices.
"""
import argparse
import colorsys
import struct
from pathlib import Path

import numpy as np

import d3_vehicle_swap as s

SOURCE = 'HUD-now'
TARGET = 'HUD-MIAMI'
LEVEL_KEY = b'Hud'
THEME = (0.094117650, 0.77254903, 0.93725491)        # f32 words 0x3dc0c0c1 0x3f45c5c6 0x3f6feff0
DRIV3R_WHITE = (0.92, 0.92, 0.92)
BLUE_HUES = (160 / 360, 260 / 360)
MIN_SATURATION = 0.25
BLOCK_BYTES = {b'DXT1': 8, b'DXT3': 16, b'DXT5': 16}
COLOUR_AT = {b'DXT1': 0, b'DXT3': 8, b'DXT5': 8}     # the colour half of a block


def grey565(c: int) -> int:
    """A 565 colour; blue ones as the grey of their mean brightness."""
    r, g, b = (c >> 11) / 31, (c >> 5 & 63) / 63, (c & 31) / 31
    hue, saturation, _ = colorsys.rgb_to_hsv(r, g, b)

    if not (BLUE_HUES[0] <= hue <= BLUE_HUES[1] and saturation >= MIN_SATURATION):
        return c

    grey = (r + g + b) / 3
    return round(grey * 31) << 11 | round(grey * 63) << 5 | round(grey * 31)


def recolour_block(block: bytearray, at: int, dxt1: bool) -> bool:
    c0, c1, indices = struct.unpack_from('<2HI', block, at)
    n0, n1 = grey565(c0), grey565(c1)

    if (n0, n1) == (c0, c1):
        return False

    if dxt1 and c0 > c1 and n0 <= n1:
        if n0 == n1:            # four-colour mode needs c0 > c1: one green step apart
            n1 = n1 - (1 << 5) if n1 & 0x7E0 else n1
            n0 = n0 if n0 > n1 else n0 + (1 << 5)
        else:                   # swap ends, and the indices with them (0 <-> 1, 2 <-> 3)
            n0, n1 = n1, n0
            indices ^= 0x55555555

    struct.pack_into('<2HI', block, at, n0, n1, indices)
    return True


def recolour_gfx(data: bytearray) -> tuple[int, int]:
    """Greys the blue of every DXT texture in place; (textures changed, blocks changed)."""
    textures = blocks = 0
    at = data.find(b'DDS ')

    while at >= 0:
        height, width, _, _, mips = struct.unpack_from('<5I', data, at + 12)
        code = bytes(data[at + 84:at + 88])

        if code in BLOCK_BYTES:
            size, colour = BLOCK_BYTES[code], COLOUR_AT[code]
            offset, changed = at + 128, 0

            for level in range(max(1, mips)):
                count = max(1, (width >> level) // 4) * max(1, (height >> level) // 4)

                for k in range(count):
                    changed += recolour_block(data, offset + k * size + colour, code == b'DXT1')

                offset += count * size

            textures += changed > 0
            blocks += changed

        at = data.find(b'DDS ', at + 4)

    return textures, blocks


def recolour_bin(data: bytearray) -> int:
    """The theme colour triple as Driv3r's white; how many were replaced."""
    theme, white = np.array(THEME, '<f4').tobytes(), np.array(DRIV3R_WHITE, '<f4').tobytes()
    count, at = 0, data.find(theme)

    while at >= 0:
        data[at:at + len(theme)] = white
        count += 1
        at = data.find(theme, at + len(theme))

    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('dpl', type=Path)
    ap.add_argument('--level', action='store_true', help=f'let the first LEVEL load Overlays\\{TARGET}')
    ap.add_argument('--restore-level', action='store_true', help=f'let the first LEVEL load Overlays\\{SOURCE} again')
    args = ap.parse_args()
    overlays = args.dpl / 'Overlays'

    if not (overlays / f'{SOURCE}.gfx').is_file():
        raise SystemExit(f'pass your Driver Parallel Lines folder (Overlays/{SOURCE}.gfx)')

    if args.restore_level:
        s.set_level(args.dpl, SOURCE.upper(), LEVEL_KEY, 'Overlays')
        return

    gfx = bytearray((overlays / f'{SOURCE}.gfx').read_bytes())
    textures, blocks = recolour_gfx(gfx)
    binary = bytearray((overlays / f'{SOURCE}.bin').read_bytes())
    colours = recolour_bin(binary)

    if not colours:
        raise SystemExit(f'{SOURCE}.bin: no theme colour {THEME} found')

    (overlays / f'{TARGET}.gfx').write_bytes(gfx)
    (overlays / f'{TARGET}.bin').write_bytes(binary)
    print(f'{TARGET}: {textures} textures greyed ({blocks} blocks), theme colour replaced {colours} times')

    if args.level:
        s.set_level(args.dpl, TARGET, LEVEL_KEY, 'Overlays')


if __name__ == '__main__':
    main()
