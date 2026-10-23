"""Driv3r terrain (GRTD v0, "THD v3.0") into DPL terrain (GRTD v2, version 1001), one chunk per DPL region.

Both games read the same layout (Driv3r 0x4E9FE0, DPL 0x529050). Header: u32 size, version (Driv3r: the 8-byte magic
"THD v3.0", so every later field sits 4 bytes further), u32 cell entries (grid cells + split cells), u32 grid width,
height, f32 min x, y, z, cell size x, z, their inverses, then 7 section offsets (from the chunk start):
- s0: a nibble per cell entry (even entry in the high nibble): 0 no ground (height -500), 0xF one plane, 0xE split in
  two halves (alternately along x and z), 1..13 that many patches (layers, e.g. a road under a bridge).
- s1: a u16 per cell entry: the plane (0xF), the split pair (0xE) or the first patch (1..13).
- s2: split pairs: u32 cell entry of the lower and the upper half.
- s3: patches: u32 offset of the root node in s4, u32 offset of the leaf list in s5.
- s4: 2D BSP nodes of 8 bytes: s16 n0, n1, n2, s8 child above, child below (n0 * fx + n1 * fz + n2 > -1e-5 is above;
  fx, fz = the point's fraction in its cell). A child < 0 is the node -child * 8 bytes after the patch root, else an
  index into the patch's leaf list.
- s5: leaf lists of u16 plane indices.
- s6: planes. Driv3r: 20 bytes, f32 a, b, c, d, u32 surface (low 16 bits; 0x20000 a flag). DPL: 8 bytes, u16 a, u16 c
  (sign bit + sqrt(|value|) * 32767; b = sqrt(1 - a^2 - c^2)), s16 d * 29 (the scale of version 1001), u16 surface.
Height at a point = (d - a * x - c * z) / b, x and z relative to the chunk's min corner. Cells, splits, nodes and leaf
lists only use cell fractions, so a cell copies unchanged between grids of the same cell size: a DPL region's chunk
gets Driv3r's own 20 m cells, on Driv3r's 20 m raster, enough of them to cover the region; only the planes move to the
new min corner.
"""
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, field

DRIV3R_MAGIC = b'THD v3.0'
DPL_VERSION = 1001
DPL_D_SCALE = 29.0      # plane distance scale of version 1001 (0x6D1B40; version 1000 sets 42 for every chunk)
HEADER = 0x4C           # DPL's header: Driv3r's minus the 4 extra magic bytes
SPLIT, ONE_PLANE, NO_GROUND = 0xE, 0xF, 0
NODE, PATCH, D3_PLANE, DPL_PLANE = 8, 8, 20, 8
MAX_D = 32767 / DPL_D_SCALE
WATER = 0x11            # DPL's water surface: a plane of it gives the water height (0x528C4D), not ground
HOLE_DEPTH = -400.0     # Driv3r marks a place without ground with a flat plane at -500
WATER_REGION, WATER_CELLS = 480.0, 64      # GWRD: one chunk per 480 m region, 64 x 64 cells of 7.5 m
WET_SAMPLES = 3         # per side of a terrain cell: the cell gets water when one of these points lies in Driv3r's water
SUNK = 0.5              # ground this far under the water level under a roof keeps its cell dry (DPL acts from 0.6 m deep)
ROOF = 4.0              # metres above the water level: ground that high over sunken ground roofs it (a deck over a road;
                        # Miami's quays stand 2.6 m above its water)
FLOOR_GROUND_POINTS = 2  # a cell with ground at more of its 9 sample points keeps it, whatever the collision covers
MAX_PATCHES = 13        # a cell nibble of 1..13 counts patches (0xE, 0xF mean a split and one plane)


class Driv3rTerrain:
    """One Driv3r GRTD v0 chunk."""

    def __init__(self, data: bytes) -> None:
        if data[4:12] != DRIV3R_MAGIC:
            raise ValueError('not a Driv3r GRTD v0 chunk')

        self.data = data
        self.entries, self.width, self.height = struct.unpack_from('<3I', data, 12)
        self.min_x, _, self.min_z, self.cell, _ = struct.unpack_from('<5f', data, 0x18)
        self.sections = struct.unpack_from('<7I', data, 0x34)

    def nibble(self, entry: int) -> int:
        v = self.data[self.sections[0] + entry // 2]
        return v >> 4 if entry % 2 == 0 else v & 0xF

    def word(self, entry: int) -> int:
        return struct.unpack_from('<H', self.data, self.sections[1] + entry * 2)[0]

    def cell_at(self, x: float, z: float) -> int | None:
        """The grid cell holding the world point, or None outside the chunk."""
        cx, cz = int((x - self.min_x) // self.cell), int((z - self.min_z) // self.cell)
        return cz * self.width + cx if 0 <= cx < self.width and 0 <= cz < self.height else None

    def patch(self, p: int) -> tuple[int, int, tuple[int, ...]]:
        """Patch p: its root node's offset, the last node its children reach and the Driv3r planes of its leaf list."""
        node_offset, leaf_offset = struct.unpack_from('<2I', self.data, self.sections[3] + p * PATCH)
        root = self.sections[4] + node_offset
        pending, seen, last_leaf = [0], {0}, 0

        while pending:
            node = pending.pop()

            for child in struct.unpack_from('<2b', self.data, root + node * NODE + 6):
                if child < 0 and -child not in seen:
                    seen.add(-child)
                    pending.append(-child)
                elif child >= 0:
                    last_leaf = max(last_leaf, child)

        return root, max(seen), struct.unpack_from(f'<{last_leaf + 1}H', self.data, self.sections[5] + leaf_offset)

    def planes_at(self, entry: int, fx: float, fz: float) -> list[int]:
        """The Driv3r planes the cell entry gives at the fraction fx, fz of its cell: one per patch, chosen by its BSP
        (both halves of a split, as dpl_ground reads it)."""
        kind, word = self.nibble(entry), self.word(entry)

        if kind == NO_GROUND:
            return []
        if kind == ONE_PLANE:
            return [word]
        if kind == SPLIT:
            return [k for half in struct.unpack_from('<2I', self.data, self.sections[2] + word * 8)
                    for k in self.planes_at(half, fx, fz)]

        planes = []
        for p in range(word, word + kind):
            root, _, leaves = self.patch(p)
            node = 0
            while True:
                n0, n1, n2, above, below = struct.unpack_from('<3h2b', self.data, root + node * NODE)
                child = above if n0 * fx + n1 * fz + n2 > -1e-5 else below
                if child >= 0:
                    planes.append(leaves[child])
                    break
                node = -child
        return planes

    def sample_heights(self, entry: int, x: float, z: float) -> list[list[float]]:
        """The ground heights (the -500 marker left out) the cell entry gives at each of WET_SAMPLES x WET_SAMPLES points
        of the cell holding x z."""
        x0 = self.min_x + (x - self.min_x) // self.cell * self.cell
        z0 = self.min_z + (z - self.min_z) // self.cell * self.cell
        fractions = [(k + 0.5) / WET_SAMPLES for k in range(WET_SAMPLES)]
        samples = []

        for fx in fractions:
            for fz in fractions:
                heights = []
                for k in self.planes_at(entry, fx, fz):
                    a, b, c, d, _ = struct.unpack_from('<4fI', self.data, self.sections[6] + k * D3_PLANE)
                    h = (d - a * (x0 + fx * self.cell - self.min_x) - c * (z0 + fz * self.cell - self.min_z)) / b
                    if h > HOLE_DEPTH:
                        heights.append(h)
                samples.append(heights)
        return samples


class Driv3rWater:
    """Where a Driv3r city has water: its GWRD chunks (read at Driv3r 0x4508A0) and the GWTR height. A chunk: f32 min x,
    max z, a flag byte (bit (z >= 32) * 2 + (x >= 32): that quadrant has its own cells; high nibble: the value of the
    others), then a nibble per cell, index x * 64 + z, the odd index in the high nibble. A value counts up from the shore
    """

    def __init__(self, chunks: list[bytes], height: float) -> None:
        self.height = height
        corners = [struct.unpack_from('<2f', c, 0) for c in chunks]
        self.min_x, self.max_z = min(x for x, _ in corners), max(z for _, z in corners)
        self.chunks = {(round((x - self.min_x) / WATER_REGION), round((self.max_z - z) / WATER_REGION)): c
                       for c, (x, z) in zip(chunks, corners)}

    def wet(self, x: float, z: float) -> bool:
        fx, fz = (x - self.min_x) / WATER_REGION, (self.max_z - z) / WATER_REGION
        if (chunk := self.chunks.get((math.floor(fx), math.floor(fz)))) is None:
            return False

        ix, iz = int(fx % 1 * WATER_CELLS), int(fz % 1 * WATER_CELLS)
        flags = chunk[8]

        if not flags & 1 << ((iz >> 5) * 2 + (ix >> 5)):
            return flags >> 4 != 0

        u = ix * WATER_CELLS + iz
        return chunk[9 + (u >> 1)] >> (u & 1) * 4 & 0xF != 0


@dataclass
class DplTerrain:
    """A DPL GRTD v2 chunk being built from Driv3r cells."""
    min_x: float
    min_z: float
    width: int
    height: int
    cell: float
    water: float = 0.0      # the city's water height
    wet: bool = False       # the cell being copied lies in Driv3r's water: it gets a water plane besides its ground
    nibbles: list[int] = field(default_factory=list)
    words: list[int] = field(default_factory=list)
    splits: list[tuple[int, int]] = field(default_factory=list)
    patches: list[tuple[int, int]] = field(default_factory=list)
    nodes: bytearray = field(default_factory=bytearray)
    leaves: list[int] = field(default_factory=list)
    planes: list[bytes] = field(default_factory=list)
    plane_index: dict[tuple[int, int], int] = field(default_factory=dict)      # (source id, Driv3r plane) -> plane
    clamped: int = 0
    floored: int = 0

    def __post_init__(self) -> None:
        self.nibbles = [NO_GROUND] * (self.width * self.height)
        self.words = [0] * (self.width * self.height)

    def plane(self, source: Driv3rTerrain, k: int) -> int:
        """The DPL plane for Driv3r plane k of source, moved to this chunk's min corner (added once)."""
        key = (id(source), k)

        if key not in self.plane_index:
            a, b, c, d, surface = struct.unpack_from('<4fI', source.data, source.sections[6] + k * D3_PLANE)

            if b < 0:       # DPL keeps b positive: the same plane with every sign flipped
                a, b, c, d = -a, -b, -c, -d

            # a * x_old + c * z_old = a * (x_new + shift_x) + c * (z_new + shift_z)
            d -= a * (self.min_x - source.min_x) + c * (self.min_z - source.min_z)

            if abs(d) > MAX_D:
                self.clamped += 1
                d = math.copysign(MAX_D, d)

            self.plane_index[key] = len(self.planes)
            self.planes.append(struct.pack('<HHhH', squash(a), squash(c), round(d * DPL_D_SCALE), surface & 0xFFFF))

        return self.plane_index[key]

    def water_plane(self) -> int:
        """The flat plane of surface WATER at the water height (added once). With no ground under it a car or walker
        sinks into it and DPL puts the player back on land; without it DPL keeps a walker out like a wall (0x519D0E)."""
        if (key := (0, -1)) not in self.plane_index:
            self.plane_index[key] = len(self.planes)
            self.planes.append(struct.pack('<HHhH', 0, 0, round(self.water * DPL_D_SCALE), WATER))

        return self.plane_index[key]

    def flat_plane(self, height: float) -> int:
        """A flat ground plane at the height (added once), surface 1 as DPL's common ground."""
        if (key := (1, round(height * DPL_D_SCALE))) not in self.plane_index:
            self.plane_index[key] = len(self.planes)
            self.planes.append(struct.pack('<HHhH', 0, 0, key[1], 1))

        return self.plane_index[key]

    def floor_cell(self, height: float) -> tuple[int, int]:
        """(nibble, word) of a cell with no Driv3r ground that collision boxes cover flat at `height` (a jetty or quay):
        a ground plane there, besides the water plane in a wet cell. Collision alone lets a walker or car drop into the
        sea when the region's boxes are not loaded yet."""
        if not self.wet:
            return ONE_PLANE, self.flat_plane(height)

        first = len(self.patches)
        self.single_patch(self.flat_plane(height))
        self.single_patch(self.water_plane())
        return 2, first

    def single_patch(self, plane: int) -> None:
        """A patch of one plane: one BSP node whose both children are leaf 0 (n = 0 always counts as above)."""
        self.patches.append((len(self.nodes), len(self.leaves) * 2))
        self.nodes += struct.pack('<3h2b', 0, 0, 0, 0, 0)
        self.leaves.append(plane)

    def copy_cell(self, source: Driv3rTerrain, entry: int) -> tuple[int, int]:
        """(nibble, word) of a copy of the source cell entry, with its splits, patches and planes appended. A wet cell
        also gets a patch of the water plane: DPL's query keeps the highest water apart from the ground (0x528C4D), so a
        sea floor sloping under the water level is water too."""
        kind = source.nibble(entry)
        word = source.word(entry)

        if kind == NO_GROUND:
            return (ONE_PLANE, self.water_plane()) if self.wet else (NO_GROUND, 0)

        if kind == ONE_PLANE and self.wet:
            first = len(self.patches)
            self.single_patch(self.plane(source, word))
            self.single_patch(self.water_plane())
            return 2, first

        if kind == ONE_PLANE:
            return ONE_PLANE, self.plane(source, word)

        if kind == SPLIT:
            halves = struct.unpack_from('<2I', source.data, source.sections[2] + word * 8)
            new = []

            for half in halves:
                self.nibbles.append(NO_GROUND)
                self.words.append(0)
                new.append(len(self.nibbles) - 1)
                self.nibbles[new[-1]], self.words[new[-1]] = self.copy_cell(source, half)

            self.splits.append((new[0], new[1]))
            return SPLIT, len(self.splits) - 1

        first = len(self.patches)

        for p in range(word, word + kind):
            self.copy_patch(source, p)

        if self.wet and kind < MAX_PATCHES:
            self.single_patch(self.water_plane())
            kind += 1

        return kind, first

    def copy_patch(self, source: Driv3rTerrain, p: int) -> None:
        """Copies a patch's node block (the root and the nodes after it its children reach) and its leaf list."""
        root, last_node, leaves = source.patch(p)
        self.patches.append((len(self.nodes), len(self.leaves) * 2))
        self.nodes += source.data[root:root + (last_node + 1) * NODE]
        self.leaves += [self.plane(source, k) for k in leaves]

    def join(self) -> bytes:
        """The chunk data."""
        def padded(b: bytes) -> bytes:
            return b + bytes(-len(b) % 4)

        nib = bytes(self.nibbles[i] << 4 | (self.nibbles[i + 1] if i + 1 < len(self.nibbles) else 0)
                    for i in range(0, len(self.nibbles), 2))
        sections = [padded(nib), padded(struct.pack(f'<{len(self.words)}H', *self.words)),
                    b''.join(struct.pack('<2I', *s) for s in self.splits), b''.join(struct.pack('<2I', *p) for p in self.patches),
                    padded(bytes(self.nodes)), padded(struct.pack(f'<{len(self.leaves)}H', *self.leaves)), b''.join(self.planes)]
        offsets, position = [], HEADER

        for s in sections:
            offsets.append(position)
            position += len(s)

        header = struct.pack('<5I7f7I', position, DPL_VERSION, len(self.nibbles), self.width, self.height, self.min_x, 0.0,
                             self.min_z, self.cell, self.cell, 1 / self.cell, 1 / self.cell, *offsets)
        return header + b''.join(sections)


def squash(v: float) -> int:
    """A plane normal's x or z as DPL stores it: sign bit and sqrt(|v|) * 32767."""
    return (0x8000 if v < 0 else 0) | round(math.sqrt(min(abs(v), 1.0)) * 32767)


def region_terrain(sources: list[Driv3rTerrain], x0: float, z0: float, x1: float, z1: float,
                   sea: Driv3rWater | None = None, sunken: Callable[[float, float], bool] | None = None,
                   floors: Callable[[list[tuple[float, float]]], float | None] | None = None) -> DplTerrain:
    """A DPL chunk covering x0..x1, z0..z1 with the Driv3r cells there, and DPL water at the water height in each cell
    inside Driv3r's water (sea) or with a place without ground (a marina between jetties that GWRD leaves dry). A cell
    stays dry where a player would stand under the water there: terrain ground more than SUNK under the water level with
    other ground more than ROOF above it at the same point (a road under a deck), or where sunken(x, z) says a road or
    pavement runs under the water level there. A cell with no ground at all gets a ground plane at floors(its sample
    points) when that gives a height (collision boxes cover it flat)."""
    cell = sources[0].cell
    min_x, min_z = math.floor(x0 / cell) * cell, math.floor(z0 / cell) * cell
    out = DplTerrain(min_x, min_z, math.ceil((x1 - min_x) / cell), math.ceil((z1 - min_z) / cell), cell,
                     sea.height if sea else 0.0)
    offsets = [(k + 0.5) / WET_SAMPLES * cell for k in range(WET_SAMPLES)]

    for row in range(out.height):
        for column in range(out.width):
            x, z = min_x + (column + 0.5) * cell, min_z + (row + 0.5) * cell
            points = [(min_x + column * cell + sx, min_z + row * cell + sz) for sx in offsets for sz in offsets]
            source, entry = next(((s, e) for s in sources if (e := s.cell_at(x, z)) is not None), (None, None))
            heights = source.sample_heights(entry, x, z) if source else [[] for _ in points]
            low, high = out.water - SUNK, out.water + ROOF
            out.wet = (sea is not None and (any(not h for h in heights) or any(sea.wet(px, pz) for px, pz in points))
                       and not any(min(h, default=high) < low and max(h, default=low) > high for h in heights)
                       and not (sunken and any(sunken(px, pz) for px, pz in points)))

            level = floors(points) if floors and source and not heights[len(heights) // 2] and sum(map(bool, heights)) <= FLOOR_GROUND_POINTS else None

            if level is not None:
                out.nibbles[row * out.width + column], out.words[row * out.width + column] = out.floor_cell(level)
                out.floored += 1
            elif source:
                out.nibbles[row * out.width + column], out.words[row * out.width + column] = out.copy_cell(source, entry)
            else:
                # no Driv3r terrain chunk here at all
                if out.wet:
                    out.nibbles[row * out.width + column], out.words[row * out.width + column] = ONE_PLANE, out.water_plane()

    return out
