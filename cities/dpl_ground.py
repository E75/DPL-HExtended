"""What a vertical line meets in a DPL city file: the terrain planes of each region's GRTD (v2, layout in d3_terrain) and
the collision boxes its broadphases (JNR!) place, turned and moved like the region instance that holds them (the GEPM
records, layout in dplchunk). For checking where cars and walkers stand: road lanes, conns and parking strips.

Simplifications: a split cell gives the planes of both halves and a cylinder or sphere counts as its bounding box, so a
line may meet a little more than the game's; a BSP edge can drop a patch's plane on a single point.
"""
import struct
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import d3_terrain
import dplchunk as d

BIN = 16.0                      # metres per side of the grid cells that index the boxes
CAR_BAND = (0.5, 2.5)           # metres above a road point that a car body fills
SPLIT, ONE_PLANE, NO_GROUND = d3_terrain.SPLIT, d3_terrain.ONE_PLANE, d3_terrain.NO_GROUND
CYLINDER, SPHERE = 1, 2


def unsquash(v: int) -> float:
    """A plane normal's x or z as stored (d3_terrain.squash) back to a float."""
    return (-1 if v & 0x8000 else 1) * ((v & 0x7FFF) / 32767) ** 2


class Terrain:
    """One DPL GRTD v2 chunk."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.width, self.height = struct.unpack_from('<2I', data, 12)
        self.min_x, _, self.min_z, self.cell = struct.unpack_from('<4f', data, 20)
        self.sections = struct.unpack_from('<7I', data, 0x30)

    def _planes(self, entry: int, fx: float, fz: float) -> list[int]:
        v = self.data[self.sections[0] + entry // 2]
        kind = v >> 4 if entry % 2 == 0 else v & 0xF
        word = struct.unpack_from('<H', self.data, self.sections[1] + entry * 2)[0]

        if kind == NO_GROUND:
            return []
        if kind == ONE_PLANE:
            return [word]
        if kind == SPLIT:
            halves = struct.unpack_from('<2I', self.data, self.sections[2] + word * 8)
            return [k for half in halves for k in self._planes(half, fx, fz)]

        planes = []
        for patch in range(word, word + kind):
            node_offset, leaf_offset = struct.unpack_from('<2I', self.data, self.sections[3] + patch * 8)
            root, node = self.sections[4] + node_offset, 0
            while True:
                n0, n1, n2, above, below = struct.unpack_from('<3h2b', self.data, root + node * 8)
                child = above if n0 * fx + n1 * fz + n2 > -1e-5 else below
                if child >= 0:
                    planes.append(struct.unpack_from('<H', self.data, self.sections[5] + leaf_offset + child * 2)[0])
                    break
                node = -child
        return planes

    def heights(self, x: float, z: float) -> list[float]:
        """Heights of the terrain layers at x z (none outside the chunk)."""
        cx, cz = (x - self.min_x) / self.cell, (z - self.min_z) / self.cell
        if not (0 <= cx < self.width and 0 <= cz < self.height):
            return []

        result = []
        for k in self._planes(int(cz) * self.width + int(cx), cx % 1, cz % 1):
            a, c, dd, _ = struct.unpack_from('<HHhH', self.data, self.sections[6] + k * 8)
            a, c = unsquash(a), unsquash(c)
            b = np.sqrt(max(1 - a * a - c * c, 1e-9))
            result.append((dd / d3_terrain.DPL_D_SCALE - a * (x - self.min_x) - c * (z - self.min_z)) / b)
        return result


@dataclass
class Ground:
    terrains: list[Terrain]
    centres: np.ndarray     # per box: centre x y z
    axes: np.ndarray        # per box: 3 unit axes (rows) x y z
    halves: np.ndarray      # per box: half extents along its axes
    bins: dict[tuple[int, int], list[int]]

    def boxes(self, x: float, z: float) -> list[tuple[float, float]]:
        """(bottom, top) of every collision box the vertical line at x z passes through."""
        found = self.bins.get((int(np.floor(x / BIN)), int(np.floor(z / BIN))))
        if not found:
            return []

        ks = np.array(found)
        c, r, h = self.centres[ks], self.axes[ks], self.halves[ks]
        # along axis k the line is at base + r_k.y * (y - c.y); it is inside while that stays within +-h_k
        base = r[:, :, 0] * (x - c[:, None, 0]) + r[:, :, 2] * (z - c[:, None, 2])
        ry = r[:, :, 1]
        level = np.abs(ry) < 1e-6
        inside = np.abs(base) <= h
        with np.errstate(divide='ignore', invalid='ignore'):
            t0, t1 = (-h - base) / ry, (h - base) / ry
        lo = np.where(level, np.where(inside, -np.inf, np.inf), np.minimum(t0, t1)).max(axis=1) + c[:, 1]
        hi = np.where(level, np.where(inside, np.inf, -np.inf), np.maximum(t0, t1)).min(axis=1) + c[:, 1]
        return [(float(a), float(b)) for a, b in zip(lo, hi) if a <= b]

    def surfaces(self, x: float, z: float) -> list[float]:
        """Heights of everything one can stand on at x z: terrain layers and box tops."""
        return [top for _, top in self.boxes(x, z)] + [h for t in self.terrains for h in t.heights(x, z)]

    def blocked(self, x: float, y: float, z: float, band: tuple[float, float]) -> bool:
        """Whether a collision box reaches into the height band above y at x z."""
        return any(lo < y + band[1] and hi > y + band[0] for lo, hi in self.boxes(x, z))


def read_ground(tree: d.ChunkTree) -> Ground:
    """The terrain and the placed collision boxes of a city file (in a record of several parts part 0 only bounds the
    others and is skipped)."""
    gepms = {struct.unpack_from('<I', c.data, 0)[0]: bytes(c.data) for top in tree.chunks
             if top.tree and top.tag in (b'GEGR', b'GESR') for c in top.tree.chunks if c.tag == b'GEPM'}
    terrains, centres, axes, halves = [], [], [], []

    for top in tree.chunks:
        if not (top.tree and top.tag == b'GEIR'):
            continue
        chunks = {c.tag: bytes(c.data) for c in top.tree.chunks}
        if b'GRTD' in chunks:
            terrains.append(Terrain(chunks[b'GRTD']))
        if b'JNR!' not in chunks:
            continue
        instances = d.region_instances(chunks[b'GEBI'])

        for objects in d.split_broadphase(chunks[b'JNR!']).objects:
            for o in objects:
                index, gepm, record = o >> 16, gepms.get(o >> 11 & 0x1F), o & 0x7FF
                if gepm is None or index >= len(instances) or record >= struct.unpack_from('<I', gepm, d.GEPM_HEADER)[0]:
                    continue
                entry = d.GEPM_HEADER + struct.unpack_from('<I', gepm, d.GEPM_HEADER + 8)[0] + record * d.PDL_ENTRY
                count, offset = struct.unpack_from('<2I', gepm, entry)
                i = instances[index]

                for k in range(1 if count > 1 else 0, count):
                    part = gepm[entry + offset + k * d.PDL_PART:entry + offset + (k + 1) * d.PDL_PART]
                    h = np.array(struct.unpack_from('<3f', part, 0))
                    shape = struct.unpack_from('<I', part, 80)[0] & 0xFF
                    h = np.array([h[0], h[1], h[0]]) if shape == CYLINDER else np.full(3, h[0]) if shape == SPHERE else h
                    rows = np.array(struct.unpack_from('<12f', part, 16)).reshape(3, 4)[:, :3]
                    x, y, z = struct.unpack_from('<3f', part, 64)
                    axes.append(np.stack([rows[:, 0] * i.cos - rows[:, 2] * i.sin, rows[:, 1],
                                          rows[:, 0] * i.sin + rows[:, 2] * i.cos], 1))
                    centres.append((i.x + x * i.cos - z * i.sin, i.y + y, i.z + x * i.sin + z * i.cos))
                    halves.append(h)

    centres_a, axes_a, halves_a = np.array(centres), np.array(axes), np.array(halves)
    reach = np.abs(axes_a * halves_a[:, :, None]).sum(axis=1)       # half size of each box along world x y z
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, (c, w) in enumerate(zip(centres_a, reach)):
        for bx in range(int(np.floor((c[0] - w[0]) / BIN)), int(np.floor((c[0] + w[0]) / BIN)) + 1):
            for bz in range(int(np.floor((c[2] - w[2]) / BIN)), int(np.floor((c[2] + w[2]) / BIN)) + 1):
                bins[bx, bz].append(k)

    return Ground(terrains, centres_a, axes_a, halves_a, bins)
