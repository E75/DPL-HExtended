"""Reader for a Driv3r city (Cities/<City>_DAY_PC.d3c + Cities/<City>_DAY_PC/SuperRegions.pcs), enough to bake placed models
into one world-space mesh. Ported from the Driv3r tools (DRIV3R-PROJECT/threejs/tools: d3chunk, d3world, export_city).

- SuperRegions.pcs: `MDPC` model packs ("pages"), header word [1] = page id. Header words: [2] containers at [3], [4] groups
  at [5], [6] meshes at [7], [10] texture data, [11] PCMP (0 on page 62: its vertices run to the end of the entry),
  [12] index count, [14] index offset, [15] vertex streams. Container 0x188 bytes: box (8 x f32x4 at +0x28), 7 LOD entries
  of 32 bytes at +0xA8 (group offset, ?, group count, ?, ?, kind). Group 88 bytes: mesh offset, 4x4 matrix at +8 (column
  vectors, translation in the 4th element of the first three rows), mesh count u16 at +72. Mesh 56 bytes: primitive type
  (5 strip, 4 list), base vertex, min index, vertex count, start index, primitive count, ..., u16 material at +48, s16
  page of its material table at +50 (the mesh's own page, or 0: the global page's table, about half of the meshes).
- Vertices: stride 48 (position, normal, uv, 16 zero bytes), 1-3 streams of at most 65536 vertices; a mesh's base vertex is
  relative to its stream, the streams follow each other (sizes in the 28-byte descriptors before the index buffer).
- PCMP (header [11]): u32 material count at +8, materials of 24 bytes at +0x38 (layer list offset, count); a layer points at
  a shader (two flag words, slot list offset and count at +16); a slot points at a texture record (flags, hash, offset,
  size): a complete DDS file at header [10] + offset.
- .d3c: `GEBI` chunks in `GESR`/`GEIR` hold instances: 0x40 header (u32 count), records of 0x20 bytes: x y z f32,
  handle u16 (page << 10 | container), flags u16 (0xFFFF = placed), 4 floats: (cos, sin, 0, 101) = yaw, or a unit
  quaternion. Both are applied inverted (yaw sign -1, quaternion conjugate).
- Collision, per `GESR`: `GEPM` v3 (same PDL layout as DPL's, see dplchunk) and `PMLD` "Building Physics Lookup Data":
  u32 GEPM id, u32 row count, u32 row offset, then rows of 20 bytes (model handle, first record, record count, radius, ?).
"""
import io
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from dplchunk import GEPM_HEADER, PDL_ENTRY, PDL_PART, entries, read_tree, split_broadphase

STRIDE = 48
PMLD_ROW = 20
DETAIL_ENTRIES = 4      # LOD entries 0..3 hold the detail meshes
# LOD kind bits 2-16: the near distances an entry draws at (30, 28, 24, 16 for entries 0-3); kinds 0 and 1 are the
# shadow hull and far model (towers' far models sit in the global region)
NEAR_KINDS = 0x1E
SHEET_FLATNESS = 1e-3   # metres off the plane: a flat sheet
SHEET_CLEAR = 0.5       # share of texels under alpha 128: a see-through texture (a cable: 96 %)
# Driv3r layers decals (dirt, trims, signs) 0-15 mm in front of a wall in later meshes of one model and draws them in
# order; DPL sorts by material, so at a distance the layers fought. A layer that close is moved out to this gap
DECAL_GAP = 0.03
DECAL_NEAR = 0.015
DECAL_EDGE = 0.02      # metres: a layer vertex this near the open edge of the faces under it stays
SHADER_TYPE_MASK = 0xFF     # first shader flag word
DECAL_SHADER = 0x03         # flat upward layers (dirt, puddles, road markings) drawn over the surface they lie on
GLOBAL_PAGE = 0             # props, trees and street furniture, placed with their origin on the ground
GROUND_UP = 0.95            # normal y of a flat plate lying on the ground (a tree's pit, a grate)
INSTANCE_SIZE = 0x20
GEBI_HEADER = 0x40
PLACED = 0xFFFF
INTERIOR_ID = 0x100         # interior n's page and GEPM ids: (n + 1) * INTERIOR_ID | Driv3r's (every interior reuses its ids)


@dataclass
class Mesh:
    prim_type: int
    base_vertex: int
    start_index: int
    prim_count: int
    material: int
    material_page: int


@dataclass
class Group:
    matrix: tuple[float, ...]
    meshes: list[Mesh] = field(default_factory=list)


@dataclass
class Instance:
    position: tuple[float, float, float]
    handle: int
    rotation: tuple[float, float, float, float]     # quaternion (x, y, z, w), already inverted as the game applies it
    collision: list[tuple[int, int]] = field(default_factory=list)  # (GEPM id, record) from its region's broadphase


@dataclass
class Material:
    flags: tuple[int, int]      # the first shader's two flag words
    dds: bytes


class Page:
    """One MDPC model pack of SuperRegions.pcs with its vertices as a numpy array (N x 12 floats)."""

    def __init__(self, data: bytes, offset: int, size: int) -> None:
        h = struct.unpack_from('<20I', data, offset)
        self.data, self.id = data, h[1]
        meshes = []

        for i in range(h[6]):
            r = struct.unpack_from('<6I I hh 4I hh I', data, offset + h[7] + i * 56)
            meshes.append(Mesh(r[0], r[1], r[4], r[5], r[13], r[14]))

        groups = []

        for i in range(h[4]):
            o = offset + h[5] + i * 88
            first = (struct.unpack_from('<I', data, o)[0] - h[7]) // 56
            count = struct.unpack_from('<H', data, o + 72)[0]
            groups.append(Group(struct.unpack_from('<16f', data, o + 8), meshes[first:first + count]))

        self.containers = []    # per container: list of (kind, groups) for the 7 LOD entries

        for i in range(h[2]):
            o = offset + h[3] + i * 0x188
            lods = []

            for k in range(7):
                group_off, _, n, _, _, kind = struct.unpack_from('<6I', data, o + 0xA8 + k * 32)
                first = (group_off - h[5]) // 88
                lods.append((kind, groups[first:first + n] if n else []))

            self.containers.append(lods)

        self.pcmp = offset + h[11] if h[11] else 0
        self.texture_data = offset + h[10]
        self.indices_offset = offset + h[14]
        vertex_offset = offset + ((h[14] + h[12] * 2 + 0xFFF) & ~0xFFF)
        vertex_end = offset + h[11] if h[11] else offset + size
        self.vertices = np.frombuffer(data, dtype='<f4', count=(vertex_end - vertex_offset) // STRIDE * 12,
                                      offset=vertex_offset).reshape(-1, 12)
        self.stream_starts, self.stream_bases = self._streams(data, offset, h, meshes)

    @staticmethod
    def _streams(data: bytes, offset: int, h: tuple[int, ...], meshes: list[Mesh]) -> tuple[list[int], list[int]]:
        """(first start index of streams 1.., first vertex of streams 1..): a stream begins where base_vertex falls back."""
        starts = sorted((m.start_index, m.base_vertex) for m in meshes)
        desc = h[14] - 0x10 - h[15] * 28
        sizes = [struct.unpack_from('<7I', data, offset + desc + i * 28)[4] for i in range(h[15])]
        boundaries, previous = [], -1

        for start, base in starts:
            if base < previous:
                boundaries.append(start)

            previous = base

        return boundaries, [int(b) for b in np.cumsum([0] + sizes)[1:len(boundaries) + 1]]

    def stream_base(self, start_index: int) -> int:
        stream = sum(1 for s in self.stream_starts if start_index >= s)
        return self.stream_bases[stream - 1] if stream else 0

    def detail_groups(self, container: int, all_near: bool = False) -> list[Group]:
        """The groups of the first near LOD entry, or with all_near of every near entry: Driv3r's entries add up (a
        light post's span arm and wire sit in entries 1 and 3), each kind a mask of the distances it draws at."""
        near = [groups for kind, groups in self.containers[container][:DETAIL_ENTRIES] if groups and kind & NEAR_KINDS]
        return [g for groups in near for g in groups] if all_near else (near[0] if near else [])

    def material(self, material: int, texture: int = 0) -> Material | None:
        """The first shader's flags and its `texture`-th texture of a material, or None (no PCMP, no layer, no such
        texture or no DDS). Slots come in pairs (texture, 0); a car's tinted paint keeps its colour mask in texture 1."""
        p = self.pcmp

        if not p or material >= struct.unpack_from('<I', self.data, p + 8)[0]:
            return None

        layer_off, layer_count = struct.unpack_from('<2I', self.data, p + 0x38 + material * 24)

        if layer_count == 0:
            return None

        shader = p + struct.unpack_from('<I', self.data, p + layer_off)[0]
        flags = struct.unpack_from('<2I', self.data, shader)
        slot_list, slot_count = struct.unpack_from('<2I', self.data, shader + 16)

        if slot_count <= 2 * texture:
            return None

        record = p + struct.unpack_from('<I', self.data, p + slot_list + 8 * texture)[0]
        _, _, offset, size = struct.unpack_from('<4I', self.data, record)
        o = self.texture_data + offset
        return Material(flags, self.data[o:o + size]) if self.data[o:o + 4] == b'DDS ' else None

    def mesh_triangles(self, mesh: Mesh) -> np.ndarray:
        if mesh.prim_type == 4:
            raw = np.frombuffer(self.data, dtype='<u2', count=mesh.prim_count * 3, offset=self.indices_offset + mesh.start_index * 2)
            tri = raw.astype(np.int64).reshape(-1, 3)
        else:
            raw = np.frombuffer(self.data, dtype='<u2', count=mesh.prim_count + 2, offset=self.indices_offset + mesh.start_index * 2)
            tri = strip_triangles(raw.astype(np.int64))

        tri = tri + mesh.base_vertex + self.stream_base(mesh.start_index)
        return tri[(tri < len(self.vertices)).all(axis=1)]


def strip_triangles(raw: np.ndarray) -> np.ndarray:
    """Triangles (Mx3) of a triangle strip; degenerate stitching triangles dropped, odd ones flipped to keep the winding."""
    a, b, c = raw[:-2], raw[1:-1], raw[2:]
    odd = (np.arange(len(a)) % 2) == 1
    tri = np.stack([np.where(odd, b, a), np.where(odd, a, b), c], axis=1)
    return tri[(a != b) & (b != c) & (a != c)]


def read_pages(path: Path) -> dict[int, Page]:
    data = path.read_bytes()
    return {p.id: p for p in (Page(data, e.offset, e.size) for e in entries(data, 0) if e.tag == 'MDPC')}


def interior_id(interior: int | None, own: int) -> int:
    """The id of Driv3r's page or GEPM id own inside this interior (None: outside, own). Id 0 stays the global one: an
    interior's props are global models with their records in GEPM 0."""
    return own if interior is None or own == GLOBAL_PAGE else (interior + 1) * INTERIOR_ID | own


def interior_handle(interior: int | None, handle: int) -> int:
    return interior_id(interior, handle >> 10) << 10 | handle & 1023


def read_interior_pages(path: Path) -> dict[int, Page]:
    """Interiors.pcs: one run of entries per page id (59, then 30), one entry per interior in IRCT order, an empty PCSL
    interior k % count. Each pack gets its interior's page id, and so do its meshes on its own material table."""
    data, result = path.read_bytes(), {}
    found = entries(data, 0)
    pages = {k: Page(data, e.offset, e.size) for k, e in enumerate(found) if e.tag == 'MDPC'}
    count = len(found) // len({p.id for p in pages.values()}) if pages else 0

    for k, page in pages.items():
        own = page.id
        page.id = interior_id(k % count, own)

        for mesh in {id(m): m for lods in page.containers for _, groups in lods for g in groups for m in g.meshes}.values():
            if mesh.material_page == own:
                mesh.material_page = page.id

        result[page.id] = page

    return result


def read_instances(path: Path, quaternion_as_stored: bool = False, interiors: bool = False) -> list[Instance]:
    """Placed instances of every region, each with its collision records from the region's broadphase; with interiors
    also those of every IRCT, on their interior's ids (they stand in world space behind their doors). The super region
    level (only Driv3r's Global Region has instances there) is left out: its large low-poly pieces stand in for whole
    districts from far away and would cover the streets."""
    result = []
    interior = -1

    for top in read_tree(path.read_bytes(), 0, 1).chunks:
        chunks = {c.tag: bytes(c.data) for c in (top.tree.chunks if top.tree else []) if c.tag in (b'GEBI', b'JNR!')}
        interior += top.tag == b'IRCT'
        inside = interior if top.tag == b'IRCT' else None

        if not (top.tag == b'GEIR' or interiors and top.tag == b'IRCT') or b'GEBI' not in chunks:
            continue

        gebi = chunks[b'GEBI']
        refs: dict[int, list[tuple[int, int]]] = {}

        if b'JNR!' in chunks:
            for objects in split_broadphase(chunks[b'JNR!']).objects:
                for o in objects:
                    refs.setdefault(o >> 16, []).append((interior_id(inside, o >> 12 & 0xF), o & 0xFFF))

        for i in range(min(struct.unpack_from('<I', gebi, 0)[0], (len(gebi) - GEBI_HEADER) // INSTANCE_SIZE)):
            o = GEBI_HEADER + i * INSTANCE_SIZE
            x, y, z, handle, flags = struct.unpack_from('<3fHH', gebi, o)

            if flags == PLACED:
                result.append(Instance((x, y, z), interior_handle(inside, handle), rotation(struct.unpack_from('<4f', gebi, o + 16), quaternion_as_stored),
                                       refs.get(i, [])))

    return result


def rotation(f: tuple[float, ...], quaternion_as_stored: bool = False) -> tuple[float, float, float, float]:
    """Quaternion of an instance's last 16 bytes: a yaw (cos, sin, 0, 101) or a unit quaternion, both inverted; with
    across their roads: of its long models next to a road, 214 follow it as stored, 19 inverted (D3-SOLVED-CASES C33).
    Miami keeps the inversion until it is checked there."""
    if abs(sum(v * v for v in f) - 1.0) < 0.02:
        return (f[0], f[1], f[2], f[3]) if quaternion_as_stored else (-f[0], -f[1], -f[2], f[3])

    a = -math.atan2(f[1], f[0])
    return (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))


def quaternion_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass
class Collision:
    gepm: dict[int, bytes]                          # GEPM chunk per id
    lookup: dict[int, list[tuple[int, int]]]        # PMLD: model handle -> (GEPM id, record)

    def records(self, instance: Instance) -> list[list[bytes]]:
        """The instance's collision records, each a list of 96-byte PDL parts in model space (in a record of several parts,
        part 0 is the box around the others): its broadphase records, else its model's PMLD records (instances outside a
        region broadphase, and models without one)."""
        result = []

        for gepm_id, record in instance.collision or self.lookup.get(instance.handle, []):
            gepm = self.gepm[gepm_id]
            e = GEPM_HEADER + struct.unpack_from('<I', gepm, GEPM_HEADER + 8)[0] + record * PDL_ENTRY
            n, off = struct.unpack_from('<2I', gepm, e)
            result.append([gepm[e + off + k * PDL_PART:e + off + (k + 1) * PDL_PART] for k in range(n)])

        return result


def read_collision(path: Path, interiors: bool = False) -> Collision:
    """The super regions' collision; with interiors also every IRCT's, on its interior's ids (read_instances). An
    interior's lookup rows for global models leave the global ones in place."""
    collision = Collision({}, {})
    interior = -1

    for top in read_tree(path.read_bytes(), 0, 1).chunks:
        interior += top.tag == b'IRCT'
        inside = interior if top.tag == b'IRCT' else None

        for c in (top.tree.chunks if top.tree and (top.tag == b'GESR' or interiors and top.tag == b'IRCT') else []):
            if c.tag == b'GEPM':
                collision.gepm[interior_id(inside, struct.unpack_from('<I', c.data, 0)[0])] = bytes(c.data)
            elif c.tag == b'PMLD':
                gepm_id, rows, rows_off = struct.unpack_from('<3I', c.data, 0)

                for r in range(rows):
                    handle, first, count = struct.unpack_from('<3I', c.data, rows_off + r * PMLD_ROW)
                    handle = interior_handle(inside, handle)
                    if inside is None or handle not in collision.lookup:
                        collision.lookup[handle] = [(interior_id(inside, gepm_id), k) for k in range(first, first + count)]

    return collision


def place_parts(parts: list[bytes], instance: Instance) -> list[bytearray]:
    """The parts in world space: their rotation rows and centre turned and moved like the instance's meshes."""
    turn = quaternion_matrix(instance.rotation)
    placed = []

    for part in parts:
        p = bytearray(part)
        rows = np.array(struct.unpack_from('<12f', p, 16)).reshape(3, 4)[:, :3] @ turn.T
        centre = turn @ np.array(struct.unpack_from('<3f', p, 64)) + np.array(instance.position)

        for k in range(3):
            struct.pack_into('<3f', p, 16 + k * 16, *rows[k])

        struct.pack_into('<3f', p, 64, *centre)
        placed.append(p)

    return placed


@dataclass
class Baked:
    vertices: np.ndarray        # N x 8: world position, normal, uv
    triangles: np.ndarray       # M x 3 into vertices
    materials: np.ndarray       # M: (page << 16 | material) per triangle


def bake_instance(page: Page, container: int, instance: Instance, all_near: bool = False) -> Baked | None:
    """The detail meshes of one placed container in world space: those of its first near LOD entry, or with all_near
    those of every near entry (Page.detail_groups)."""
    parts, tris, mats, ground = [], [], [], []
    offset = 0
    turn = quaternion_matrix(instance.rotation)

    for group in page.detail_groups(container, all_near):
        matrix = np.array(group.matrix, dtype=np.float64).reshape(4, 4)

        for mesh in group.meshes:
            tri = page.mesh_triangles(mesh)

            if not len(tri):
                continue

            used, inverse = np.unique(tri.reshape(-1), return_inverse=True)
            v = page.vertices[used, :8].astype(np.float64)
            v[:, 0:3] = v[:, 0:3] @ matrix[:3, :3].T + matrix[:3, 3]
            v[:, 3:6] = v[:, 3:6] @ matrix[:3, :3].T
            on_ground = page.id == GLOBAL_PAGE and bool((np.abs(v[:, 1]) < DECAL_NEAR).all() and (v[:, 4] > GROUND_UP).all())
            v[:, 0:3] = v[:, 0:3] @ turn.T + np.array(instance.position)
            v[:, 3:6] = v[:, 3:6] @ turn.T
            faces = inverse.reshape(-1, 3)

            # a flat see-through sheet (a light post's cable) shows from both sides in Driv3r; DPL culls back faces
            if all_near and is_sheet(page, mesh, v):
                back = v.copy()
                back[:, 3:6] *= -1
                faces = np.concatenate([faces, faces[:, ::-1] + len(v)])
                v = np.concatenate([v, back])

            parts.append(v)
            ground.append(np.full(len(v), on_ground))
            tris.append(faces + offset)
            mats.append(np.full(len(tris[-1]), mesh.material_page << 16 | mesh.material, dtype=np.int64))
            offset += len(v)

    if not parts:
        return None

    baked = Baked(np.concatenate(parts), np.concatenate(tris), np.concatenate(mats))
    key = (page.id, container, all_near)

    if key not in _decal_lifts:
        _decal_lifts[key] = decal_lifts(baked, np.repeat(np.arange(len(tris)), [len(t) for t in tris])) @ turn

    baked.vertices[:, 0:3] += _decal_lifts[key] @ turn.T

    # a model of decal layers only (dirt, puddles, markings) lies on another model's surface, and a prop's ground plate
    # (a tree pit) on the pavement under it: decal_lifts sees neither, so both fought the road or pavement
    if all(material is not None and material.flags[0] & SHADER_TYPE_MASK == DECAL_SHADER
           for group in page.detail_groups(container, all_near) for mesh in group.meshes
           for material in [page.material(mesh.material) if mesh.material_page in (0, page.id) else None]):
        baked.vertices[:, 0:3] += DECAL_GAP * turn[:, 1]
    else:
        baked.vertices[np.concatenate(ground), 0:3] += DECAL_GAP * turn[:, 1]

    return baked


def decal_lifts(baked: Baked, meshes: np.ndarray) -> np.ndarray:
    """Per vertex the move that puts each decal layer DECAL_GAP in front of the face it lies on: a triangle that faces
    the same way as an earlier mesh's triangle, overlaps it and lies at most DECAL_NEAR in front (meshes ascend)."""
    p = baked.vertices[:, 0:3][baked.triangles]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = np.linalg.norm(cross, axis=1)
    lifts = np.zeros((len(baked.vertices), 3))
    big = np.flatnonzero(area > 1e-4)

    if len(np.unique(meshes[big])) < 2:
        return lifts

    normal = cross[big] / area[big, None]
    depth = (normal * p[big].mean(axis=1)).sum(axis=1)
    lo, hi = p[big].min(axis=1), p[big].max(axis=1)
    mesh = meshes[big]
    push = np.zeros(len(big))
    covered = np.zeros(len(big), dtype=bool)

    # in mesh order, so a decal on a decal stacks DECAL_GAP above the moved layer under it
    # limit: n x n per model (about a minute for all of Miami), upgrade: bucket by plane when models grow
    for layer in np.unique(mesh)[1:]:
        for start in range(np.searchsorted(mesh, layer), np.searchsorted(mesh, layer, 'right'), 512):
            rows = slice(start, min(start + 512, np.searchsorted(mesh, layer, 'right')))
            below = mesh < layer
            gap = depth[rows, None] - depth[None, below]
            ext = np.minimum(hi[rows, None], hi[None, below]) - np.maximum(lo[rows, None], lo[None, below])
            under = ((normal[rows] @ normal[below].T > 0.995) & (gap > -0.001) & (gap < DECAL_NEAR) &
                     (ext > -0.01).all(axis=-1) & (np.sort(ext, axis=-1)[..., 1] > 0.05))
            push[rows] = np.where(under, push[None, below] + DECAL_GAP - gap, 0.0).max(axis=1, initial=0.0)
            covered[np.flatnonzero(below)[under.any(axis=0)]] = True

    # a vertex shared by two layers takes the larger move: later assignments win, so the largest goes last
    order = np.argsort(push)
    order = order[push[order] > 0]
    moves = np.repeat(push[order, None] * normal[order], 3, axis=0)
    lifts[baked.triangles[big[order]].reshape(-1)] = moves

    # vertices at one place move together, or a layer's edge opened a crack (a kerb, the pavement next to a road);
    # a place that holds a vertex of a face under a layer stays, so that face keeps meeting its neighbours
    _, place = np.unique(np.round(baked.vertices[:, 0:3] * 1000).astype(np.int64), axis=0, return_inverse=True)
    place = place.reshape(-1)
    anchored = np.zeros(place.max() + 1, dtype=bool)
    anchored[place[baked.triangles[big[covered]].reshape(-1)]] = True
    anchored[place[on_open_edge(baked, big[push == 0], np.flatnonzero(np.linalg.norm(lifts, axis=1) > 0))]] = True
    by_size = np.argsort(np.linalg.norm(lifts, axis=1))
    largest = np.zeros((len(anchored), 3))
    largest[place[by_size]] = lifts[by_size]
    largest[anchored] = 0.0
    lifts = largest[place]
    return lifts


def on_open_edge(baked: Baked, faces: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """The candidate vertices within DECAL_EDGE of an open edge of the faces (an edge only one of them uses): where the
    faces under a layer end, often at the model's edge against the next model, the layer must stay or a step opened."""
    corners = np.round(baked.vertices[:, 0:3] * 1000).astype(np.int64)
    _, key = np.unique(corners, axis=0, return_inverse=True)
    key = key.reshape(-1)[baked.triangles[faces]]
    edges = np.sort(np.concatenate([key[:, [0, 1]], key[:, [1, 2]], key[:, [2, 0]]]), axis=1)
    unique, first, count = np.unique(edges, axis=0, return_index=True, return_counts=True)
    ends = np.concatenate([baked.triangles[faces][:, [0, 1]], baked.triangles[faces][:, [1, 2]],
                           baked.triangles[faces][:, [2, 0]]])[first[count == 1]]

    if not len(ends) or not len(candidates):
        return np.zeros(0, dtype=np.int64)

    a, b = baked.vertices[ends[:, 0], 0:3], baked.vertices[ends[:, 1], 0:3]
    span = b - a
    length = np.maximum((span * span).sum(axis=1), 1e-12)
    near = np.zeros(len(candidates), dtype=bool)

    for start in range(0, len(candidates), 256):
        v = baked.vertices[candidates[start:start + 256], 0:3]
        t = np.clip(((v[:, None] - a[None]) * span[None]).sum(axis=2) / length[None], 0.0, 1.0)
        gap = np.linalg.norm(v[:, None] - (a[None] + t[..., None] * span[None]), axis=2)
        near[start:start + 256] = (gap < DECAL_EDGE).any(axis=1)

    return candidates[near]


def is_sheet(page: Page, mesh: Mesh, v: np.ndarray) -> bool:
    """Whether a mesh is one flat plane with a see-through texture (DXT3/5, mostly transparent texels)."""
    spread = v[:, 0:3] - v[:, 0:3].mean(axis=0)

    if len(v) < 3 or np.linalg.svd(spread, compute_uv=False)[-1] > SHEET_FLATNESS:
        return False

    key = (page.id, mesh.material)

    if key not in _clear_textures:
        material = page.material(mesh.material) if mesh.material_page in (0, page.id) else None
        _clear_textures[key] = material is not None and see_through(material)

    return _clear_textures[key]


def see_through(material: Material) -> bool:
    """Whether the material's texture is DXT3/5 with mostly transparent texels."""
    if material.dds[84:88] not in (b'DXT3', b'DXT5'):
        return False

    alpha = np.asarray(Image.open(io.BytesIO(material.dds)).convert('RGBA'))[..., 3]
    return float((alpha < 128).mean()) >= SHEET_CLEAR


_clear_textures: dict[tuple[int, int], bool] = {}     # (page, material) -> see-through texture
_decal_lifts: dict[tuple[int, int, bool], np.ndarray] = {}     # (page, container, all_near) -> lifts in the model frame


def bake_area(pages: dict[int, Page], instances: list[Instance], centre: tuple[float, float], half: float,
              max_vertices: int) -> tuple[Baked, list[Instance]]:
    """Placed models whose position lies in the square around centre (x, z), nearest first, until max_vertices.
    Returns (the merged world-space mesh, the instances used)."""
    inside = [i for i in instances if abs(i.position[0] - centre[0]) <= half and abs(i.position[2] - centre[1]) <= half
              and (i.handle >> 10) in pages and (i.handle & 1023) < len(pages[i.handle >> 10].containers)]
    inside.sort(key=lambda i: (i.position[0] - centre[0]) ** 2 + (i.position[2] - centre[1]) ** 2)
    parts: list[Baked] = []
    used: list[Instance] = []
    count = 0

    for instance in inside:
        baked = bake_instance(pages[instance.handle >> 10], instance.handle & 1023, instance)

        if baked is None:
            continue

        if count + len(baked.vertices) > max_vertices:
            break

        parts.append(baked)
        count += len(baked.vertices)
        used.append(instance)

    offsets = np.cumsum([0] + [len(b.vertices) for b in parts])[:-1]
    return Baked(np.concatenate([b.vertices for b in parts]), np.concatenate([b.triangles + o for b, o in zip(parts, offsets)]),
                 np.concatenate([b.materials for b in parts])), used
