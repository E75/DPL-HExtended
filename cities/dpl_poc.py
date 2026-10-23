"""Puts a square of a Driv3r city into one container of a COPY of a DPL city file (geometry and textures).

usage: dpl_poc.py <copy.d4c> <driv3r game dir> [--city Miami] [--centre -384 128] [--half 128]
                  [--page 12] [--container 171] [--preview out.glb]

The placed Driv3r models in the square (nearest the centre first, at most 65536 vertices) are baked into world space,
moved so the square's centre and ground sit at the container's origin, and written as a new vertex stream with one mesh
per Driv3r material. Every Driv3r material becomes a DPL material in the pack's PCMP with the Driv3r DDS file and one of
DPL's own flag sets (a material DPL cannot take keeps the container's first DPL material). The container keeps
its instance (position, yaw); its scale, stream, box and first used LOD entry point at the new geometry, its other LOD
entries are emptied. The models' Driv3r collision goes into the super region's GEPM and the region's broadphase;
--show-collision draws those collision boxes instead of the buildings, to check in game where they lie. The Driv3r roads
of the square join DPL's road network (AIEX) as a network of their own, so DPL's traffic drives on them.
"""
import argparse
import hashlib
import io
import math
import mmap
import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import d3_ai
import d3city
import dpl_ai
import dplchunk as d
from glb import GlbBuilder

SCALE = 1.0 / 256       # raw vertex position * SCALE = metres, the range DPL itself uses for 128 m containers
MAX_RAW_POSITION = 32767        # DPL's own raw vertex positions stay in int16 range: larger models get a larger scale
GROUND_PERCENTILE = 2   # the square's ground height: this percentile of its vertex heights
# Texture record and substance words that are the same in (nearly) every DPL record; their meaning is unknown.
TEXTURE_WORDS = (0xF00, 0x40001)
TEXTURE_FORMAT_LOW = 0x29
SUBSTANCE_LAST = 0x08000000
# DPL's three most used substance flag sets: opaque DXT1, DXT3/5, alpha-tested DXT3/5. Driv3r's own flags must not be
# copied: a set DPL has no shader for crashes the container load (0x5EE51E).
OPAQUE_FLAGS = (0, 0x4)
ALPHA_FLAGS = (0, 0x4004)
ALPHA_TEST_FLAGS = (0x201, 0x4004)
DRIV3R_ALPHA_TEST = 0x200       # in Driv3r's second shader flag word
# DPL's own decal shader (NYC's dirt and kerb-edge layers: 0x201 or 0x2000201 with it, alpha about 47); Driv3r's flags
# swapped, as Driv3r's type-3 layers carry the same textures
DECAL_FLAGS = 0x4003
# Driv3r's environment-map shader: a second, empty texture slot the engine fills, and the alpha is the reflection mask
DRIV3R_ENV_MAP = 0x2000000
# Driv3r's foliage shader (first flag word, low byte): leaf cards of trees and bushes
FOLIAGE_SHADER = 0x1B
# Driv3r's helper shader (first flag word): in Miami one flat-red material (0x801) on facade-sized sheets 2-4 cm in
# front of the walls; drawn as a texture it showed as flickering red patches on buildings
DRIV3R_HIDDEN = 0x800
# Vertex colour (last 4 bytes of the vertex tail) of converted models. The old per-stream copy of one NYC vertex gave
# 63, 121, 128 or 255 by stream (a third of Miami's vertices at 25-50 %), so roads came out black or white. DPL doubles
# the vertex colour (0x80 is 1.0): Driv3r's lighter textures at 0xFF clipped walls and kerb ramps to pure white in game
# (2026-10-23), at 0x80 they show as in Driv3r; 0xA0 and 0xC0 still clipped.
VERTEX_COLOUR = 0x80
# LOD entry kind (record +0xA4 + entry * 24 + 16) is a mask of distance bands; the one entry of a Driv3r model takes
# them all, or far buildings popped in and out with the camera
ALL_DISTANCES = 0x7
MAX_RECORD_PARTS = 15   # the most parts DPL's own collision records have, the bounding part 0 included
# Surface ids differ per game (Driv3r's 0x1b, 0x06 do not occur on DPL buildings): every part gets DPL's most common one.
DPL_SURFACE = 0x01
BOX_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
# Driv3r leaves slots of about 20 cm between a deck's box and the terrain it meets; a bike's wheel dropped into them.
# Level boxes this wide (both half widths) grow by the overlap on every side; walls, kerbs and posts stay as they are
DECK_HALF_WIDTH = 2.0
DECK_LEVEL = 0.9        # the up axis's y component
DECK_OVERLAP = 0.3
DECK_HALF_HEIGHT = 1.5  # thicker is a building or a pier: its walls must stay where they are drawn
# a bus tunnelled through a 0.4 m bridge deck into the 0.6 m gap above the box under it; thinner decks grow downward
DECK_MIN_HALF_HEIGHT = 0.6
ROAD_REACH = 80.0       # a road touching the square is taken when it ends within this distance outside it


def strip_of(triangles: np.ndarray) -> list[int]:
    """One triangle strip drawing these triangles (same winding, in order). A triangle that shares the strip's last edge
    with the right winding adds one index (Driv3r's own strips decode to such runs); any other starts on an even position,
    joined by degenerates."""
    out: list[int] = []

    for a, b, c in triangles.tolist():
        if out:
            # the strip's next triangle is (p, q, v): its last two indices, swapped on an odd position
            p, q = (out[-2], out[-1]) if len(out) % 2 == 0 else (out[-1], out[-2])
            v = next((r[2] for r in ((a, b, c), (b, c, a), (c, a, b)) if r[0] == p and r[1] == q), None)

            if v is not None:
                out.append(v)
                continue

            out.append(out[-1])

            if len(out) % 2 == 0:       # the second `a` below must land on an even position
                out.append(out[-1])

            out += [a, a, b, c]
        else:
            out += [a, b, c]

    return out


def strip_of_greedy(triangles: np.ndarray) -> tuple[list[int], np.ndarray]:
    """(strips joined by degenerates, the triangles in the order they are drawn). Unlike strip_of the triangles are
    reordered: each strip starts at a triangle with few free neighbours and walks on to the free neighbour that has the
    fewest, which gives about 2 indices a vertex instead of 4 on a decimated car body."""
    tris = triangles.tolist()
    across: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for n, (a, b, c) in enumerate(tris):
        for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
            across.setdefault((p, q), []).append((n, r))

    free = [True] * len(tris)

    def neighbours(n: int) -> int:
        a, b, c = tris[n]
        return sum(1 for p, q in ((b, a), (c, b), (a, c)) for m, _ in across.get((p, q), ()) if free[m] and m != n)

    out: list[int] = []
    order: list[int] = []

    for n in sorted(range(len(tris)), key=neighbours):
        if not free[n]:
            continue

        free[n] = False
        order.append(n)
        a, b, c = tris[n]

        if out:
            out.append(out[-1])

            if len(out) % 2 == 0:
                out.append(out[-1])

            out.append(a)

        out += [a, b, c]

        while True:
            p, q = (out[-2], out[-1]) if len(out) % 2 == 0 else (out[-1], out[-2])
            options = [(m, r) for m, r in across.get((p, q), ()) if free[m]]

            if not options:
                break

            m, r = min(options, key=lambda o: neighbours(o[0]))
            free[m] = False
            order.append(m)
            out.append(r)

    return out, triangles[order]


def check_strip(strip: list[int], triangles: np.ndarray) -> None:
    """The strip decodes to the triangles, in order and winding (a triangle may start at another corner)."""
    def canonical(t: np.ndarray) -> np.ndarray:
        rolled = np.stack([np.roll(t, -k, axis=1) for k in range(3)], axis=1)      # n x 3 rotations x 3
        return rolled[np.arange(len(t)), rolled[:, :, 0].argmin(axis=1)]

    decoded = d3city.strip_triangles(np.array(strip, dtype=np.int64))

    if decoded.shape != triangles.shape or not (canonical(decoded) == canonical(triangles)).all():
        raise SystemExit(f'strip check failed: {len(decoded)} triangles decoded, {len(triangles)} expected')


def bake(args: argparse.Namespace, pages: dict[int, d3city.Page]) -> tuple[d3city.Baked, list[d3city.Instance], np.ndarray]:
    """(the area's mesh around the container origin, the placed models in it, the Driv3r point that became the origin)."""
    instances = d3city.read_instances(args.driv3r / 'Cities' / f'{args.city}_DAY_PC.d3c')
    baked, used = d3city.bake_area(pages, instances, tuple(args.centre), args.half, d.MAX_STREAM_VERTICES)
    origin = np.array([args.centre[0], np.percentile(baked.vertices[:, 1], GROUND_PERCENTILE), args.centre[1]])
    baked.vertices[:, 0:3] -= origin
    print(f'{args.city}: {len(used)} placed models, {len(baked.vertices)} vertices, {len(baked.triangles)} triangles, '
          f'{len(np.unique(baked.materials))} materials, ground {origin[1]:.1f} m')
    return baked, used, origin


def preview(baked: d3city.Baked, path: Path) -> None:
    glb = GlbBuilder()
    v = baked.vertices
    glb.add_node('area', glb.add_mesh('area', [{'positions': np.ascontiguousarray(v[:, 0:3], dtype='<f4'),
                                                'normals': np.ascontiguousarray(v[:, 3:6], dtype='<f4'),
                                                'uvs': np.ascontiguousarray(v[:, 6:8], dtype='<f4'),
                                                'indices': baked.triangles.reshape(-1).astype('<u4'), 'material': glb.add_material('grey')}]))
    glb.save(path)


def cutout_dds(dds: bytes) -> bytes:
    """The DXT3/DXT5 texture with its alpha cut clean to 0 / full on every mip level. Driv3r's leaf cards sit around the
    alpha-test limit (Miami: half the texels at 136-153) and the mips average to full, so DPL's alpha test cut the leaves
    differently at each distance: they shimmered and popped. Level 0 keeps what passed the test; each smaller level keeps
    the same share of texels (the biggest box averages), so the cards keep their coverage at every distance."""
    fourcc = bytes(dds[84:88])
    height, width, _, _, mips = struct.unpack_from('<5I', dds, 12)

    if fourcc not in (b'DXT3', b'DXT5'):
        return dds

    out = bytearray(dds)
    offset, mask0 = 128, None

    for level in range(max(1, mips)):
        w, h = max(1, width >> level), max(1, height >> level)
        bx, by = (w + 3) // 4, (h + 3) // 4
        blocks = np.frombuffer(out, dtype=np.uint8, offset=offset, count=bx * by * 16).reshape(by, bx, 16)

        if level == 0:
            if fourcc == b'DXT3':
                nibbles = np.stack([blocks[..., 0:8] & 15, blocks[..., 0:8] >> 4], axis=-1).reshape(by, bx, 4, 4)
                alpha = nibbles.transpose(0, 2, 1, 3).reshape(by * 4, bx * 4).astype(float) * 17
            else:
                alpha = np.asarray(Image.open(io.BytesIO(dds)).convert('RGBA'))[..., 3].astype(float)

            mask0 = (alpha[:h, :w] >= 128).astype(float)
            share = mask0.mean()
            mask = mask0
        else:
            f = mask0[:h << level, :w << level].reshape(h, 1 << level, w, 1 << level).mean(axis=(1, 3)) if (
                h << level == height and w << level == width) else np.full((h, w), share)
            order = np.sort(f.ravel())[::-1]
            limit = order[min(len(order) - 1, max(0, round(share * len(order)) - 1))] if share > 0 else 2.0
            mask = (f >= limit) if share > 0 else np.zeros_like(f, dtype=bool)

        full = np.pad(mask.astype(np.uint8), ((0, by * 4 - h), (0, bx * 4 - w)), mode='edge')
        tiles = full.reshape(by, 4, bx, 4).transpose(0, 2, 1, 3).reshape(by, bx, 16)

        if fourcc == b'DXT3':
            blocks[..., 0:8] = (tiles[..., 0::2] | tiles[..., 1::2] << 4) * 15
        else:
            blocks[..., 0] = 255
            blocks[..., 1] = 0
            bits = np.zeros((by, bx), dtype=np.uint64)

            for i in range(16):
                bits |= (tiles[..., i].astype(np.uint64) ^ np.uint64(1)) << np.uint64(3 * i)     # index 0 = full, 1 = clear

            blocks[..., 2:8] = (bits[..., None] >> (np.arange(6, dtype=np.uint64) * 8)).astype(np.uint8)

        offset += bx * by * 16

    return bytes(out)


def add_material(pcmp: d.Pcmp, material: d3city.Material, textures: dict[bytes, int]) -> int | None:
    """Appends a DPL material drawing this Driv3r texture (one texture record per distinct DDS); its index, or None
    for a texture DPL cannot take (not DXT1/3/5, or a side that is not a power of two)."""
    dds = material.dds

    if material.flags[0] & d3city.SHADER_TYPE_MASK == FOLIAGE_SHADER and material.flags[1] & DRIV3R_ALPHA_TEST:
        dds = cutout_dds(dds)

    height, width, _, _, mips = struct.unpack_from('<5I', dds, 12)
    code = d.TEXTURE_CODES.get(dds[84:88])

    if code is None or width & (width - 1) or height & (height - 1):
        return None

    if dds not in textures:
        fmt = TEXTURE_FORMAT_LOW | code << 8 | max(1, mips) << 16 | (width.bit_length() - 1) << 20 | (height.bit_length() - 1) << 24
        textures[dds] = len(pcmp.textures)
        pcmp.textures.append(bytearray(struct.pack('<8I', *struct.unpack('<2I', hashlib.md5(dds).digest()[:8]), *TEXTURE_WORDS, 0, 0, fmt, 0)))
        pcmp.dds.append(dds)

    alpha_test = material.flags[1] & DRIV3R_ALPHA_TEST

    # DPL's decal set (DECAL_FLAGS) for these layers lit roads and kerb ramps wrongly in game (2026-10-23, user: yesterday's
    # plain alpha test looked right), so they take the generic paths below
    # leaf cards carry the env-map bit too, but are only part clear; drawn opaque a tree became a solid dark cone
    if material.flags[0] & d3city.SHADER_TYPE_MASK == FOLIAGE_SHADER and alpha_test and code != d.TEXTURE_CODES[b'DXT1']:
        flags = ALPHA_TEST_FLAGS
    # Driv3r sets both bits on dirt and puddle overlays too; drawn opaque they showed as pale patches fighting the road
    elif code == d.TEXTURE_CODES[b'DXT1'] or material.flags[1] & DRIV3R_ENV_MAP and not (alpha_test and d3city.see_through(material)):
        flags = OPAQUE_FLAGS
    else:
        flags = ALPHA_TEST_FLAGS if alpha_test else ALPHA_FLAGS

    pcmp.substance_slots.append(len(pcmp.slots))
    pcmp.slots.append(textures[dds])
    pcmp.material_pointers.append(len(pcmp.pointers))
    pcmp.pointers.append(len(pcmp.substances))
    pcmp.substances.append(bytearray(struct.pack('<7I', *flags, 0, 1, 0, 0, SUBSTANCE_LAST)))
    pcmp.materials.append(bytearray(struct.pack('<4I', 0, 1, 0, 0)))
    return len(pcmp.materials) - 1


def write_into(raw: d.RawPack, container: int, baked: d3city.Baked, pages: dict[int, d3city.Page],
               textures: dict[bytes, int] | None = None, materials: dict[int, int | None] | None = None,
               share_stream: bool = False) -> int:
    """Writes the baked area into the container; returns how many Driv3r materials got their own DPL material. Pass the
    same textures (DDS -> texture record) and materials (Driv3r material -> DPL material) to every call on one pack, so
    each is stored once. share_stream appends the vertices to the pack's last stream while it has room (DPL's own packs
    hold 1-2 streams), else they get a stream of their own."""
    record = raw.containers[container]
    lods = raw.lod_groups[container]
    first_group = next(g for g in lods if g is not None)
    template_group = raw.groups[first_group]
    template_mesh = raw.meshes[raw.group_meshes[first_group]]
    old_stream = struct.unpack_from('<H', record, 0x18)[0]
    tail = bytes(raw.streams[old_stream][32:d.VERTEX_STRIDE])

    # vertices: position / scale (raw positions stay in int16 range, as DPL's own), normal, uv, then the 28-byte tail of
    # one of the container's own vertices
    v = baked.vertices
    scale = max(SCALE, 2.0 ** math.ceil(math.log2(max(float(np.abs(v[:, 0:3]).max()), 1e-6) / MAX_RAW_POSITION)))
    head = np.empty((len(v), 8), dtype='<f4')
    head[:, 0:3] = v[:, 0:3] / scale
    head[:, 3:8] = v[:, 3:8]
    rows = np.frombuffer(head.tobytes(), dtype=np.uint8).reshape(len(v), 32)
    tails = np.tile(np.frombuffer(tail, dtype=np.uint8), (len(v), 1))
    lit = np.zeros(len(v), dtype=bool)      # vertices with a Driv3r material; water and mud keep the template's colour

    for key in set(baked.materials.tolist()):
        if pages[key >> 16].material(key & 0xFFFF):
            lit[baked.triangles[baked.materials == key].ravel()] = True

    tails[lit, -4:] = (VERTEX_COLOUR, VERTEX_COLOUR, VERTEX_COLOUR, 255)
    stream = np.hstack([rows, tails]).tobytes()

    if share_stream and len(raw.streams[-1]) // d.VERTEX_STRIDE + len(v) <= d.MAX_STREAM_VERTICES:
        base = len(raw.streams[-1]) // d.VERTEX_STRIDE
        raw.streams[-1] += stream
    else:
        base = 0
        raw.streams.append(bytearray(stream))
        raw.descriptors.append(bytearray(raw.descriptors[old_stream]))

    # one mesh per Driv3r material, each with its own new DPL material where the texture fits
    first_mesh = len(raw.meshes)
    textures = {} if textures is None else textures
    materials = {} if materials is None else materials
    converted = 0

    # meshes in the order their materials first appear: a flat overlay (the overhead map) draws later layers on top
    for material in dict.fromkeys(baked.materials.tolist()):
        found = pages[material >> 16].material(material & 0xFFFF)

        if found and found.flags[0] & DRIV3R_HIDDEN:
            continue

        triangles = baked.triangles[baked.materials == material] + base
        strip = strip_of(triangles)
        check_strip(strip, triangles)
        mesh = bytearray(template_mesh)
        struct.pack_into('<I', mesh, 12, len(strip))

        if material not in materials:
            materials[material] = add_material(raw.pcmp, found, textures) if found and raw.pcmp else None

        index = materials[material]

        if index is not None:
            # material word: index, then the table's page (0 = the global pack's): the new material is in this pack's
            struct.pack_into('<Hh', mesh, 48, index, struct.unpack_from('<I', raw.header, 4)[0])
            converted += 1
        raw.meshes.append(mesh)
        raw.mesh_indices.append(len(raw.indices) // 2)
        raw.indices += np.array(strip, dtype='<u2').tobytes()

    group = bytearray(template_group)
    struct.pack_into('<16f', group, 4, *np.eye(4).reshape(-1))
    struct.pack_into('<H', group, 0x44, len(raw.meshes) - first_mesh)
    raw.groups.append(group)
    raw.group_meshes.append(first_mesh)
    new_group = len(raw.groups) - 1

    lo, hi = v[:, 0:3].min(axis=0), v[:, 0:3].max(axis=0)
    struct.pack_into('<3f', record, 8, scale, scale, scale)
    struct.pack_into('<H', record, 0x18, len(raw.streams) - 1)

    for k, (x, y, z) in enumerate((x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])):
        struct.pack_into('<4f', record, 0x24 + k * 16, x, y, z, 1.0)

    # Only the first used LOD entry draws the new group: the loader walks every entry's groups (and patches the meshes of
    # entries 5 and 6), so one group in several entries crashed the game. The other entries are emptied.
    first_entry = lods.index(first_group)

    for k, group_index in enumerate(lods):
        if group_index is not None:
            lods[k] = new_group if k == first_entry else None
            struct.pack_into('<I', record, 0xA8 + k * 24, 1 if k == first_entry else 0)

    struct.pack_into('<I', record, 0xA4 + first_entry * 24 + 16, ALL_DISTANCES)

    return converted


def pdl_record(parts: list[bytes]) -> d.PdlRecord:
    """A record of these parts; its radius reaches every part from the container origin."""
    radius = max(float(np.linalg.norm(struct.unpack_from('<3f', p, 64)) + np.linalg.norm(struct.unpack_from('<3f', p, 0))) for p in parts)
    return d.PdlRecord(parts, radius, 0)


def upright(part: bytearray) -> None:
    """Reorders a part's axes so axis 1 is the one nearest to up, as on DPL's own parts (Driv3r's Z-up export swaps y
    and z). The shape stays the same: half extents move with their axes, signs keep the rotation proper."""
    rows = np.array(struct.unpack_from('<12f', part, 16)).reshape(3, 4)[:, :3]
    half = np.array(struct.unpack_from('<3f', part, 0))
    up = int(np.argmax(abs(rows[:, 1])))
    order = [k for k in range(3) if k != up]
    order.insert(1, up)
    rows, half = rows[order], half[order]

    if rows[1][1] < 0:
        rows[1] = -rows[1]
        rows[0] = -rows[0]

    if np.linalg.det(rows) < 0:
        rows[2] = -rows[2]

    struct.pack_into('<3f', part, 0, *half)

    for k in range(3):
        struct.pack_into('<3f', part, 16 + k * 16, *rows[k])


def widen_deck(part: bytearray) -> None:
    """Grows an upright box that is a deck or road slab (level, thin, wide on both horizontal axes) by DECK_OVERLAP sideways,
    and down to DECK_MIN_HALF_HEIGHT with its top kept."""
    half = np.array(struct.unpack_from('<3f', part, 0))
    axis_up = np.array(struct.unpack_from('<3f', part, 16 + 16))

    if axis_up[1] > DECK_LEVEL and min(half[0], half[2]) >= DECK_HALF_WIDTH and half[1] <= DECK_HALF_HEIGHT:
        grow = max(DECK_MIN_HALF_HEIGHT - half[1], 0.0)
        struct.pack_into('<3f', part, 0, half[0] + DECK_OVERLAP, half[1] + grow, half[2] + DECK_OVERLAP)
        struct.pack_into('<3f', part, 64, *(np.array(struct.unpack_from('<3f', part, 64)) - grow * axis_up))


def part_corners(part: bytes) -> np.ndarray:
    """The 8 corners (in BOX_SIGNS order) of a part's box; cylinders and spheres as their bounding boxes."""
    half = np.array(struct.unpack_from('<3f', part, 0))
    shape = struct.unpack_from('<I', part, 80)[0] & 0xFF
    half = {1: half[[0, 1, 0]], 2: half[[0, 0, 0]]}.get(shape, half)
    rows = np.array(struct.unpack_from('<12f', part, 16)).reshape(3, 4)[:, :3]
    return (BOX_SIGNS * half) @ rows + np.array(struct.unpack_from('<3f', part, 64))


def bounding_part(parts: list[bytes]) -> bytes:
    """Part 0 of a record of several parts, as in both games: an axis-aligned box (shape and surface word 0) around the
    others. The game tests the other parts only inside it."""
    corners = np.concatenate([part_corners(p) for p in parts])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    part = bytearray(d.PDL_PART)
    struct.pack_into('<3f', part, 0, *((hi - lo) / 2))
    struct.pack_into('<12f', part, 16, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    struct.pack_into('<4f', part, 64, *((hi + lo) / 2), 1.0)
    return bytes(part)


def collision_records(records: list[list[bytearray]], origin: np.ndarray) -> list[d.PdlRecord]:
    """One model's world-space records as DPL records around the container origin. Driv3r's bounding parts are dropped
    and made anew around the parts in DPL's frame; a record of more than MAX_RECORD_PARTS is split."""
    out = []

    for record in records:
        parts = record[1:] if len(record) > 1 else record

        for p in parts:
            if struct.unpack_from('<I', p, 80)[0] & 0xFF == 0:     # boxes only: a cylinder's axes carry radius and height
                upright(p)
                widen_deck(p)

            struct.pack_into('<3f', p, 64, *(np.array(struct.unpack_from('<3f', p, 64)) - origin))
            struct.pack_into('<3I', p, 84, 0, 0, 0)     # Driv3r's material floats; DPL's v4 parts keep zeros there
            struct.pack_into('<I', p, 80, struct.unpack_from('<I', p, 80)[0] & 0xFF | DPL_SURFACE << 8)

        out += [bounded([bytes(p) for p in parts[i:i + MAX_RECORD_PARTS - 1]]) for i in range(0, len(parts), MAX_RECORD_PARTS - 1)]

    return out


def bounded(parts: list[bytes]) -> d.PdlRecord:
    """A record of these parts, with a bounding part 0 when there are several."""
    return pdl_record([bounding_part(parts)] + parts if len(parts) > 1 else parts)


def container_instance(tree: d.ChunkTree, page: int, container: int) -> tuple[d.Chunk, int, d.RegionInstance]:
    """(region, index in its GEBI, instance) of the region instance that places the container; it must have yaw 0."""
    handle = page << 10 | container

    for region in (top for top in tree.chunks if top.tree and top.tag == b'GEIR'):
        chunks = {c.tag: c for c in region.tree.chunks}
        instances = d.region_instances(bytes(chunks[b'GEBI'].data)) if b'GEBI' in chunks and b'JNR!' in chunks else []
        index = next((i for i, inst in enumerate(instances) if inst.handle == handle), None)

        if index is not None:
            break
    else:
        raise SystemExit(f'no region places page {page} container {container}')

    if abs(instances[index].sin) > 1e-3:   # limit: nothing is turned into the instance's frame, upgrade: a turned target
        raise SystemExit('the target instance must have yaw 0')

    return region, index, instances[index]


def new_road(road: d3_ai.Road, shift: np.ndarray) -> dpl_ai.NewRoad:
    """A Driv3r road for dpl_ai.add_roads, moved by shift. Lanes lie on the road surface in both games (an earlier 0.8 m
    lift came from drop tests: that is the character's origin above the ground). Driv3r lane offsets count to the
    right of the road direction, DPL's to the left."""
    return dpl_ai.NewRoad([[p + shift for p in s.points()] for s in road.segments],
                          ([-o for o in road.lanes[0]], [-o for o in road.lanes[1]]),
                          [-o for o in road.pavements], [-o for o in road.parking], road.fast)


def write_roads(tree: d.ChunkTree, page: int, container: int, d3c: Path, centre: tuple[float, float], half: float,
                origin: np.ndarray) -> str:
    """Adds the Driv3r roads of the square (those touching it and lying within ROAD_REACH of it), with their pavements and
    parking strips, to DPL's road network (AIEX of the global region), moved like the square's geometry. Returns a one-line
    summary."""
    _, _, place = container_instance(tree, page, container)
    shift = np.array([place.x, place.y, place.z]) - origin
    roads = []

    for road in (road for road in d3_ai.read_roads(d3_ai.aiex_chunk(d3c)) if not road.rail):
        xz =np.array([p[[0, 2]] for p in road.points()]) - centre

        if np.abs(xz).max(axis=1).min() <= half and np.abs(xz).max() <= half + ROAD_REACH:
            roads.append(new_road(road, shift))

    aiex = next(c for top in tree.chunks if top.tree and top.tag == b'GEGR' for c in top.tree.chunks if c.tag == b'AIEX')
    ai = dpl_ai.read(bytes(aiex.data))
    summary = dpl_ai.add_roads(ai, roads)
    aiex.data = dpl_ai.write(ai)
    return f'roads: {summary}'


def write_collision(tree: d.ChunkTree, page: int, container: int, models: list[list[d.PdlRecord]]) -> str:
    """Replaces the collision of the container's instance by the records (per model; coordinates around the container
    origin): appended to the super region's GEPM, referenced from the broadphase node of the region that holds each
    model's box. Returns a one-line summary."""
    super_region = next(top for top in tree.chunks if top.tree and top.tag == b'GESR' and
                        any(c.tag == b'MDXN' and struct.unpack_from('<I', c.data, 4)[0] == page for c in top.tree.chunks))
    gepm = next(c for c in super_region.tree.chunks if c.tag == b'GEPM')
    gepm_id = struct.unpack_from('<I', gepm.data, 0)[0]
    region, index, place = container_instance(tree, page, container)
    chunks = {c.tag: c for c in region.tree.chunks}
    bp = d.split_broadphase(bytes(chunks[b'JNR!'].data))
    bp.objects = [[o for o in objects if o >> 16 != index] for objects in bp.objects]
    first = struct.unpack_from('<I', gepm.data, d.GEPM_HEADER)[0]
    pending = [r for model in models for r in model]
    records: list[d.PdlRecord] = []
    missed = 0

    while pending:
        record = pending.pop(0)
        centres = np.array([struct.unpack_from('<3f', p, 64) for p in record.parts])
        reach = np.array([np.linalg.norm(struct.unpack_from('<3f', p, 0)) for p in record.parts])
        lo = (centres[:, [0, 2]] - reach[:, None]).min(axis=0) + (place.x, place.z)
        hi = (centres[:, [0, 2]] + reach[:, None]).max(axis=0) + (place.x, place.z)

        if d.broadphase_insert(bp, index << 16 | gepm_id << 11 | (first + len(records)), lo[0], lo[1], hi[0], hi[1]):
            records.append(record)
        elif len(record.parts) > 1:     # a box across cell edges lands in a full shallow node: its parts fit deeper alone
            pending[:0] = [pdl_record([p]) for p in record.parts[1:]]
        else:
            missed += 1

    if missed:
        raise SystemExit(f'broadphase: no room for {missed} parts')

    gepm.data = d.append_pdl_records(bytes(gepm.data), records)
    chunks[b'JNR!'].data = d.join_broadphase(bp)
    return (f'collision: {sum(len(r.parts) for r in records)} parts in {len(records)} records (GEPM {gepm_id} from {first}), '
            f'region {region.name.decode()} instance {index}')


def collision_mesh(models: list[list[d.PdlRecord]], material: int) -> d3city.Baked:
    """The records' parts as closed boxes (cylinders and spheres as their bounding boxes), both windings, for checking in
    game where the collision lies: row k of a part is its box axis k."""
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    vertices, triangles = [], []

    for part in (p for model in models for record in model for p in (record.parts[1:] or record.parts)):
        base = len(vertices) * 8
        vertices.append(np.hstack([part_corners(part), BOX_SIGNS / np.sqrt(3), np.zeros((8, 2))]))

        for a, b, c, e in quads:
            triangles += [(base + a, base + b, base + c), (base + a, base + c, base + e),
                          (base + a, base + c, base + b), (base + a, base + e, base + c)]

    tri = np.array(triangles, dtype=np.int64)
    return d3city.Baked(np.concatenate(vertices), tri, np.full(len(tri), material, dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('d4c', type=Path)
    ap.add_argument('driv3r', type=Path)
    ap.add_argument('--city', default='Miami')
    ap.add_argument('--centre', type=float, nargs=2, default=(-384.0, 128.0), metavar=('X', 'Z'))
    ap.add_argument('--half', type=float, default=128.0)
    ap.add_argument('--page', type=int, default=12)
    ap.add_argument('--container', type=int, default=171)
    ap.add_argument('--preview', type=Path)
    ap.add_argument('--show-collision', action='store_true', help='draw the collision boxes instead of the buildings')
    args = ap.parse_args()
    d.require_copy(args.d4c)
    pages = d3city.read_pages(args.driv3r / 'Cities' / f'{args.city}_DAY_PC' / 'SuperRegions.pcs')
    baked, used, origin = bake(args, pages)
    collision = d3city.read_collision(args.driv3r / 'Cities' / f'{args.city}_DAY_PC.d3c')
    models = [collision_records([d3city.place_parts(r, i) for r in records], origin)
              for i in used if (records := collision.records(i))]

    if args.show_collision:
        baked = collision_mesh(models, int(baked.materials[0]))

    if args.preview:
        preview(baked, args.preview)
        print('preview', args.preview)

    new = args.d4c.with_suffix('.d4c.new')

    with args.d4c.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        tree = d.read_tree(data)
        chunk = d.find_pack(tree, args.page)
        raw = d.split_pack(chunk.data, 0, len(chunk.data))
        converted = write_into(raw, args.container, baked, pages)
        chunk.data = d.join_pack(raw)
        summary = write_collision(tree, args.page, args.container, models)
        roads = write_roads(tree, args.page, args.container, args.driv3r / 'Cities' / f'{args.city}_DAY_PC.d3c',
                            tuple(args.centre), args.half, origin)

        with new.open('wb') as out:
            for piece in d.tree_pieces(tree):
                out.write(piece)

        del tree, chunk, raw      # release the views into the old file before it is replaced

    os.replace(new, args.d4c)
    print(f'page {args.page} container {args.container}: new stream with {len(baked.vertices)} vertices, {converted} Driv3r materials '
          f'as DPL materials; file {os.path.getsize(args.d4c):#x} bytes')
    print(summary)
    print(roads)


if __name__ == '__main__':
    main()
