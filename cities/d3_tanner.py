"""Tanner, Driv3r's player, as the player of Driver Parallel Lines: the character of the Driv3r DLC.

usage: d3_tanner.py <DPL dir> <Driv3r dir> [--level | --restore-level] [--preview out.glb]

Writes Characters/nyc_d3.sp: a copy of DPL's Characters/nyc.sp with Tanner, the character every Driv3r mission holds
(Missions/mission01.dam, MDPC container uid 0x97370000), added as character TANNER_ID; TK (character 4) and every other
character stay. The Extended window's player dropdown switches the player to that id. DPL's own nyc.sp is never written. --level lets the first LEVEL of Cities/levels.txt (the Driv3r city
level) load the copy (CharacterModels), --restore-level points it back at nyc.sp. With the game running the copy
waits as nyc_d3.sp.new.

Conversion (docs/D3-PORT-KEYPOINTS.md "Character facts"):
- Pose: both skeletons share their bind pose except the arms (Driv3r level, DPL hanging). Every vertex goes model ->
  Driv3r bone (its inverse bind) -> the same DPL bone (bind), blended by its weights; the toes ride on the feet and
  the upper chest on the back (DPL has neither bone). Normals and the vector in words 8-10 turn along.
- Vertex: Driv3r 16 words = position (metres), normal, uv, vector, u32 bone bytes, 3 f32 weights; DPL 16 words =
  position / container scale, normal, uv, the same vector, a second unit vector (words 11-13, cross of the vector and
  the normal on DPL's own characters), u32 bone bytes, u32 weights (two 11-bit fields of 1023; DPL uses two bones,
  so a third Driv3r weight is dropped and the two renormalised).
- Spool: a copy of TK's CHAC with the new id in UPXD, its MDXN on page 0xC00 + id, before BINF; the id is added to the
  SSLP lookup (id, CHAC ordinal, kind 1) and the CSCT list (count, then id and 8 zero bytes per character).
- Pack: TK's MDXN is the template (container record, scale, LOD kinds 16/8/4, mesh and group records, page 0xC04); its
  meshes, vertices and materials are replaced by Tanner's (triangle strips, one mesh per material and LOD), his two
  textures get DPL's plain character substance.
"""
import argparse
import dataclasses
import mmap
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d3_vehicle_swap as s
import d3city
import dplchunk as d
import skeletons as sk
import vehchunk as v
from dpl_poc import DRIV3R_ALPHA_TEST, DRIV3R_ENV_MAP, add_material, check_strip, strip_of
from glb import GlbBuilder

ORIGINAL = 'nyc.sp'
COPY = 'nyc_d3.sp'
LEVEL_KEY = b'CharacterModels'
TK_ID = 4
# an id nyc.sp lacks whose entry in the exe's character table (0x20 bytes per id from 0x6CFEE0, ids up to 205) is of
# TK's kind; an id past the table loads but is never drawn
TANNER_ID = 165
CHARACTER_PAGE = 0xC00
SSLP_HEADER = 0x1C              # u16 2, u16 count, sizes; then (u16 character id, u16 CHAC ordinal, u16 kind) per CHAC
SSLP_COUNT = 2
SSLP_RECORD = 6
# kind 0 (TK's) never loaded next to TK (one player slot? the header's 4.9 MB size); kind 1 loads (the 43 MB pool?)
SSLP_KIND = 1
CSCT_RECORD = 12
TANNER_FILE = Path('Missions') / 'mission01.dam'
TANNER_UID = 0x97370000
DPL_SKELETON = Path('Anims') / 'nyc_now.an4'
SKELETON = 'male_skeleton'
CARRIED_BY = {'bone_left_toe': 'bone_left_foot', 'bone_right_toe': 'bone_right_foot', 'bone_upperchest': 'bone_back'}
VERTEX_FLOATS = 16
D3_CONTAINER = 0x188
LOD_RECORD = 24
WEIGHT_ONE = 1023
WEIGHT_BITS = 11
BONES_PER_VERTEX = 3            # Driv3r: bone bytes 0-2 with the weights in words 12-14
# substance flags of DPL's characters: opaque (TK's skin) and alpha-tested (bit 0x1000). Driv3r's environment-map
# materials (Tanner's skin and clothes) keep a reflection mask in their alpha: drawn alpha-tested he is invisible
CHARACTER_OPAQUE = (0, 0x4017)
CHARACTER_ALPHA_TEST = (0, 0x5017)


def tanner(driv3r: Path) -> tuple[d3city.Page, int]:
    """Tanner's MDPC pack (vertices re-read at the 64-byte character stride) and his drawn container."""
    data = (driv3r / TANNER_FILE).read_bytes()
    pack = v.find_all(data, 'MDPC')[0]
    page = d3city.Page(data, pack.offset, pack.size)
    h = struct.unpack_from('<20I', data, pack.offset)
    start = pack.offset + ((h[14] + h[12] * 2 + 0xFFF) & ~0xFFF)
    end = pack.offset + h[11]
    page.vertices = np.frombuffer(data, dtype='<f4', count=(end - start) // 4 // VERTEX_FLOATS * VERTEX_FLOATS,
                                  offset=start).reshape(-1, VERTEX_FLOATS)

    for i, lods in enumerate(page.containers):
        uid = struct.unpack_from('<I', data, pack.offset + h[3] + i * D3_CONTAINER)[0]

        if uid == TANNER_UID and any(groups and kind for kind, groups in lods):
            return page, i

    raise SystemExit(f'{TANNER_FILE}: no container {TANNER_UID:#x} (Tanner)')


def pose_transforms(dpl: Path, driv3r: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per Driv3r bone: the 4x4 that moves a model-space point from Driv3r's bind pose into DPL's, and its DPL bone."""
    d3_pose = sk.bind_pose(driv3r / TANNER_FILE, SKELETON)
    dpl_pose = sk.bind_pose(dpl / DPL_SKELETON, SKELETON)
    dpl_bones = list(dpl_pose)
    transforms, targets = [], []

    for name in d3_pose:
        frame = CARRIED_BY.get(name, name)
        transforms.append(np.linalg.inv(d3_pose[frame]) @ dpl_pose[frame])
        targets.append(dpl_bones.index(frame))

    return np.array(transforms), np.array(targets)


def dpl_vertices(d3: np.ndarray, transforms: np.ndarray, targets: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(DPL vertex rows N x 16 as float32 with the two u32 skin words, positions in metres)."""
    bone_words = d3[:, 11].view('<u4')
    bones = np.stack([(bone_words >> (8 * i)) & 0xFF for i in range(BONES_PER_VERTEX)], axis=1).astype(np.int64)
    weights = d3[:, 12:12 + BONES_PER_VERTEX].astype(np.float64)
    points = np.hstack([d3[:, 0:3], np.ones((len(d3), 1))])
    position = np.zeros((len(d3), 3))
    normal = np.zeros((len(d3), 3))
    vector = np.zeros((len(d3), 3))

    for i in range(BONES_PER_VERTEX):
        m = transforms[bones[:, i]]
        w = weights[:, i:i + 1]
        position += w * np.einsum('nk,nkl->nl', points, m)[:, 0:3]
        normal += w * np.einsum('nk,nkl->nl', d3[:, 3:6], m[:, 0:3, 0:3])
        vector += w * np.einsum('nk,nkl->nl', d3[:, 8:11], m[:, 0:3, 0:3])

    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    vector *= (np.linalg.norm(d3[:, 8:11], axis=1) / np.maximum(np.linalg.norm(vector, axis=1), 1e-9))[:, None]
    second = np.cross(vector, normal)
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-9)

    # the two heaviest bones, renormalised; a lone bone keeps bone 0 in the second byte with weight 0, as DPL's own
    order = np.argsort(-weights, axis=1)[:, :2]
    top = np.take_along_axis(weights, order, axis=1)
    first = np.rint(WEIGHT_ONE * top[:, 0] / top.sum(axis=1)).astype(np.uint32)
    second_weight = np.where(top[:, 1] > 0, WEIGHT_ONE - first, 0).astype(np.uint32)
    dpl_bones = targets[np.take_along_axis(bones, order, axis=1)].astype(np.uint32)
    dpl_bones[:, 1] = np.where(second_weight > 0, dpl_bones[:, 1], 0)

    rows = np.zeros((len(d3), VERTEX_FLOATS), dtype='<f4')
    rows[:, 0:3] = position / scale
    rows[:, 3:6] = normal
    rows[:, 6:8] = d3[:, 6:8]
    rows[:, 8:11] = vector
    rows[:, 11:14] = second
    words = rows.view('<u4')
    words[:, 14] = dpl_bones[:, 0] | dpl_bones[:, 1] << 8
    words[:, 15] = first | second_weight << WEIGHT_BITS
    return rows, position


def build_pack(template: d.RawPack, page: d3city.Page, container: int, transforms: np.ndarray,
               targets: np.ndarray) -> tuple[bytes, np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]:
    """TK's pack with Tanner's meshes, vertices and textures; also the DPL vertex rows, their positions in metres and the
    most detailed LOD's triangles per material (for the preview)."""
    raw = template
    record = raw.containers[0]
    scale = np.array(struct.unpack_from('<3f', record, 8))
    page_word = struct.unpack_from('<I', raw.header, 4)[0] << 16
    template_mesh = raw.meshes[0]
    lod_templates = {k: raw.groups[g] for k, g in enumerate(raw.lod_groups[0]) if g is not None}
    kinds = {struct.unpack_from('<I', record, 0xA4 + k * LOD_RECORD + 16)[0]: k for k in lod_templates}
    raw.groups, raw.group_meshes, raw.meshes, raw.mesh_indices = [], [], [], []
    raw.indices = bytearray()
    raw.lod_groups[0] = [None] * len(raw.lod_groups[0])
    pcmp = raw.pcmp
    pcmp.materials, pcmp.material_pointers, pcmp.pointers, pcmp.substances = [], [], [], []
    pcmp.substance_slots, pcmp.slots, pcmp.textures, pcmp.dds = [], [], [], []
    textures: dict[bytes, int] = {}
    materials: dict[int, int] = {}
    order: list[int] = []
    block: dict[int, int] = {}
    preview: list[tuple[int, np.ndarray]] = []

    for kind, groups in page.containers[container]:
        if not groups or kind not in kinds:
            continue

        slot = kinds[kind]
        first_index, first_vertex, triangles_drawn = len(raw.indices) // 2, len(order), 0
        group = bytearray(lod_templates[slot])
        raw.lod_groups[0][slot] = len(raw.groups)
        raw.group_meshes.append(len(raw.meshes))

        for mesh in (m for g in groups for m in g.meshes):
            if mesh.material not in materials:
                found = page.material(mesh.material)
                added = add_material(pcmp, found, textures) if found else None

                if added is None:
                    raise SystemExit(f'Tanner material {mesh.material}: no DXT texture DPL can take')

                cut_out = found.flags[1] & DRIV3R_ALPHA_TEST and not found.flags[1] & DRIV3R_ENV_MAP
                struct.pack_into('<2I', pcmp.substances[-1], 0, *(CHARACTER_ALPHA_TEST if cut_out else CHARACTER_OPAQUE))
                materials[mesh.material] = added

            triangles = page.mesh_triangles(mesh)

            for i in np.unique(triangles).tolist():
                if i not in block:
                    block[i] = len(order)
                    order.append(i)

            triangles = np.array([block[i] for i in triangles.reshape(-1).tolist()]).reshape(triangles.shape)
            strip = strip_of(triangles)
            check_strip(strip, triangles)
            m = bytearray(template_mesh)
            struct.pack_into('<I', m, 12, len(strip))
            struct.pack_into('<I', m, 48, page_word | materials[mesh.material])
            raw.meshes.append(m)
            raw.mesh_indices.append(len(raw.indices) // 2)
            raw.indices += np.array(strip, dtype='<u2').tobytes()
            triangles_drawn += len(triangles)

            if slot == 0:
                preview.append((materials[mesh.material], triangles))

        struct.pack_into('<H', group, 0x44, len(raw.meshes) - raw.group_meshes[-1])
        raw.groups.append(group)
        # (strip indices << 16 | triangles) as on DPL's own characters
        struct.pack_into('<I', record, 0xA4 + slot * LOD_RECORD + 8, (len(raw.indices) // 2 - first_index) << 16 | triangles_drawn)

    if len(order) > 0xFFFF:
        raise SystemExit(f'{len(order)} vertices: more than 16-bit mesh indices reach')

    rows, positions = dpl_vertices(page.vertices[order], transforms, targets, scale)
    raw.streams = [bytearray(rows.tobytes())]
    lo, hi = positions.min(axis=0), positions.max(axis=0)

    for k, (x, y, z) in enumerate((x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])):
        struct.pack_into('<4f', record, 0x24 + k * 16, x, y, z, 1.0)

    return d.join_pack(raw), rows, positions, preview


def character(tree: d.ChunkTree, character_id: int) -> d.Chunk | None:
    """The CHAC whose UPXD names this character id."""
    for chunk in tree.chunks:
        if chunk.tag == b'CHAC' and chunk.tree:
            upxd = next(c for c in chunk.tree.chunks if c.tag == b'UPXD')

            if struct.unpack_from('<I', upxd.data)[0] == character_id:
                return chunk

    return None


def add_character(tree: d.ChunkTree, template: d.Chunk, character_id: int) -> d.Chunk:
    """A copy of the template CHAC as a new character before BINF, listed in SSLP and CSCT; returns its MDXN chunk, whose
    pack header already names the new page (build_pack writes its meshes on that page)."""
    parts = [dataclasses.replace(c) for c in template.tree.chunks]
    upxd = next(c for c in parts if c.tag == b'UPXD')
    data = bytearray(upxd.data)
    struct.pack_into('<2I', data, 0, character_id, character_id)
    upxd.data = bytes(data)
    mdxn = next(c for c in parts if c.tag == b'MDXN')
    data = bytearray(mdxn.data)
    struct.pack_into('<I', data, 4, CHARACTER_PAGE + character_id)
    mdxn.data = bytes(data)
    tail = next(i for i, c in enumerate(tree.chunks) if c.tag == b'BINF')
    tree.chunks.insert(tail, dataclasses.replace(template, tree=dataclasses.replace(template.tree, chunks=parts), gap_extra=0))
    ssic = {c.tag: c for c in next(c for c in tree.chunks if c.tag == b'SSIC').tree.chunks}
    sslp = bytearray(ssic[b'SSLP'].data)
    count = struct.unpack_from('<H', sslp, SSLP_COUNT)[0]
    end = SSLP_HEADER + count * SSLP_RECORD
    struct.pack_into('<H', sslp, SSLP_COUNT, count + 1)
    # the CHAC ordinal: SSIC is entry 0
    ssic[b'SSLP'].data = bytes(sslp[:end] + struct.pack('<3H', character_id, tail, SSLP_KIND) + sslp[end:])
    csct = bytearray(ssic[b'CSCT'].data)
    count = struct.unpack_from('<I', csct, 0)[0]
    struct.pack_into('<I', csct, 0, count + 1)
    end = 4 + count * CSCT_RECORD
    ssic[b'CSCT'].data = bytes(csct[:end] + struct.pack('<I8x', character_id) + csct[end:])
    return mdxn


def write_preview(path: Path, rows: np.ndarray, positions: np.ndarray, preview: list[tuple[int, np.ndarray]]) -> None:
    glb = GlbBuilder()
    primitives = [{'positions': np.ascontiguousarray(positions, dtype='<f4'), 'normals': np.ascontiguousarray(rows[:, 3:6]),
                   'uvs': np.ascontiguousarray(rows[:, 6:8]),
                   'indices': triangles.reshape(-1).astype('<u4'), 'material': glb.add_material(f'material {material}')}
                  for material, triangles in preview]
    glb.add_node('tanner', glb.add_mesh('tanner', primitives))
    glb.save(path)
    print(f'preview {path}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('dpl', type=Path)
    ap.add_argument('driv3r', type=Path)
    ap.add_argument('--level', action='store_true', help=f'let the first LEVEL load Characters\\{COPY}')
    ap.add_argument('--restore-level', action='store_true', help=f'let the first LEVEL load Characters\\{ORIGINAL} again')
    ap.add_argument('--preview', type=Path, help='also write Tanner in DPL\'s bind pose as .glb (glb_views.py draws it)')
    args = ap.parse_args()
    characters = args.dpl / 'Characters'

    if not (characters / ORIGINAL).is_file() or not (args.driv3r / TANNER_FILE).is_file():
        raise SystemExit(f'pass your Driver Parallel Lines folder (Characters/{ORIGINAL}) and your Driv3r folder ({TANNER_FILE})')

    if args.restore_level:
        s.set_level(args.dpl, ORIGINAL, LEVEL_KEY, 'Characters')
        return

    page, container = tanner(args.driv3r)
    transforms, targets = pose_transforms(args.dpl, args.driv3r)
    target = characters / COPY
    new = target.with_suffix('.sp.new')

    with (characters / ORIGINAL).open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as original:
        tree = d.read_tree(original, 0, 2)

        if character(tree, TANNER_ID) is not None:
            raise SystemExit(f'{ORIGINAL} already has a character {TANNER_ID}')

        chunk = add_character(tree, character(tree, TK_ID), TANNER_ID)
        template = d.split_pack(chunk.data, 0, len(chunk.data))
        chunk.data, rows, positions, preview = build_pack(template, page, container, transforms, targets)

        with new.open('wb') as out:
            for piece in d.tree_pieces(tree):
                out.write(piece)

        del tree, chunk, template     # release the views into the original before the mmap closes

    print(f'Tanner added as character {TANNER_ID}: {len(positions)} vertices, '
          f'height {positions[:, 1].max() - positions[:, 1].min():.2f} m, width {np.ptp(positions[:, 0]):.2f} m')

    if args.preview:
        write_preview(args.preview, rows, positions, preview)

    try:
        os.replace(new, target)
    except PermissionError as error:
        raise SystemExit(f'{target} is in use: close the game and move {new.name} over it') from error

    print(f'written {target}')

    if args.level:
        s.set_level(args.dpl, COPY, LEVEL_KEY, 'Characters')


if __name__ == '__main__':
    main()
