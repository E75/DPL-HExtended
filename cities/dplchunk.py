"""Reader for the MDXN model pack of Driver Parallel Lines (Cities/*.d4c, one per GESR super region, GEGR global, IRCT interiors).

Same CHNK container as Driv3r; the pack differs from Driv3r's MDPC:
- header words: [2] containers, [3] container offset, [4] groups, [5] group offset, [6] meshes, [7] mesh offset,
  [10] texture data offset, [11] PCMP offset, [12] index count, [13] index bytes, [14] index offset, [15] vertex streams,
  [16] stream descriptor offset (32 bytes each: vertex count, bytes, offset, stride 60).
- container 0x14C bytes: +0x08 position scale per axis (f32x3; vertex position * scale = metres), +0x18 u16 vertex stream,
  +0x24 box (8 corners of f32x4), +0xA4 7 LOD entries of 24 bytes (group offset, group count, ?, ?, kind, ?).
- group 78 bytes: mesh offset, 4x4 matrix (+0x04), mesh count u16 (+0x44).
- mesh 56 bytes: [0] primitive type (6 = triangle strip), [3] index count, [4] first index in BYTES, [12] low u16 material.
- vertex 60 bytes: position f32x3 (scaled, see container), normal f32x3, uv f32x2, 28 bytes tail.
- PCMP v3, header 0x48 bytes, tables packed back to back: [2] materials at [3] (16 bytes: substance pointer list offset,
  count), substance pointers ([4]/[5], u32), substances ([6]/[7], 28 bytes: flags, flags, slot list offset, slot count,
  0, 0, ?), slots ([12]/[13], u32 texture record offset), textures ([14]/[15], 32 bytes: hash, hash, ?, ?, data offset,
  ?, format, ?). [16] texture data offset (tables end aligned to 0x1000): complete DDS files in record order, each padded
  with zeros to 0x80. Format word: bits 8-15 code (0x0C DXT1, 0x0E DXT3, 0x0F DXT5), bits 16-19 mip count,
  bits 20-23 log2 width, bits 24-27 log2 height. Pack header [10] = [11] + PCMP [16] * 256 in every pack.
- GEPM v4 (DPL also loads Driv3r's v3): u32 GEPM id (the super region; 0 = global region), u32 offset of the
  PDL block (0x10), 2 words. PDL ("PDL001.002.003a", same in both games): u32 record count, u32 part count, u32 table
  offset, u32 part data offset (both relative to the PDL block), the 16-byte name, then records of 16 bytes (part count,
  offset of the first part relative to the record, radius f32, flags) and their parts, back to back. Part 96 bytes:
  half extents f32x3 (+0), rotation rows f32x3 at +16/+32/+48 (row k = box axis k), centre f32x3 at +64 (w 1), u32 at
  +80 = shape (low byte: 0 box, 1 cylinder (radius, half height), 2 sphere (radius)) | surface << 8, 3 words (DPL 0,
  Driv3r 0.7, 0.6, 1.0). In a record of several parts (never 2), part 0 is an axis-aligned box with word +80 = 0 around
  the others: the game tests the other parts only inside it, and a part outside it never collides.
- JNR! v6 (v5 loads too) "Physics Broadphase", per region: u32 0x10, u32 size - 0x10, then from +0x10: node offset, node
  count, object offset, object count (offsets relative to +0x10), region size, min x, min z (f32), u32 column, row, maximum node depth (15), 16-byte name;
  nodes of 8 u16: object count << 12 | first object, depth, cell x, cell z (cell = size >> depth from min x/z), 4 child
  nodes for cells (2x, 2z) (2x+1, 2z) (2x, 2z+1) (2x+1, 2z+1) (0 = none; a child may skip depths). A node holds the
  objects whose position lies in its cell; object = instance index in the region's GEBI << 16 | GEPM id << 11 | record.
  A query (FUN_005db696) visits a node when its box meets the node's cell grown by LOOSE cells on every side.
"""
import mmap
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VERTEX_STRIDE = 60
MAX_STREAM_VERTICES = 65536     # 16-bit indices
CONTAINER_SIZE = 0x14C
GROUP_SIZE = 78
MESH_SIZE = 56
STRIP = 6
TEXTURE_CODES = {b'DXT1': 0x0C, b'DXT3': 0x0E, b'DXT5': 0x0F}


@dataclass
class Entry:
    tag: str
    offset: int         # absolute in the file
    size: int


def entries(data: bytes | mmap.mmap, base: int = 0) -> list[Entry]:
    """Entries of a CHNK container: 16 byte header (CHNK, size, count, version), then count x (tag, offset, flags, size)."""
    if data[base:base + 4] != b'CHNK':
        return []

    count = struct.unpack_from('<I', data, base + 8)[0]
    result = []

    for i in range(count):
        tag, offset, _, size = struct.unpack_from('<4sIII', data, base + 16 + i * 16)
        result.append(Entry(tag.decode('latin1'), base + offset, size))

    return result


CHUNK_FILL = b'>'
ORIGINAL_CITIES = {'nyc_now_pc.d4c', 'nyc_then_pc.d4c'}


def require_copy(path: Path) -> None:
    """Stops a writing tool that was pointed at one of the game's own city files."""
    if path.name.lower() in ORIGINAL_CITIES:
        raise SystemExit(f'{path.name} is an original game file: work on a copy (dpl_poc_level.py makes one)')


@dataclass
class Chunk:
    """One entry of a CHNK container, for rewriting. Entry: tag, offset, u8 version, u8 name length, u16 log2 alignment, size.
    Layout: the data starts at the next multiple of 2**align (plus `gap_extra`, kept from odd originals), the name follows the data."""
    tag: bytes
    version: int
    align: int
    name: bytes
    data: bytes | memoryview | None = None      # leaf data
    tree: 'ChunkTree | None' = None             # or a nested container
    gap_extra: int = 0


@dataclass
class ChunkTree:
    version: int
    chunks: list[Chunk]
    tail_extra: int = 0     # bytes beyond aligning the end to the last chunk's alignment


def align_up(value: int, alignment: int) -> int:
    return value + (-value % alignment)


def read_tree(data: bytes | mmap.mmap, base: int = 0, depth: int = 1) -> ChunkTree:
    """The container at base; entries that hold a CHNK become nested trees down to `depth` levels."""
    view = memoryview(data)
    _, size, count, version = struct.unpack_from('<4sIII', data, base)
    chunks = []
    end = 16 + count * 16

    for i in range(count):
        tag, offset, chunk_version, name_length, align, chunk_size = struct.unpack_from('<4sIBBHI', data, base + 16 + i * 16)
        body = view[base + offset:base + offset + chunk_size]
        tree = read_tree(data, base + offset, depth - 1) if depth > 0 and body[:4] == b'CHNK' else None
        chunks.append(Chunk(tag, chunk_version, align, bytes(view[base + offset + chunk_size:base + offset + chunk_size + name_length]),
                            None if tree else body, tree, offset - align_up(end, 1 << align)))
        end = offset + chunk_size + name_length

    return ChunkTree(version, chunks, size - align_up(end, 1 << chunks[-1].align) if chunks else 0)


def find_pack(tree: ChunkTree, page: int) -> Chunk:
    """The MDXN chunk with this page, inside a GESR or GEGR chunk."""
    found = [c for top in tree.chunks if top.tree and top.tag in (b'GESR', b'GEGR')
             for c in top.tree.chunks if c.tag == b'MDXN' and struct.unpack_from('<I', c.data, 4)[0] == page]

    if len(found) != 1:
        raise SystemExit(f'page {page}: {len(found)} packs, need exactly one')

    return found[0]


def tree_layout(tree: ChunkTree) -> tuple[list[int], list[int], int]:
    """(chunk offsets, chunk sizes, container size) of a container as it will be written."""
    sizes = [len(c.data) if c.tree is None else tree_layout(c.tree)[2] for c in tree.chunks]
    offsets = []
    end = 16 + len(tree.chunks) * 16

    for chunk, size in zip(tree.chunks, sizes):
        offsets.append(align_up(end, 1 << chunk.align) + chunk.gap_extra)
        end = offsets[-1] + size + len(chunk.name)

    return offsets, sizes, (align_up(end, 1 << tree.chunks[-1].align) if tree.chunks else end) + tree.tail_extra


def tree_pieces(tree: ChunkTree) -> Iterator[bytes | memoryview]:
    """The container's bytes as consecutive pieces; leaf data is passed through, nothing is joined in memory."""
    offsets, sizes, total = tree_layout(tree)
    header = bytearray(struct.pack('<4sIII', b'CHNK', total, len(tree.chunks), tree.version))

    for chunk, offset, size in zip(tree.chunks, offsets, sizes):
        header += struct.pack('<4sIBBHI', chunk.tag, offset, chunk.version, len(chunk.name), chunk.align, size)

    yield bytes(header)
    position = len(header)

    for chunk, offset, size in zip(tree.chunks, offsets, sizes):
        yield CHUNK_FILL * (offset - position)

        if chunk.tree is None:
            yield chunk.data
        else:
            yield from tree_pieces(chunk.tree)

        yield chunk.name
        position = offset + size + len(chunk.name)

    yield CHUNK_FILL * (total - position)


@dataclass
class Mesh:
    prim_type: int
    index_count: int
    start_index: int
    material: int


@dataclass
class Group:
    matrix: tuple[float, ...]
    meshes: list[Mesh] = field(default_factory=list)


@dataclass
class Container:
    scale: tuple[float, float, float]
    stream: int
    box: list[tuple[float, float, float]]
    lods: list[tuple[int, list[Group]]]     # 7 x (kind, groups)


@dataclass
class Pack:
    base: int
    header: tuple[int, ...]
    containers: list[Container]
    streams: list[tuple[int, int]]          # (vertex count, absolute offset)
    indices_offset: int
    pcmp_offset: int


def load_pack(data: bytes | mmap.mmap, base: int) -> Pack:
    h = struct.unpack_from('<17I', data, base)
    n_containers, c_off, n_groups, g_off, n_meshes, m_off = h[2:8]

    meshes = []

    for i in range(n_meshes):
        w = struct.unpack_from('<14I', data, base + m_off + i * MESH_SIZE)
        meshes.append(Mesh(w[0], w[3], w[4] // 2, w[12] & 0xFFFF))

    groups = []

    for i in range(n_groups):
        o = base + g_off + i * GROUP_SIZE
        mesh_off = struct.unpack_from('<I', data, o)[0]
        count = struct.unpack_from('<H', data, o + 0x44)[0]
        first = (mesh_off - m_off) // MESH_SIZE
        groups.append(Group(struct.unpack_from('<16f', data, o + 4), meshes[first:first + count]))

    containers = []

    for i in range(n_containers):
        o = base + c_off + i * CONTAINER_SIZE
        box = [struct.unpack_from('<3f', data, o + 0x24 + k * 16) for k in range(8)]
        lods = []

        for k in range(7):
            offset, n, _, _, kind, _ = struct.unpack_from('<6I', data, o + 0xA4 + k * 24)
            first = (offset - g_off) // GROUP_SIZE
            lods.append((kind, groups[first:first + n] if n else []))

        containers.append(Container(struct.unpack_from('<3f', data, o + 8), struct.unpack_from('<H', data, o + 0x18)[0], box, lods))

    streams = []

    for i in range(h[15]):
        count, _, offset, _ = struct.unpack_from('<4I', data, base + h[16] + i * 32)
        streams.append((count, base + offset))

    return Pack(base, h, containers, streams, base + h[14], base + h[11])


def container_indices(data: bytes | mmap.mmap, pack: Pack, container: Container, lods: range = range(7)) -> np.ndarray:
    """Vertex indices (into the container's stream) of every mesh in the given LOD entries."""
    parts = [np.frombuffer(data, dtype='<u2', count=m.index_count, offset=pack.indices_offset + m.start_index * 2)
             for k in lods for g in container.lods[k][1] for m in g.meshes]
    return np.concatenate(parts) if parts else np.zeros(0, dtype='<u2')


DDS_HEADER = 128


def dds_size(data: bytes | mmap.mmap, offset: int) -> int:
    """Bytes of the DXT DDS file at offset: header plus every mip level (a mip count of 0 means one level)."""
    height, width, _, _, mips = struct.unpack_from('<5I', data, offset + 12)
    block = 8 if data[offset + 84:offset + 88] == b'DXT1' else 16
    return DDS_HEADER + sum(max(1, (max(1, width >> m) + 3) // 4) * max(1, (max(1, height >> m) + 3) // 4) * block
                            for m in range(max(1, mips)))


def material_texture(data: bytes | mmap.mmap, pack: Pack, material: int) -> bytes | None:
    """The DDS file of the first texture of a material, or None."""
    p = pack.pcmp_offset

    if data[p:p + 4] != b'PCMP':      # packs without their own texture table
        return None

    h = struct.unpack_from('<17I', data, p)

    if material >= h[2]:
        return None

    list_off, list_count = struct.unpack_from('<2I', data, p + h[3] + material * 16)

    if list_count == 0:
        return None

    substance = struct.unpack_from('<I', data, p + list_off)[0]
    slot_off, slot_count = struct.unpack_from('<2I', data, p + substance + 8)

    if slot_count == 0:
        return None

    record = struct.unpack_from('<I', data, p + slot_off)[0]
    o = p + h[16] + struct.unpack_from('<I', data, p + record + 16)[0]

    if data[o:o + 4] != b'DDS ':
        return None

    return bytes(data[o:o + dds_size(data, o)])


PCMP_HEADER = 0x48
MATERIAL_SIZE = 16
SUBSTANCE_SIZE = 28
TEXTURE_SIZE = 32
DDS_ALIGN = 0x80


@dataclass
class Pcmp:
    """A PCMP material table as records plus links (indices), for rewriting: join_pcmp recomputes every offset.
    material -> first substance pointer, substance pointer -> substance, substance -> first slot, slot -> texture record.
    A link is None for an empty list (its stored offset is kept)."""
    header: bytes
    materials: list[bytearray]
    material_pointers: list[int | None]
    pointers: list[int]
    substances: list[bytearray]
    substance_slots: list[int | None]
    slots: list[int]
    textures: list[bytearray]
    dds: list[bytes]


def _link(offset: int, first: int, record: int, count: int) -> int | None:
    index, rest = divmod(offset - first, record)
    return index if count and rest == 0 and offset >= first else None


def split_pcmp(data: bytes) -> Pcmp:
    h = struct.unpack_from(f'<{PCMP_HEADER // 4}I', data, 0)

    def records(offset: int, count: int, size: int) -> list[bytearray]:
        return [bytearray(data[offset + i * size:offset + (i + 1) * size]) for i in range(count)]

    def targets(offset: int, count: int, first: int, size: int) -> list[int]:
        return [(v - first) // size for v in struct.unpack_from(f'<{count}I', data, offset)]

    materials = records(h[3], h[2], MATERIAL_SIZE)
    substances = records(h[7], h[6], SUBSTANCE_SIZE)
    textures = records(h[15], h[14], TEXTURE_SIZE)
    dds = []

    for t in textures:
        o = h[16] + struct.unpack_from('<I', t, 16)[0]
        dds.append(bytes(data[o:o + dds_size(data, o)]))

    return Pcmp(bytes(data[:PCMP_HEADER]), materials, [_link(*struct.unpack_from('<I', m), h[5], 4, struct.unpack_from('<I', m, 4)[0]) for m in materials],
                targets(h[5], h[4], h[7], SUBSTANCE_SIZE), substances,
                [_link(*struct.unpack_from('<I', s, 8), h[13], 4, struct.unpack_from('<I', s, 12)[0]) for s in substances],
                targets(h[13], h[12], h[15], TEXTURE_SIZE), textures, dds)


def join_pcmp(pcmp: Pcmp) -> bytes:
    material_off = PCMP_HEADER
    pointer_off = material_off + len(pcmp.materials) * MATERIAL_SIZE
    substance_off = pointer_off + len(pcmp.pointers) * 4
    slot_off = substance_off + len(pcmp.substances) * SUBSTANCE_SIZE
    texture_off = slot_off + len(pcmp.slots) * 4
    data_off = align_up(texture_off + len(pcmp.textures) * TEXTURE_SIZE, PAGE_ALIGN)
    out = bytearray(pcmp.header)

    for m, first in zip(pcmp.materials, pcmp.material_pointers):
        out += m if first is None else struct.pack('<I', pointer_off + first * 4) + m[4:]

    out += struct.pack(f'<{len(pcmp.pointers)}I', *(substance_off + s * SUBSTANCE_SIZE for s in pcmp.pointers))

    for s, first in zip(pcmp.substances, pcmp.substance_slots):
        out += s if first is None else s[:8] + struct.pack('<I', slot_off + first * 4) + s[12:]

    out += struct.pack(f'<{len(pcmp.slots)}I', *(texture_off + t * TEXTURE_SIZE for t in pcmp.slots))
    position = 0

    for t, dds in zip(pcmp.textures, pcmp.dds):
        out += t[:16] + struct.pack('<I', position) + t[20:]
        position += align_up(len(dds), DDS_ALIGN)

    out += bytes(data_off - len(out))

    for dds in pcmp.dds:
        out += dds + bytes(-len(dds) % DDS_ALIGN)

    struct.pack_into('<5I', out, 8, len(pcmp.materials), material_off, len(pcmp.pointers), pointer_off, len(pcmp.substances))
    struct.pack_into('<I', out, 28, substance_off)
    struct.pack_into('<5I', out, 48, len(pcmp.slots), slot_off, len(pcmp.textures), texture_off, data_off)
    return bytes(out)


@dataclass
class RawPack:
    """An MDXN pack as raw records plus the links between them, for rewriting. join_pack(split_pack(x)) == x.

    Links are indices, so records can be added or changed and join_pack recomputes every offset:
    container LOD entry -> first group (None: empty entry, offset kept), group -> first mesh, mesh -> first index (u16 units).
    `pcmp` is the material table with its texture data (None in packs without one).
    """
    header: bytes
    containers: list[bytearray]
    lod_groups: list[list[int | None]]
    groups: list[bytearray]
    group_meshes: list[int | None]
    meshes: list[bytearray]
    mesh_indices: list[int]
    descriptors: list[bytearray]
    indices: bytearray
    streams: list[bytearray]
    pcmp: Pcmp | None


HEADER_SIZE = 0x80
SECTION_ALIGN = 0x80        # groups and stream descriptors
PAGE_ALIGN = 0x1000         # vertex streams, PCMP, pack end
FILL = b'>'
DESCRIPTOR_SIZE = 32


def split_pack(data: bytes | mmap.mmap, base: int, size: int) -> RawPack:
    h = struct.unpack_from('<17I', data, base)
    n_containers, c_off, n_groups, g_off, n_meshes, m_off = h[2:8]

    def records(offset: int, count: int, record: int) -> list[bytearray]:
        return [bytearray(data[base + offset + i * record:base + offset + (i + 1) * record]) for i in range(count)]

    containers = records(c_off, n_containers, CONTAINER_SIZE)
    groups = records(g_off, n_groups, GROUP_SIZE)
    meshes = records(m_off, n_meshes, MESH_SIZE)
    descriptors = records(h[16], h[15], DESCRIPTOR_SIZE)
    lod_groups = [[_link(offset, g_off, GROUP_SIZE, count) for offset, count in (struct.unpack_from('<2I', c, 0xA4 + k * 24) for k in range(7))]
                  for c in containers]
    group_meshes = [_link(struct.unpack_from('<I', g, 0)[0], m_off, MESH_SIZE, struct.unpack_from('<H', g, 0x44)[0]) for g in groups]
    mesh_indices = [struct.unpack_from('<I', m, 16)[0] // 2 for m in meshes]
    streams = [bytearray(data[base + o:base + o + n]) for _, n, o, _ in (struct.unpack_from('<4I', r) for r in descriptors)]
    pcmp = split_pcmp(bytes(data[base + h[11]:base + size])) if h[11] else None
    return RawPack(bytes(data[base:base + HEADER_SIZE]), containers, lod_groups, groups, group_meshes, meshes, mesh_indices,
                   descriptors, bytearray(data[base + h[14]:base + h[14] + h[13]]), streams, pcmp)


def join_pack(raw: RawPack) -> bytes:
    out = bytearray(raw.header)

    def pad(align: int) -> None:
        out.extend(FILL * (-len(out) % align))

    c_off = len(out)
    out.extend(FILL * (len(raw.containers) * CONTAINER_SIZE))
    pad(SECTION_ALIGN)
    g_off = len(out)
    m_off = g_off + len(raw.groups) * GROUP_SIZE

    for c_index, record in enumerate(raw.containers):
        c = bytearray(record)

        for k, group in enumerate(raw.lod_groups[c_index]):
            if group is not None:
                struct.pack_into('<I', c, 0xA4 + k * 24, g_off + group * GROUP_SIZE)

        out[c_off + c_index * CONTAINER_SIZE:c_off + (c_index + 1) * CONTAINER_SIZE] = c

    for g_index, record in enumerate(raw.groups):
        g = bytearray(record)

        if raw.group_meshes[g_index] is not None:
            struct.pack_into('<I', g, 0, m_off + raw.group_meshes[g_index] * MESH_SIZE)

        out.extend(g)

    for m_index, record in enumerate(raw.meshes):
        m = bytearray(record)
        struct.pack_into('<I', m, 16, raw.mesh_indices[m_index] * 2)
        out.extend(m)

    pad(SECTION_ALIGN)
    desc_off = len(out)
    ib_off = desc_off + len(raw.descriptors) * DESCRIPTOR_SIZE
    vb_off = ib_off + len(raw.indices)
    vb_off += -vb_off % PAGE_ALIGN
    stream_offsets = []

    for stream in raw.streams:
        stream_offsets.append(vb_off)
        vb_off += len(stream)

    for record, stream, offset in zip(raw.descriptors, raw.streams, stream_offsets):
        r = bytearray(record)
        stride = struct.unpack_from('<I', r, 12)[0] or VERTEX_STRIDE      # 64 in skinned character packs
        struct.pack_into('<3I', r, 0, len(stream) // stride, len(stream), offset)
        out.extend(r)

    out.extend(raw.indices)
    pad(PAGE_ALIGN)

    for stream in raw.streams:
        out.extend(stream)

    pad(PAGE_ALIGN)
    pcmp_off = texture_off = 0

    if raw.pcmp:
        pcmp_off = len(out)
        out.extend(join_pcmp(raw.pcmp))
        texture_off = pcmp_off + (struct.unpack_from('<I', out, pcmp_off + 64)[0] << 8)

    pad(PAGE_ALIGN)
    struct.pack_into('<17I', out, 0, *struct.unpack_from('<2I', raw.header), len(raw.containers), c_off, len(raw.groups), g_off,
                     len(raw.meshes), m_off, *struct.unpack_from('<2I', raw.header, 32), texture_off, pcmp_off,
                     len(raw.indices) // 2, len(raw.indices), ib_off, len(raw.streams), desc_off)
    return bytes(out)


GEPM_HEADER = 0x10
PDL_ENTRY = 16
PDL_PART = 96
MAX_PDL_RECORDS = 1 << 11
NODE_OBJECTS = 15           # 4 bits of a node's object word
MAX_OBJECTS = 1 << 12       # 12 bits of a node's first object
GEBI_RECORD = 16
LOOSE = 0.5                 # the exe's float at 0x641344


@dataclass
class PdlRecord:
    parts: list[bytes]
    radius: float
    flags: int


def append_pdl_records(gepm: bytes, records: list[PdlRecord]) -> bytes:
    """The GEPM chunk with the records appended after its own (their indices start at its record count). Record offsets are
    relative to the record, so the old part data moves as one block behind the longer table."""
    count, parts, table, data_off = struct.unpack_from('<4I', gepm, GEPM_HEADER)

    if count + len(records) > MAX_PDL_RECORDS or data_off != table + count * PDL_ENTRY:
        raise SystemExit(f'GEPM: cannot add {len(records)} records to {count}')

    shift = len(records) * PDL_ENTRY
    out = bytearray(gepm[:GEPM_HEADER + table])

    for i in range(count):
        n, off, radius, flags = struct.unpack_from('<IIfI', gepm, GEPM_HEADER + table + i * PDL_ENTRY)
        out += struct.pack('<IIfI', n, off + shift, radius, flags)

    old_data = gepm[GEPM_HEADER + data_off:]
    position = len(out) + shift + len(old_data)

    for record in records:
        out += struct.pack('<IIfI', len(record.parts), position - len(out), record.radius, record.flags)
        position += len(record.parts) * PDL_PART

    out += old_data

    for record in records:
        for part in record.parts:
            out += part

    struct.pack_into('<4I', out, GEPM_HEADER, count + len(records), parts + sum(len(r.parts) for r in records), table, data_off + shift)
    return bytes(out)


@dataclass
class Broadphase:
    header: bytes                   # everything before the nodes
    nodes: list[list[int]]          # 8 u16 each; the object word is recomputed by join_broadphase
    objects: list[list[int]]        # per node
    fill: bytes                     # the bytes after the objects up to a multiple of 16 (the tool left text there)


def split_broadphase(data: bytes) -> Broadphase:
    nodes_off, n_nodes, objects_off, n_objects = struct.unpack_from('<4I', data, 0x10)
    nodes = [list(struct.unpack_from('<8H', data, 0x10 + nodes_off + i * 16)) for i in range(n_nodes)]
    objects = [list(struct.unpack_from(f'<{n[0] >> 12}I', data, 0x10 + objects_off + (n[0] & 0xFFF) * 4)) for n in nodes]
    return Broadphase(bytes(data[:0x10 + nodes_off]), nodes, objects, bytes(data[0x10 + objects_off + n_objects * 4:]))


def join_broadphase(bp: Broadphase) -> bytes:
    out = bytearray(bp.header)
    first = 0

    for node, objects in zip(bp.nodes, bp.objects):
        if len(objects) > NODE_OBJECTS or first >= MAX_OBJECTS:
            raise SystemExit(f'broadphase: node with {len(objects)} objects at {first}')

        out += struct.pack('<8H', len(objects) << 12 | first, *node[1:])
        first += len(objects)

    objects_off = len(out) - 0x10

    for objects in bp.objects:
        out += struct.pack(f'<{len(objects)}I', *objects)

    need = -len(out) % 16
    out += bp.fill if len(bp.fill) == need else bytes(need)
    struct.pack_into('<I', out, 4, len(out) - 0x10)
    struct.pack_into('<4I', out, 0x10, len(bp.header) - 0x10, len(bp.nodes), objects_off, first)
    return bytes(out)


def broadphase_insert(bp: Broadphase, obj: int, x0: float, z0: float, x1: float, z1: float) -> bool:
    """Puts obj in the deepest node whose cell holds the box (x0, z0)-(x1, z1). A full node gets a new child for the
    quadrant that holds the box (appended to the node list), else hands obj to its parent. False when nothing has room."""
    size, min_x, min_z = struct.unpack_from('<3f', bp.header, 0x20)
    max_depth = struct.unpack_from('<I', bp.header, 0x34)[0]

    mid_x, mid_z = (x0 + x1) / 2, (z0 + z1) / 2

    def holds(depth: int, cx: int, cz: int) -> bool:
        """The cell has the box's centre and its grown cell the whole box."""
        cell = size / (1 << depth)
        return int((mid_x - min_x) // cell) == cx and int((mid_z - min_z) // cell) == cz and \
            min_x + (cx - LOOSE) * cell <= x0 and x1 <= min_x + (cx + 1 + LOOSE) * cell and \
            min_z + (cz - LOOSE) * cell <= z0 and z1 <= min_z + (cz + 1 + LOOSE) * cell

    path = [0]

    while (child := next((c for c in bp.nodes[path[-1]][4:] if c and holds(*bp.nodes[c][1:4])), 0)):
        path.append(child)

    while len(bp.objects[path[-1]]) >= NODE_OBJECTS and bp.nodes[path[-1]][1] < max_depth:
        depth, cx, cz = bp.nodes[path[-1]][1:4]
        quadrant = next(((qx, qz) for qx in (0, 1) for qz in (0, 1) if holds(depth + 1, 2 * cx + qx, 2 * cz + qz)), None)

        if quadrant is None:
            break

        slot = 4 + quadrant[0] + 2 * quadrant[1]
        node = [0, depth + 1, 2 * cx + quadrant[0], 2 * cz + quadrant[1], 0, 0, 0, 0]
        old = bp.nodes[path[-1]][slot]

        if old:     # a deeper child that skipped this depth goes under the new node
            o_depth, o_cx, o_cz = bp.nodes[old][1:4]
            shift = o_depth - depth - 2
            node[4 + (o_cx >> shift & 1) + 2 * (o_cz >> shift & 1)] = old

        bp.nodes[path[-1]][slot] = len(bp.nodes)
        bp.nodes.append(node)
        bp.objects.append([])
        path.append(len(bp.nodes) - 1)

    for node in reversed(path):
        if len(bp.objects[node]) < NODE_OBJECTS:
            bp.objects[node].append(obj)
            return True

    return False


@dataclass
class RegionInstance:
    x: float
    y: float
    z: float
    cos: float
    sin: float
    handle: int


def region_instances(gebi: bytes) -> list[RegionInstance]:
    """Instances of a GEBI v10 chunk (format in docs/PLAN-CITY-IMPORT.md, part 2), decoded as the exe does (0x47AD16,
    0x47ADE8): yaw sin in the low and cos in the high half of the first word, both biased by 0x7FFF."""
    start = struct.unpack_from('<I', gebi, 0)[0]
    result = []

    for o in range(start, len(gebi) - GEBI_RECORD + 1, GEBI_RECORD):
        s, c, w1, w2, w3 = struct.unpack_from('<2H3I', gebi, o)
        result.append(RegionInstance((w1 & 0xFFFFF) / 64 - 5760, ((w2 & 0xFFFF) - 0x3200) / 128, (w3 & 0xFFFFF) / 64 - 5760,
                                     (c - 0x7FFF) / 32767, (s - 0x7FFF) / 32767, w2 >> 16))

    return result
