"""Readers for the vehicle data of both games: handling blocks (DVO3), the vehicle hierarchy (UPVH), Driv3r's BigVO3
handling files and the LOD entries of Driv3r's MDPC model packs. Format notes: docs/D3-PORT-KEYPOINTS.md.

- DVO3 block (Driv3r .BO3, DPL VO3D chunk): 'DVO3', u32 version 1, u32 count, count x (u16 key, u16 marker, u32 value);
  marker 0xAAAA = f32 value, other markers hold integers or flags. Sparse: only the keys a vehicle sets.
- DPL VO3D chunk: u32 vehicle id (low 12 bits = VSCT index, bits 12-15 = variant), then one DVO3 block.
- Driv3r .BO3: u32 'BVO3', 2, ?, file size, slot table offset, data offset, slot count; slot table = u16 per vehicle
  type (0xFFFF = none), block at data offset + slot value. A .vvs car's top-level chunk tag is its type (slot) number.
- UPVH ('AWHF'): node count + 1 u16 at +0x3C, header +0x20 = transforms << 16 | locators, +0x24 = wheel instance
  matrices << 16 | collision parts. Nodes of 32 bytes from +0x50 (16 x u16; +0x40 is half a root record): [0] node
  type, [1] part id, [6] subtree size (depth-first order), [8] 1 = moving part, [9] model handle ((pack page & 63) << 10 | container, 0xFFFF = none),
  [10] PDL collision record, [11] locator, [12] transform (DPL reads the handle at node +0x12 in 0x59D1EB, the locator at +0x16 in 0x5B3CFA).
  Then the transforms (32 bytes: position x, y, z, 1, then rotation), the collision parts (0x50 bytes), the locators
  (DPL 32 bytes: position x, y, z, 1 and a quaternion; Driv3r 16 bytes: position only), the instance matrices and
  the PDL collision block (same format in both games; header word 6 = its byte size):
  u32 record count, part count, record table offset (0x20), part offset; then the magic 'PDL001.002.003a'; records of
  16 bytes (u32 part count, u32 offset, f32 bounding radius, 0); parts of 0x60 bytes (half size x y z, 3 rotation
  rows, centre, u32 shape code + 3 floats). A record's parts follow the previous record's. Record offsets count the
  first part of a record as 0x50 bytes and each further one as 0x60. Record 0 = the body (part 0 = its bounding box,
  shape 0); node word 10 = record index of that node's part. Shape code: low byte 0 = box, 1 = cylinder (wheel:
  half x = radius, half y = half width); high byte = surface, which differs per game.
"""
import mmap
import struct
from dataclasses import dataclass

import dplchunk as d

DVO3 = b'DVO3'
FLOAT_MARKER = 0xAAAA
UPVH_NODE_COUNT = 0x3C
UPVH_COUNTS = 0x20
UPVH_NODES = 0x50
UPVH_NODE_SIZE = 32
UPVH_TRANSFORM_SIZE = 32
UPVH_COLLISION_SIZE = 0x50
DPL_LOCATOR_SIZE = 32
D3_LOCATOR_SIZE = 16
UPVH_INSTANCE_SIZE = 64
UPVH_PDL_SIZE = 24          # header word 6: byte size of the PDL block
PDL_MAGIC = b'PDL001.002.003a'
PDL_PART_SIZE = 0x60
PDL_OFFSET_SHORT = 0x10     # record offsets count each record's first part 0x10 bytes short
NO_HANDLE = 0xFFFF
MDPC_CONTAINER_SIZE = 0x188
MDPC_LOD_ENTRIES = 0xA8
MDPC_LOD_SIZE = 32
LOD_COUNT = 7
MAX_DEPTH = 3


@dataclass
class HandlingRecord:
    key: int
    marker: int
    raw: int

    @property
    def value(self) -> float:
        return struct.unpack('<f', struct.pack('<I', self.raw))[0]


@dataclass
class HierarchyNode:
    handle: int
    node_type: int
    part: int
    moving: bool
    locator: int
    transform: int
    subtree: int        # nodes in its subtree, itself included: the nodes after it up to that count are its children
    record: int         # PDL collision record of its part (0xFFFF = none)


def find_all(data: bytes | mmap.mmap, tag: str, base: int = 0, depth: int = 0) -> list[d.Entry]:
    """Every chunk with this tag, searching nested containers."""
    found = []

    for entry in d.entries(data, base):
        if entry.tag == tag:
            found.append(entry)

        if depth < MAX_DEPTH and data[entry.offset:entry.offset + 4] == b'CHNK':
            found.extend(find_all(data, tag, entry.offset, depth + 1))

    return found


def read_dvo3(data: bytes | mmap.mmap, offset: int) -> list[HandlingRecord]:
    """The records of the DVO3 block at offset."""
    if data[offset:offset + 4] != DVO3:
        raise ValueError(f'no DVO3 block at {offset:#x}')

    count = struct.unpack_from('<I', data, offset + 8)[0]
    return [HandlingRecord(*struct.unpack_from('<HHI', data, offset + 12 + i * 8)) for i in range(count)]


def bo3_blocks(data: bytes) -> dict[int, list[HandlingRecord]]:
    """Driv3r BigVO3 file: {vehicle type: records}."""
    if data[:4] != b'BVO3':
        raise ValueError('not a BVO3 file')

    table, base, slots = struct.unpack_from('<3I', data, 16)
    found = {}

    for vehicle_type in range(slots):
        slot = struct.unpack_from('<H', data, table + vehicle_type * 2)[0]

        if slot != NO_HANDLE:
            found[vehicle_type] = read_dvo3(data, base + slot)

    return found


def hierarchy_nodes(data: bytes | mmap.mmap, offset: int) -> list[HierarchyNode]:
    """The nodes of the UPVH chunk at offset."""
    count = struct.unpack_from('<H', data, offset + UPVH_NODE_COUNT)[0]
    nodes = []

    for i in range(count - 1):
        w = struct.unpack_from('<16H', data, offset + UPVH_NODES + i * UPVH_NODE_SIZE)
        nodes.append(HierarchyNode(w[9], w[0], w[1], w[8] == 1, w[11], w[12], w[6], w[10]))

    return nodes


def transform_table(data: bytes | bytearray | mmap.mmap, offset: int) -> int:
    """Offset of the first UPVH transform (right after the nodes)."""
    count = struct.unpack_from('<H', data, offset + UPVH_NODE_COUNT)[0]
    return offset + UPVH_NODES + (count - 1) * UPVH_NODE_SIZE


def locator_table(data: bytes | bytearray | mmap.mmap, offset: int) -> int:
    """Offset of the first UPVH locator: after the transforms and the collision parts."""
    counts = struct.unpack_from('<I', data, offset + UPVH_COUNTS)[0]
    collisions = struct.unpack_from('<H', data, offset + UPVH_COUNTS + 4)[0]
    return transform_table(data, offset) + (counts >> 16) * UPVH_TRANSFORM_SIZE + collisions * UPVH_COLLISION_SIZE


def pdl_block(data: bytes | bytearray | mmap.mmap, offset: int, locator_size: int) -> int:
    """Offset of the PDL collision block: after the locators and the wheel instance matrices."""
    counts, instances = struct.unpack_from('<II', data, offset + UPVH_COUNTS)
    at = locator_table(data, offset) + (counts & 0xFFFF) * locator_size + (instances >> 16) * UPVH_INSTANCE_SIZE

    if data[at + 16:at + 16 + len(PDL_MAGIC)] != PDL_MAGIC:
        raise ValueError(f'no PDL block at {at:#x}')

    return at


def mdpc_lod_kinds(data: bytes | mmap.mmap, offset: int) -> list[list[tuple[int, int]]]:
    """Driv3r MDPC pack: per container the (kind, group count) of its 7 LOD entries."""
    n_containers, c_off = struct.unpack_from('<2I', data, offset + 8)
    result = []

    for i in range(n_containers):
        o = offset + c_off + i * MDPC_CONTAINER_SIZE + MDPC_LOD_ENTRIES
        entries = [struct.unpack_from('<6I', data, o + k * MDPC_LOD_SIZE) for k in range(LOD_COUNT)]
        result.append([(w[5], w[2]) for w in entries])

    return result
