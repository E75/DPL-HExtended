"""Driv3r's movable props (cones, bins, barriers, benches) as DPL props that topple and break when hit.

Driv3r: the Global Region's `GEPD` / `HRDT` "Prop hierarchy information" lists its props, one block per prop
('AWHF', ..., u32 model handle (page << 10 | container) at +0x14, part count at +0x10, two floats at +0x1C and +0x20,
a u32 at +0x24, parts after). Every other model is static. +0x20 reads as the mass in tonnes (cone and chair 0.001, sign
0.01, hydrant 0.4, concrete bollard 0.8; not proven in the exe).

DPL (exe addresses of the GOG build; see docs/D3-TO-DPL-CONVERSION.md, props):
- `PROP` of the global region (one per page; its +0 is the page): u32 page, u32 record count, then offsets of the
  0x60-byte records (sorted by u16 container at +0x0C), of 0x50-byte physics nodes, of 16-byte items and of 16-byte
  boxes. A record (0x5310EA, 0x530FB1, 0x533856): u8 flags (bit 0: it moves when hit, 2: only on a special hit,
  3: also on shots, 4: no body), u8 mass class (0: 10 kg, 1: 800 kg, 2: 10 t, 3: its own value at +0x5C), u8 threshold
  class (0: reacts above 0.3, 1: to any hit), u8 kind (0: topples, 1-2: breaks into pieces), u8 hit mode (0: a body at
  once, 1-2: bends first), running first indices into the other tables, a body matrix at +0x10 (rotation, minus the
  centre of mass in row 3) and the inertia per kg at +0x50 (box: (y² + z²) / 12, ...). Counts follow from the next
  record's first indices, so records cannot be moved or copied, only rewritten where they are.
- A GEBI instance becomes a prop when hit (0x533952) if its kind lies in [end of kind 2, end of kind 4) (0x53125F; NYC:
  every prop instance is kind 3) and `PROP` has a record for its container (0x531990); the static instance is then
  hidden (the region's instance bitset) and a body follows the record. The draw stays the container's own (NYC's prop
  containers all have one group; the nodes are physics only).
- The moving body collides with one record of the global `GEPM` (id 0): the record's u16 at +0x0E is that record's index
  plus one (0x534531 -> 0x531cb7; NYC: every prop instance's broadphase object is that record).
- So a Driv3r prop goes into a NYC prop's container slot whose NYC model has the nearest size and that slot's record
  is rewritten from the Driv3r prop (write_records); no other Driv3r model may take a NYC prop slot.
"""
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np

import d3city
import dplchunk as d

HRDT_MAGIC = b'AWHF'
HRDT_HANDLE, HRDT_MASS = 0x14, 0x20
PROP_HEADER = 0x20
PROP_RECORD = 0x60
PROP_KEY, PROP_SHAPE, PROP_BODY, PROP_INERTIA, PROP_MASS = 0x0C, 0x0E, 0x10, 0x50, 0x5C
MOVES = 1               # PROP record flag: the instance turns into a moving prop when hit
ON_SHOTS = 8            # PROP record flag: shots move it too (most of NYC's props)
OWN_MASS = 3            # PROP mass class: the value at PROP_MASS
NO_INDEX = 0xFFFF
SIZE_PAD = 0.05         # metres added to every extent before comparing (flat props)
NODE_COST = 0.05        # per physics node past two: simple NYC templates first
MIN_MASS = 10.0         # kg, DPL's lightest class; Driv3r gives tables and parasols 0
LIGHT = 0.1             # t: lighter props react to any hit (threshold class 1), heavier ones above DPL's 0.3


def prop_models(d3c: Path) -> dict[int, float]:
    """Model handles of Driv3r's props (the Global Region's first HRDT) -> mass in tonnes."""
    tree = d.read_tree(d3c.read_bytes(), 0, 3)
    region = next(top for top in tree.chunks if top.tag == b'GESR')
    hrdt = next(bytes(h.data) for c in region.tree.chunks if c.tag == b'GEPD' for h in c.tree.chunks if h.tag == b'HRDT')
    count = struct.unpack_from('<I', hrdt, 0)[0]
    offsets = struct.unpack_from(f'<{count}I', hrdt, 4)
    return {struct.unpack_from('<I', hrdt, o + HRDT_HANDLE)[0]: struct.unpack_from('<f', hrdt, o + HRDT_MASS)[0]
            for o in offsets if hrdt[o:o + 4] == HRDT_MAGIC}


def templates(tree: d.ChunkTree) -> tuple[dict[int, int], set[int]]:
    """NYC's global PROP: (physics node count per container that moves when hit, every container with a record)."""
    prop = bytes(next(c for top in tree.chunks if top.tree and top.tag == b'GEGR' for c in top.tree.chunks
                      if c.tag == b'PROP').data)
    count, first, nodes_at, items_at = struct.unpack_from('<4I', prop, 4)
    records = [prop[first + PROP_RECORD * i:first + PROP_RECORD * (i + 1)] for i in range(count)]
    starts = [struct.unpack_from('<H', r, PROP_SHAPE)[0] for r in records]
    total = (items_at - nodes_at) // 0x50
    moving = {}

    for i, r in enumerate(records):
        nxt = next((s for s in starts[i + 1:] if s != NO_INDEX), total)
        if r[0] & MOVES and starts[i] != NO_INDEX:
            moving[struct.unpack_from('<H', r, PROP_KEY)[0]] = nxt - starts[i]

    return moving, {struct.unpack_from('<H', r, PROP_KEY)[0] for r in records}


def _extents(points: np.ndarray) -> np.ndarray:
    return np.sort(points.max(axis=0) - points.min(axis=0)) + SIZE_PAD


def _points(page: d3city.Page, container: int) -> np.ndarray | None:
    baked = d3city.bake_instance(page, container, d3city.Instance((0.0, 0.0, 0.0), 0, (0.0, 0.0, 0.0, 1.0)), all_near=True)
    return None if baked is None else baked.vertices[:, :3]


def assign(page: d3city.Page, models: dict[int, float], slots: dict[int, int], tree: d.ChunkTree) -> dict[int, int]:
    """Driv3r global prop container -> NYC prop slot (slots: NYC slot -> node count), nearest sorted extents first.
    models: prop handles; those of other pages stay static (their page is a super region pack)."""
    pack = d.load_pack(bytes(d.find_pack(tree, 0).data), 0)
    nyc = {s: _extents(np.array(pack.containers[s].box)[:, :3]) for s in slots}
    ours = {}

    for c in (h for h in models if h >> 10 == 0 and h < len(page.containers)):   # global props only
        points = _points(page, c)
        if points is not None:
            ours[c] = _extents(points)

    pairs = sorted((sum(abs(math.log(a / b)) for a, b in zip(ours[c], nyc[s])) + NODE_COST * max(0, slots[s] - 2), c, s)
                   for c in ours for s in nyc)
    result: dict[int, int] = {}

    for _, c, s in pairs:
        if c not in result and s not in result.values():
            result[c] = s

    return result


def write_records(tree: d.ChunkTree, page: d3city.Page, props: dict[int, int], masses: dict[int, float],
                  shapes: Counter) -> int:
    """Rewrites the PROP record of each used prop slot from its Driv3r prop (props: Driv3r container -> slot): its own
    collision record in the global GEPM (shapes: (slot, record) per prop instance, the most common wins; NYC's
    records are gone from there), mass, centre of mass and box inertia; it topples at once when hit and reacts to
    any hit when light. The table indices stay NYC's. Returns the number of records written."""
    chunk = next(c for top in tree.chunks if top.tree and top.tag == b'GEGR' for c in top.tree.chunks if c.tag == b'PROP')
    prop = bytearray(chunk.data)
    count, first = struct.unpack_from('<2I', prop, 4)
    at = {struct.unpack_from('<H', prop, first + PROP_RECORD * i + PROP_KEY)[0]: first + PROP_RECORD * i for i in range(count)}
    record = {}

    for (slot, code), _ in shapes.most_common():
        record.setdefault(slot, code)

    for container, slot in props.items():
        points = _points(page, container)
        if slot not in record or points is None:
            continue
        o = at[slot]
        lo, hi = points.min(axis=0), points.max(axis=0)
        x, y, z = hi - lo
        body = np.eye(4)
        body[3, :3] = -(lo + hi) / 2
        mass = masses[container]
        prop[o:o + 5] = bytes((prop[o] & 0x10 | MOVES | ON_SHOTS, OWN_MASS, 1 if mass <= LIGHT else 0, 0, 0))
        struct.pack_into('<H', prop, o + PROP_SHAPE, record[slot] + 1)
        struct.pack_into('<16f', prop, o + PROP_BODY, *body.ravel())
        struct.pack_into('<4f', prop, o + PROP_INERTIA, (y * y + z * z) / 12, (x * x + z * z) / 12, (x * x + y * y) / 12,
                         max(MIN_MASS, mass * 1000))

    chunk.data = bytes(prop)
    return sum(1 for slot in props.values() if slot in record)
