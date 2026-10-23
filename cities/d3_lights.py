"""Driv3r traffic lights for a converted DPL city: which road ends stop for which phase, and the lamp records DPL draws.

Driv3r (AIEX v28 in GERD, see d3_ai; found from the data and the loader 0x402A10 / 0x403820 of the Driv3r exe):
- Table 5, junctions with traffic lights, 16 bytes: u32 phase count (2-4), i32 offset of a per-phase list, 2 x u32 0.
- Table 7, light heads, 64 bytes: a 4 x 4 matrix of rows (x axis, y axis, z axis, position) whose fourth column holds
  i32 +0x0C (head layout in table 6), i32 +0x1C (its junction in table 5) as offsets from the row start, and i32 +0x2C
  the head's phase (0x7FFFFFF8: none). Every row is a placed head (global containers 0x78 vertical, 0x98 horizontal,
  matched on position); the lens faces the head's +z.

DPL (exe addresses of the GOG build; see docs/D3-TO-DPL-CONVERSION.md, traffic lights):
- A road end's light (AIEX road +0x1C..+0x21, dpl_ai `light`): slot, base and program. Program k is h44 item k
  (u32 entry count, offset into h88); its entries in h88 are phases, one at a time green (1, then 2 = amber), the others
  red (0) (0x5C32EB). Entry `base` is the pedestrians' phase (they cross on it, 0x444A1B); the roads' phases follow it,
  slot = base + 1 + phase. NYC's five programs (3-7 entries) stay as they are, so every junction with as many phases
  cycles in step. Traffic stops on red and on amber further than clamp(0.5 v, 2, 18) m from the line.
- The lamps are drawn by code (0x4DA89C): per frame the type-5 attractor records within 60 m (at most 20) light one
  quad each, facing their normal, in red 0.39 m above, amber at, or green below the record, unless the instance they
  name is destroyed. Type-5 records live in the region's ATRD (v3): u32 0x20 (list table offset), u32 16 (lists),
  i16 column, i16 row, f32 min x, min z, size x, size z, u32 27, then 17 u32 list offsets (the 16 lists end to end).
  A list is an implicit kd tree: median at n // 2, split on the wider of x and z (flag bit 0: z), leaves x.
- Type-5 record, 18 bytes: u16 x, u16 packed, u16 z (x = ((packed & 0x3000) << 4 | u16 x) / 22 - 5760, y = ((packed &
  0xFFF) - 0x514) / 13, z likewise with packed bits 14-15; 0x44DF5D), u8 flags (2 | axis), u8 0x14, u16 normal
  (0x57C718), u8 column, u8 row, u16 the head's instance index in that region's GEBI, u16 program, u16 slot.
- Street lights are type-4 attractors (list 4, light handler 0x53FEA2: the nearest 20 within 60 m light the scene at
  night): the same record with byte 7 = 0x10 (type << 2), w14 an RGB565 colour (0x44E174; NYC: mostly 0xFFFF) and w16
  0xFFFF (NYC's most common). NYC puts them on its lamp posts' heads (post 0x35: 8.1 m up, 2.7 m out). Driv3r's lamp
  lights glowed on lawns and lit only the ground round the player): one light per lens, at the centre of its
  vertices (clusters 0.5 m apart).
"""
import math
import struct
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import d3city
import dpl_ai
import dplchunk as d

JUNCTIONS, HEADS = 5, 7
HEAD_SIZE, JUNCTION_SIZE = 64, 16
NO_PHASE = 0x7FFFFFF8
# middle lens of each head model, in its own space: the lens plate is at z 0 and its visors reach z 0.18, so the quad
# sits just in front of them (NYC's records stand 9 cm in front of their heads)
LAMPS = {0x78: (0.0, 0.5, 0.2), 0x98: (0.0, 0.15, 0.2),
         # a second head on an arm, the drivers' one; only head positions are looked up, so the ids may mean other
         # models in another city
         0x80: (-1.07, 0.8, 0.38), 0x81: (0.0, -0.71, 0.38), 0x82: (0.0, 0.21, 0.22),
         # and a pedestrian head at y 0.9-1.4, the only one with lit-lens geometry
         0x3B: (0.0, 3.09, 0.48)}
# 0x4DA89C stacks red, amber and green vertically: a lying head (0x98, lenses side by side) would glow under its housing
UPRIGHT_HEADS = {0x78, 0x80, 0x81, 0x82, 0x3B}
# heads not in LAMPS: their lit lenses use this shader type, facing +z; the lamp sits in front of the top head's middle
LIGHT_SHADER = 0x1E
LAMP_FRONT = 0.2        # metres in front of the lens plate, as LAMPS
HEAD_GAP = 0.3          # metres of height between two heads' lenses (one head's lenses lie 0.25 m apart)
# 0x4DA89C leans its quad 29 deg back from the normal (corner z -0.1 at the top, +0.1 at the bottom): a level normal
# shows a thin slanted glow from below, a normal pitched down this far faces the drivers
GLOW_PITCH = math.radians(45)
FACING = 0.7            # a head serves a road end whose arriving traffic it faces this squarely (cos of the angle)
CROSSING_REACH = 30.0   # road ends this near each other meet at one crossing
ACROSS = 0.3           # ends whose directions meet at a cos below this cross each other
HEAD_REACH = 35.0       # heads this near a road end can serve it (Miami: the facing head is 3.6 m away, p90 7.6 m)
HEAD_GROUP = 5.0        # heads this much further than the nearest facing one vote along
LOOSE_GROUP = 60.0      # heads on no junction this near each other share one crossing
PEDESTRIAN_PHASES = 1  # DPL programs have one entry more than road phases: the pedestrians' (entry 0)
LAMP_LIST, LAMP_SIZE = 5, 18
LIGHT_LIST, OTHER_LIGHT_LIST = 4, 14     # type 14: NYC lights without an instance, nothing of Driv3r's goes there
LAMP_FLAGS = 2
# model (street furniture), high up: super region models use the type for lit windows and glow patches on the ground
LENS_SHADER_TYPE = 0x22
LENS_MIN_HEIGHT = 2.5   # metres above the model's origin (lamp heads 2.9-11.7 m; ground glows and low signs below)
LENS_GAP = 0.5          # lens vertices this near each other belong to one lamp
WHITE, NO_GROUP = 0xFFFF, 0xFFFF
POS_BIAS, POS_STEP, HEIGHT_BIAS, HEIGHT_STEP = 0x1EF00, 22, 0x514, 13
NORMAL_STEPS = 89


@dataclass
class Head:
    position: np.ndarray
    axes: np.ndarray          # rows: x, y, z axis (z: the lens normal)
    junction: int | None
    phase: int | None


def position_key(p: np.ndarray | tuple[float, float, float]) -> tuple[float, float, float]:
    """A head's position as a lookup key: the AIEX row and the instance hold the same floats."""
    return tuple(round(float(v), 2) for v in p)


def _table(data: bytes, k: int) -> tuple[int, int]:
    count, offset = struct.unpack_from('<II', data, 8 * k)
    return count, 8 * k + offset


def read_heads(aiex: bytes) -> tuple[list[Head], list[int]]:
    """The light heads of a Driv3r AIEX v28 chunk and the phase count of each light junction."""
    junction_count, junction_at = _table(aiex, JUNCTIONS)
    head_count, head_at = _table(aiex, HEADS)
    phases = [struct.unpack_from('<I', aiex, junction_at + JUNCTION_SIZE * j)[0] for j in range(junction_count)]
    heads = []

    for i in range(head_count):
        at = head_at + HEAD_SIZE * i
        m = np.array(struct.unpack_from('<16f', aiex, at), dtype=np.float64).reshape(4, 4)
        junction, phase = struct.unpack_from('<i', aiex, at + 0x1C)[0], struct.unpack_from('<i', aiex, at + 0x2C)[0]
        junction = (at + junction - junction_at) // JUNCTION_SIZE if junction else None
        heads.append(Head(m[3, :3], m[:3, :3], junction, None if phase == NO_PHASE else phase))

    _adopt_loose_heads(heads, phases)
    return heads, phases


def _adopt_loose_heads(heads: list[Head], phases: list[int]) -> None:
    """Heads on no junction (Miami: 15, at four crossings) would stay dark and their roads unlit: each group of them
    within LOOSE_GROUP becomes a two-phase junction of its own, heads along the first head's facing in phase 0, the
    others in phase 1."""
    loose = [h for h in heads if h.junction is None or h.phase is None]

    while loose:
        first = loose[0]
        group = [h for h in loose if np.linalg.norm((h.position - first.position)[[0, 2]]) <= LOOSE_GROUP]
        phases.append(2)

        for h in group:
            h.junction, h.phase = len(phases) - 1, 0 if abs(h.axes[2] @ first.axes[2]) >= 0.5 else 1

        taken = {id(h) for h in group}
        loose = [h for h in loose if id(h) not in taken]


def programs(ai: dpl_ai.AiExport) -> list[tuple[int, int]]:
    """(entry count, first h88 entry) of DPL's light programs (h44)."""
    h44 = ai.tables['h44'].reshape(-1, 4)
    return [(int(count), int(offset) // 4) for count, offset, _, _ in h44]


def assign_lights(ai: dpl_ai.AiExport, roads: list[dpl_ai.NewRoad], first: int, heads: list[Head],
                  phases: list[int]) -> Counter:
    """Sets the light of every road end that arrives at a Driv3r light junction: the phase of the heads facing its
    traffic, on the DPL program with that junction's phase count. Roads are the NewRoads at network-1 index first..
    The heads' phases are set to those of the ends they serve, for lamps()."""
    by_phases = {count - PEDESTRIAN_PHASES: (k, base) for k, (count, base) in enumerate(programs(ai))}
    lit = [h for h in heads if h.junction is not None and h.phase is not None]
    points = np.array([h.position[[0, 2]] for h in lit])
    normals = np.array([h.axes[2, [0, 2]] / (np.linalg.norm(h.axes[2, [0, 2]]) + 1e-9) for h in lit])
    light = ai.tables['roads']['light']
    counts: Counter = Counter()
    ends: list[tuple[int, int, int, np.ndarray, int | None]] = []     # road, end, junction, arriving direction, phase

    # one Driv3r junction may control several crossings (heads up to 300 m from their centre): match the heads near the
    # end, the nearest facing head and those up to HEAD_GROUP further
    for i, road in enumerate(roads):
        for e in (0, 1):
            if not road.lanes[1 - e]:   # no traffic arrives at this end
                continue

            point, direction = road.end(e)
            arriving = direction if e else -direction
            distance = np.linalg.norm(points - point[[0, 2]], axis=1)
            near = np.nonzero(distance <= HEAD_REACH)[0]

            if not len(near):
                continue

            facing = normals[near] @ -arriving
            ahead = sorted((distance[k], k) for k, f in zip(near, facing) if f >= FACING)

            if ahead:
                group = [lit[k] for dist, k in ahead if dist <= ahead[0][0] + HEAD_GROUP]
                j = Counter(h.junction for h in group).most_common(1)[0][0]
                phase = Counter(h.phase for h in group if h.junction == j).most_common(1)[0][0]
                ends.append((i, e, j, arriving, phase))
                counts['ends facing a head'] += 1
            else:   # a head at 60 degrees serves the cross street: never take its phase
                ends.append((i, e, lit[near[int(np.argmin(distance[near]))]].junction, arriving, None))

    # an end no head faces (a road through a junction whose heads serve only the cross street) shares the phase of an
    # end it runs along, or with two phases takes the other one than an end crossing it
    for k, (i, e, j, arriving, phase) in enumerate(ends):
        if phase is not None:
            continue

        point = roads[i].end(e)[0][[0, 2]]
        same = [(a, p) for o, f, other, a, p in ends if other == j and p is not None
                and np.linalg.norm(roads[o].end(f)[0][[0, 2]] - point) < dpl_ai.JUNCTION]
        along = [p for a, p in same if abs(a @ arriving) >= FACING]
        across = [p for a, p in same if abs(a @ arriving) < ACROSS]

        if along:
            ends[k] = (i, e, j, arriving, Counter(along).most_common(1)[0][0])
            counts['ends along a lit end'] += 1
        elif across and phases[j] == 2:
            ends[k] = (i, e, j, arriving, 1 - Counter(across).most_common(1)[0][0])
            counts['ends across a lit end'] += 1
        else:
            counts['ends near a light junction, unlit'] += 1

    _split_crossing(roads, ends, phases, counts)

    # each head shows the phase of the nearest end it faces, so lamps and stop lines agree after the fixes above
    starts = np.array([roads[i].end(e)[0][[0, 2]] for i, e, *_ in ends]).reshape(-1, 2)

    for head, point, normal in zip(lit, points, normals):
        served = sorted((float(np.linalg.norm(starts[n] - point)), n) for n, (_, _, j, a, p) in enumerate(ends)
                        if j == head.junction and p is not None and normal @ -a >= FACING
                        and np.linalg.norm(starts[n] - point) <= HEAD_REACH)

        if served and ends[served[0][1]][4] != head.phase:
            head.phase = ends[served[0][1]][4]
            counts['heads re-phased with their end'] += 1

    for i, e, j, _, phase in ends:
        if phase is None:
            continue

        if phases[j] not in by_phases or phase >= phases[j]:
            counts[f'junctions of {phases[j]} phases: no DPL program'] += 1
            continue

        program, base = by_phases[phases[j]]
        light[first + i, :, e] = (base + PEDESTRIAN_PHASES + phase, base, program)

    return counts


def _split_crossing(roads: list[dpl_ai.NewRoad], ends: list[tuple[int, int, int, np.ndarray, int | None]],
                    phases: list[int], counts: Counter) -> None:
    """At a crossing of two-phase junctions where crossing ends share a phase (Driv3r gives all heads of 6 Miami
    junctions phase 0, and two of its junctions can meet at one crossing: DPL runs both on one program), the ends along
    the first phase-0 end keep phase 0 and the ends across it take phase 1: DPL's traffic would meet in the middle."""
    points = [roads[i].end(e)[0][[0, 2]] for i, e, *_ in ends]

    for k, (_, _, j, arriving, phase) in enumerate(ends):
        if phase is None or phases[j] != 2:
            continue

        crossing = [n for n, (_, _, o, a, p) in enumerate(ends) if phases[o] == 2 and p is not None
                    and np.linalg.norm(points[n] - points[k]) < CROSSING_REACH]

        if not any(ends[a][4] == ends[b][4] and abs(ends[a][3] @ ends[b][3]) < 0.5 for a in crossing for b in crossing):
            continue

        axis = next((ends[n][3] for n in crossing if ends[n][4] == 0), ends[crossing[0]][3])

        for n in crossing:
            i, e, o, a, p = ends[n]
            new = 0 if abs(a @ axis) >= 0.5 else 1
            counts['ends re-phased at a crossing'] += new != p
            ends[n] = (i, e, o, a, new)


def pack_normal(n: np.ndarray) -> int:
    """DPL's 16-bit unit vector (encoder 0x57C718, decoder 0x57C7F0)."""
    ax, ay, az = np.abs(n)
    inverse = 1.0 / (ax + ay + az)
    rest = 1.0 - inverse * ay
    i = int(rest * NORMAL_STEPS)
    j = int(inverse * az / rest * (i * 2)) if rest >= 1e-5 else 0
    return i * i + j | ((n[0] >= 0) << 2 | (n[1] >= 0) << 1 | int(n[2] >= 0)) << 13


def unpack_normal(w: int) -> np.ndarray:
    u, signs = w & 0x1FFF, w >> 13
    i = math.isqrt(u)
    f = i / NORMAL_STEPS

    if i:
        share = (u - i * i) / (i * 2)
        v = np.array([(1 - share) * f, 1 - f, share * f])
    else:
        v = np.array([0.0, 1.0, 0.0])

    v *= [(signs >> 2) * 2 - 1, (signs >> 1 & 1) * 2 - 1, (signs & 1) * 2 - 1]
    return v / np.linalg.norm(v)


def pack_position(p: np.ndarray) -> tuple[int, int, int]:
    x, z = (round(v * POS_STEP) + POS_BIAS for v in (p[0], p[2]))
    y = round(p[1] * HEIGHT_STEP) + HEIGHT_BIAS

    if not (0 <= x < 1 << 18 and 0 <= z < 1 << 18 and 0 <= y < 1 << 12):
        raise ValueError(f'lamp at {p} out of range')

    return x & 0xFFFF, (x >> 16) << 12 | (z >> 16) << 14 | y, z & 0xFFFF


def unpack_position(r: bytes) -> np.ndarray:
    a, w, b = struct.unpack_from('<3H', r, 0)
    return np.array([(((w & 0x3000) << 4 | a) - POS_BIAS) / POS_STEP, ((w & 0xFFF) - HEIGHT_BIAS) / HEIGHT_STEP,
                     (((w & 0xC000) << 2 | b) - POS_BIAS) / POS_STEP])


@dataclass
class Lamp:
    position: np.ndarray
    normal: np.ndarray
    region: tuple[int, int]
    instance: int
    kind: int                 # attractor list: 5 traffic lamp, 4 street light
    word14: int               # traffic lamp: program; street light: RGB565 colour
    word16: int               # traffic lamp: slot


def head_lamp(page: d3city.Page, container: int) -> tuple[np.ndarray, bool] | None:
    """A head model's lamp in its own space and whether its lenses stand upright: LAMPS where measured, else in front
    of the middle of its lit lenses (shader LIGHT_SHADER facing +z), upright when they are taller than wide. None: no
    """
    if container in LAMPS:
        return np.array(LAMPS[container]), container in UPRIGHT_HEADS

    points = []

    for group in page.detail_groups(container, all_near=True):
        matrix = np.array(group.matrix, dtype=np.float64).reshape(4, 4)

        for mesh in group.meshes:
            material = page.material(mesh.material) if mesh.material_page in (0, page.id) else None

            if material is not None and material.flags[0] == LIGHT_SHADER:
                used = np.unique(page.mesh_triangles(mesh))
                if (page.vertices[used, 3:6].astype(np.float64) @ matrix[:3, :3].T)[:, 2].mean() > 0:
                    points.append(page.vertices[used, :3].astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3])

    if not points:
        return None

    # and 1.3-1.8)
    v = np.concatenate(points)
    ys = np.sort(v[:, 1])
    gaps = np.nonzero(np.diff(ys) > HEAD_GAP)[0]
    v = v[v[:, 1] >= (ys[gaps[-1] + 1] if len(gaps) else ys[0])]
    low, high = v.min(axis=0), v.max(axis=0)
    return np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2] + LAMP_FRONT]), \
        bool(high[1] - low[1] > high[0] - low[0])


def head_models(page: d3city.Page, heads: list[Head], instances: list[d3city.Instance]) -> dict[int, tuple[np.ndarray, bool]]:
    """The global models standing at the heads' positions that have a lamp (head_lamp) -> (lamp, upright)."""
    keys = {position_key(h.position) for h in heads}
    containers = {i.handle for i in instances if i.handle >> 10 == 0 and position_key(i.position) in keys}
    return {c: lamp for c in containers if (lamp := head_lamp(page, c)) is not None}


def lamps(heads: list[Head], phases: list[int], ai: dpl_ai.AiExport,
          placed: dict[tuple[float, float, float], tuple[int, tuple[int, int], int]],
          models: dict[int, tuple[np.ndarray, bool]]) -> tuple[list[Lamp], Counter]:
    """One lamp per head on a light junction: at the middle lens of its model, lit by its phase's slot. placed: the
    head instances by their Driv3r position -> (Driv3r container, DPL region, instance index in that region); models:
    head_models."""
    by_phases = {count - PEDESTRIAN_PHASES: (k, base) for k, (count, base) in enumerate(programs(ai))}
    result, counts = [], Counter()

    for head in heads:
        where = placed.get(position_key(head.position))

        if head.junction is None or head.phase is None:
            counts['heads without a phase'] += 1
        elif where is None:
            counts['heads without a placed instance'] += 1
        elif phases[head.junction] not in by_phases:
            counts['heads on a junction without a DPL program'] += 1
        elif not models[where[0]][1]:
            counts['lying heads, unlit'] += 1
        else:
            container, region, instance = where
            program, base = by_phases[phases[head.junction]]
            level = head.axes[2] * [1, 0, 1] / np.linalg.norm(head.axes[2, [0, 2]])
            normal = level * math.cos(GLOW_PITCH) - np.array([0, math.sin(GLOW_PITCH), 0])
            result.append(Lamp(head.position + models[container][0] @ head.axes, normal, region, instance,
                               LAMP_LIST, program, base + PEDESTRIAN_PHASES + head.phase))
            counts['lamps'] += 1

    return result, counts


def _kd(records: list[tuple[np.ndarray, bytes]], flags: int = LAMP_FLAGS) -> list[bytes]:
    """Records (position, record without its flag byte) in DPL's implicit kd order, flag byte set: bit 0 the split
    axis, the other bits the list's own (flags)."""
    if not records:
        return []

    if len(records) == 1:
        return [records[0][1][:6] + bytes([flags]) + records[0][1][6:]]

    spread = np.ptp(np.array([p for p, _ in records]), axis=0)
    axis = int(spread[2] > spread[0])
    records = sorted(records, key=lambda r: r[0][2 if axis else 0])
    mid = len(records) // 2
    pivot = records[mid][1]
    return (_kd(records[:mid], flags) + [pivot[:6] + bytes([flags | axis]) + pivot[6:]]
            + _kd(records[mid + 1:], flags))


def point_list(points: list[np.ndarray], kind: int) -> bytes:
    """A list of bare positions (8-byte records, e.g. list 10: player recovery points) in kd order."""
    records = []

    for p in points:
        packed = struct.pack('<3H', *pack_position(p))
        records.append((unpack_position(packed), packed + bytes([kind << 2])))

    return b''.join(_kd(records, 0))


def lamp_list(region_lamps: list[Lamp]) -> bytes:
    records = []

    for lamp in region_lamps:
        packed = pack_position(lamp.position)
        # positions as the game decodes them, so the kd order matches what it compares
        rest = struct.pack('<B H 2B 3H', lamp.kind << 2, pack_normal(lamp.normal), *lamp.region, lamp.instance,
                           lamp.word14, lamp.word16)
        records.append((unpack_position(struct.pack('<3H', *packed)), struct.pack('<3H', *packed) + rest))

    return b''.join(_kd(records))


def replace_list(atrd: bytes, k: int, items: bytes) -> bytes:
    """The ATRD with list k replaced; the later lists move."""
    table, count = struct.unpack_from('<2I', atrd, 0)
    offsets = list(struct.unpack_from(f'<{count + 1}I', atrd, table))
    old = offsets[k + 1] - offsets[k]
    data = bytearray(atrd[:offsets[k]] + items + atrd[offsets[k + 1]:])

    for n in range(k + 1, count + 1):
        offsets[n] += len(items) - old

    struct.pack_into(f'<{count + 1}I', data, table, *offsets)
    return bytes(data)


def lens_points(page: d3city.Page, container: int) -> list[np.ndarray]:
    """The lamps of a Driv3r model in its own space: the centre of each cluster of lens vertices (shader type
    LENS_SHADER_TYPE, a mesh at least LENS_MIN_HEIGHT up)."""
    points = []

    for group in page.detail_groups(container, all_near=True):
        matrix = np.array(group.matrix, dtype=np.float64).reshape(4, 4)

        for mesh in group.meshes:
            material = page.material(mesh.material) if mesh.material_page in (0, page.id) else None

            if material is not None and material.flags[0] & 0xFF == LENS_SHADER_TYPE:
                tri = page.mesh_triangles(mesh)
                v = page.vertices[np.unique(tri), :3].astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
                if v[:, 1].min() >= LENS_MIN_HEIGHT:
                    points.append(v)

    if not points:
        return []

    v = np.concatenate(points)
    near = np.linalg.norm(v[:, None] - v[None], axis=2) <= LENS_GAP     # limit: O(n^2), a model has at most ~200 lens vertices
    label = np.arange(len(v))

    for _ in range(len(v)):     # single-link clusters: spread the smallest label until nothing changes
        merged = np.where(near, label[None, :], len(v)).min(axis=1)
        if (merged == label).all():
            break
        label = merged

    return [v[label == k].mean(axis=0) for k in np.unique(label)]


def street_lights(pages: dict[int, d3city.Page], instances: list[d3city.Instance],
                  placed: dict[tuple[float, float, float], tuple[tuple[int, int], int]],
                  fallback: Callable[[float, float], tuple[int, int] | None]) -> tuple[list[Lamp], Counter]:
    """One white light per lamp lens of every placed global Driv3r model (lens_points). placed: instance index per
    Driv3r position -> (DPL region, index); a baked instance names fallback(x, z): its region's first instance, or
    None (no light)."""
    lenses: dict[int, list[np.ndarray]] = {}
    result, counts = [], Counter()

    for inst in instances:
        page = pages.get(inst.handle >> 10) if inst.handle >> 10 == d3city.GLOBAL_PAGE else None

        if page is None or inst.handle & 1023 >= len(page.containers):
            continue

        if inst.handle not in lenses:
            lenses[inst.handle] = lens_points(page, inst.handle & 1023)

        if not lenses[inst.handle]:
            continue

        turn = d3city.quaternion_matrix(inst.rotation)
        where = placed.get(position_key(inst.position))
        where = where or ((region, 0) if (region := fallback(inst.position[0], inst.position[2])) else None)

        if where is None:
            counts['lights of instances in no region'] += len(lenses[inst.handle])
            continue

        for point in lenses[inst.handle]:
            result.append(Lamp(np.array(inst.position) + turn @ point, turn @ np.array([0.0, 0.0, -1.0]), *where,
                               LIGHT_LIST, WHITE, NO_GROUP))

        counts['street lights'] += len(lenses[inst.handle])
        counts['street lights on baked instances'] += len(lenses[inst.handle]) if position_key(inst.position) not in placed else 0

    return result, counts


def write_lamps(regions: dict[tuple[int, int], d.Chunk], all_lamps: list[Lamp]) -> Counter:
    """Every region's ATRD gets its traffic lamps as list 5 and its street lights as list 4; NYC's go, and so do its
    lights without an instance (list 14). regions: (column, row) -> the region's ATRD chunk."""
    per_region: dict[tuple[int, int, int], list[Lamp]] = defaultdict(list)
    counts: Counter = Counter()

    for lamp in all_lamps:
        per_region[(*lamp.region, lamp.kind)].append(lamp)

    for key, chunk in regions.items():
        data = bytes(chunk.data)

        for kind in (LAMP_LIST, LIGHT_LIST, OTHER_LIGHT_LIST):
            table = struct.unpack_from('<I', data, 0)[0]
            start, end = struct.unpack_from('<2I', data, table + 4 * kind)
            counts[f'NYC list {kind} records removed'] += (end - start) // LAMP_SIZE
            data = replace_list(data, kind, lamp_list(per_region.get((*key, kind), [])))
            counts[f'list {kind} records written'] += len(per_region.get((*key, kind), []))

        chunk.data = data

    counts['lamps without an ATRD'] = sum(len(v) for k, v in per_region.items() if k[:2] not in regions)
    return counts
