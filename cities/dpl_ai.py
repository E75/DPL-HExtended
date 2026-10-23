"""Reader and writer for the AI road network of Driver Parallel Lines: chunk AIEX v6 "AI Export Data" in the global region
(GEGR) of a city file. Cities\\AIExport holds the same format; the game reads it only when the chunk is missing or has
another version (loader 0x57B4CD).

Found from the data and the exe (road library around 0x5B9000-0x5C3500), see docs/D3-TO-DPL-CONVERSION.md:
- Everything is index based; on load the game only adds the chunk address to the header offsets (0x5C3370).
- Header 0xA0 bytes: two networks of 0x20 bytes at +0x00 (streets) and +0x20 (a small second network): u32 road count,
  u32 road offset, i32 grid min x, i32 grid min z (cells of 240 m), u32 grid width, u32 grid height, u32 cell count,
  u32 cell offset. +0x40 u32 item count of table +0x44. +0x44..+0x88: offsets of the tables below. Tables start on 0x20
  in this order: roads (network 2 directly after network 1), seg, piece, h54, lane, pred, conn, conflict, c, h6c, h70,
  d, a, b, cells (network 2 directly after network 1), cell_roads, h44, h88. Padding is zero.
- Position (seg, conn): u32 x and u32 z, low 24 bits = centimetres + 0x800000; height in cm + 0x8000 is split over the
  top bytes (x high byte, z low byte).
- Road 40 bytes: f32 length; u16 first a / per end; u16 first b per end; u16 first lane per direction (0xffff: none);
  u16 first seg; u16 first c; u16 first d; u8 a count per end; u8 b count per end; u8 lanes per direction; 3 x u8 per end
  (traffic light slot, base, kind; 0xff: none); u8 seg count (low 6 bits) | 2 flag bits; u8 flags (bit 2: fast road,
  traffic capped at 27 m/s instead of 18, 0x5BCC62); u8 c count; u8 d
  count; u16 8. Direction 0 runs from the road start to its end, direction 1 back.
- a: u16 road | end << 15, the road ends that meet this end in a plain node. b: the same for the road ends of a junction
  this end leads into (lanes cross it on conn curves).
- seg 16 bytes: position, f32 length, u16 first piece, u8 piece count, u8 kind (0 straight, 1, 2). piece 20 bytes:
  quaternion turning +z to the piece direction (yaw, then pitch; NYC's ramps carry their slope here), f32 length.
- lane 14 bytes: i16 offset from the centre line in cm, u16, u16 road, u8, u8, u16 first pred, u16 first conn, u8 pred
  count (low 7 bits) | flag << 7, u8 conn count. pred holds, per plain node, the next lanes: u16 lane | end << 15, the
  list written twice. conn are the curves through a junction.
- conn 24 bytes: cubic Bezier from the end of lane `from` over P1 (position) and P2 (P1 + dx, dz in cm; y in cm) to the
  start of lane `to` (0x5BE0C1); u16 length in cm (14 bits) | speed class << 14 (0x5BE06C: traffic takes the curve
  at 27 m/s for class 0 and 3, 20 for 1, 5 for 2); u32 info: conflict count (8 bits), first conflict
  (16 bits), from end, to end, 2 x 2 bits; u16 from, u16 to. conflict: u16 conn list.
- cells: u32 first << 16 | count into cell_roads (u16 road), per 240 m cell.
- c, pavements, 22 bytes (0x5C2ED8): i16 offset in cm, u16 (10), u16 road, u8 0, u8 0xff, u16 first h6c per end,
  2 x u16 h70 item per end (0xffff: none; link k of end e at +0x0c + 4k + 2e), u8 h6c count per end. h6c: the pavement
  ends that continue this one, u16 c | end << 15. h70 items of 4 bytes: u16 c | end << 15 (the owner), u16 c | end << 15:
  link 0 goes round the junction corner to the nearest pavement of another road, link 1 crosses that road to the
  pavement of a third road. d, parking strips, 8 bytes: i16 offset in cm, u16 0, u16 road, u16 0xff00.
- h54, h44 (u32 id, u32 offset into h88, 2 x 0) and h88 are kept as they are; h6c and h70 too (new items go after).
"""
import struct
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

HEADER = 0xA0
ALIGN = 0x20
NETS = 2
POS_BIAS = 0x800000
HEIGHT_BIAS = 0x8000
CELL = 240.0
NONE = 0xFFFF

ROAD = np.dtype([('length', '<f4'), ('a', '<u2', 2), ('b', '<u2', 2), ('lane', '<u2', 2), ('seg', '<u2'), ('c', '<u2'),
                 ('d', '<u2'), ('na', 'u1', 2), ('nb', 'u1', 2), ('nlane', 'u1', 2), ('light', 'u1', (3, 2)),
                 ('nseg', 'u1'), ('flags', 'u1'), ('nc', 'u1'), ('nd', 'u1'), ('x26', '<u2')])
SEG = np.dtype([('x', '<u4'), ('z', '<u4'), ('length', '<f4'), ('piece', '<u2'), ('npiece', 'u1'), ('kind', 'u1')])
PIECE = np.dtype([('q', '<f4', 4), ('length', '<f4')])
LANE = np.dtype([('offset', '<i2'), ('x02', '<u2'), ('road', '<u2'), ('x06', 'u1'), ('x07', 'u1'), ('pred', '<u2'),
                 ('conn', '<u2'), ('npred', 'u1'), ('nconn', 'u1')])
CONN = np.dtype([('x', '<u4'), ('z', '<u4'), ('dx', '<i2'), ('y', '<i2'), ('dz', '<i2'), ('length', '<u2'),
                 ('info', '<u4'), ('src', '<u2'), ('dst', '<u2')])
C_ITEM = np.dtype([('offset', '<i2'), ('x02', '<u2'), ('road', '<u2'), ('x06', 'u1'), ('x07', 'u1'), ('next', '<u2', 2),
                   ('link', '<u2', (2, 2)), ('nnext', 'u1', 2)])
D_ITEM = np.dtype([('offset', '<i2'), ('x02', '<u2'), ('road', '<u2'), ('x06', '<u2')])

# header offset field -> table name, in field order (+0x44 .. +0x88)
FIELDS = ('h44', 'roads', 'seg', 'piece', 'h54', 'lane', 'pred', 'conn', 'conflict', 'c', 'h6c', 'h70', 'd', 'a', 'b',
          'cells', 'cell_roads', 'h88')
ORDER = ('roads', 'seg', 'piece', 'h54', 'lane', 'pred', 'conn', 'conflict', 'c', 'h6c', 'h70', 'd', 'a', 'b', 'cells',
         'cell_roads', 'h44', 'h88')
RAW = {'h54', 'h6c', 'h70', 'h88'}      # unknown layout: kept byte for byte, padding included
H44_ITEM = 16


@dataclass
class Grid:
    min_x: int
    min_z: int
    width: int
    height: int


@dataclass
class AiExport:
    grids: list[Grid]
    road_counts: list[int]            # roads per network; network 2 follows network 1 in `roads`
    tables: dict[str, np.ndarray | bytes]
    h44_count: int

    def net_slice(self, net: int) -> slice:
        first = sum(self.road_counts[:net])
        return slice(first, first + self.road_counts[net])


def align(value: int) -> int:
    return (value + ALIGN - 1) // ALIGN * ALIGN


def pack_position(x: float, y: float, z: float) -> tuple[int, int]:
    """(u32 x, u32 z) of a position in metres."""
    height = round(y * 100) + HEIGHT_BIAS
    px = round(x * 100) + POS_BIAS
    pz = round(z * 100) + POS_BIAS
    if not (0 <= px < 1 << 24 and 0 <= pz < 1 << 24 and 0 <= height < 1 << 16):
        raise ValueError(f'position {x}, {y}, {z} out of range')
    return (height >> 8) << 24 | px, (height & 0xFF) << 24 | pz


def unpack_position(x: int, z: int) -> tuple[float, float, float]:
    height = (x >> 24) << 8 | z >> 24
    return ((x & 0xFFFFFF) - POS_BIAS) / 100, (height - HEIGHT_BIAS) / 100, ((z & 0xFFFFFF) - POS_BIAS) / 100


def _end(starts: np.ndarray, counts: np.ndarray) -> int:
    """Items a table needs: the largest start + count over the entries that have items."""
    used = (counts > 0) & (starts != NONE)
    return int((starts[used].astype(np.int64) + counts[used]).max()) if used.any() else 0


def read(raw: bytes) -> AiExport:
    head = struct.unpack_from('<16I', raw, 0)
    grids = [Grid(*struct.unpack_from('<2i2I', raw, 0x20 * n + 8)) for n in range(NETS)]
    road_counts = [head[0], head[8]]
    h44_count = struct.unpack_from('<I', raw, 0x40)[0]
    offset = dict(zip(FIELDS, struct.unpack_from(f'<{len(FIELDS)}I', raw, 0x44)))
    t: dict[str, np.ndarray | bytes] = {}

    def table(name: str, dtype: np.dtype | str, count: int) -> np.ndarray:
        t[name] = np.frombuffer(raw, dtype, count, offset[name]).copy()
        return t[name]

    roads = table('roads', ROAD, sum(road_counts))
    seg = table('seg', SEG, _end(roads['seg'], roads['nseg'] & 0x3F))
    table('piece', PIECE, _end(seg['piece'], seg['npiece']))
    lane = table('lane', LANE, max(_end(roads['lane'][:, k], roads['nlane'][:, k]) for k in range(2)))
    table('pred', '<u2', _end(lane['pred'], lane['npred'] & 0x7F))
    conn = table('conn', CONN, _end(lane['conn'], lane['nconn']))
    table('conflict', '<u2', _end(conn['info'] >> 8 & 0xFFFF, conn['info'] & 0xFF))
    table('c', C_ITEM, _end(roads['c'], roads['nc']))
    table('d', D_ITEM, _end(roads['d'], roads['nd']))
    table('a', '<u2', max(_end(roads['a'][:, k], roads['na'][:, k]) for k in range(2)))
    table('b', '<u2', max(_end(roads['b'][:, k], roads['nb'][:, k]) for k in range(2)))
    cells = t['cells'] = np.frombuffer(raw, '<u4', sum(g.width * g.height for g in grids), offset['cells']).copy()
    table('cell_roads', '<u2', _end(cells >> 16, cells & 0xFFFF))
    table('h44', '<u4', h44_count * H44_ITEM // 4)
    following = sorted(set(offset.values()) | {len(raw)})

    for name in RAW:
        t[name] = raw[offset[name]:next(o for o in following if o > offset[name])]

    t['h44'][1::4] -= offset['h88']     # offsets into h88 become relative
    return AiExport(grids, road_counts, t, h44_count)


def write(ai: AiExport) -> bytes:
    out = bytearray(HEADER)
    offset: dict[str, int] = {}

    for name in ORDER:
        value = ai.tables[name]
        out += bytes(align(len(out)) - len(out))
        offset[name] = len(out)
        out += value if isinstance(value, bytes) else value.tobytes()

    h44 = ai.tables['h44'].copy()     # its offsets into h88 are known only now
    h44[1::4] += offset['h88']
    out[offset['h44']:offset['h44'] + h44.nbytes] = h44.tobytes()
    first_cell = offset['cells']

    for n, grid in enumerate(ai.grids):
        cells = grid.width * grid.height
        struct.pack_into('<II2i4I', out, 0x20 * n, ai.road_counts[n], offset['roads'] + sum(ai.road_counts[:n]) * ROAD.itemsize,
                         grid.min_x, grid.min_z, grid.width, grid.height, cells, first_cell)
        first_cell += cells * 4

    struct.pack_into('<I', out, 0x40, ai.h44_count)
    struct.pack_into(f'<{len(FIELDS)}I', out, 0x44, *(offset[name] for name in FIELDS))
    return bytes(out)


def clear(ai: AiExport) -> None:
    """Removes every road of both networks (and what hangs on them); the grids and the raw tables stay."""
    for name in ('roads', 'seg', 'piece', 'lane', 'pred', 'conn', 'conflict', 'c', 'd', 'a', 'b', 'cell_roads'):
        ai.tables[name] = ai.tables[name][:0]

    ai.tables['cells'] = np.zeros_like(ai.tables['cells'])
    ai.road_counts = [0] * NETS


JOIN = 1.0              # road ends this close continue into each other (plain node)
LANE_JOIN = 2.5         # a lane end this close to a lane start of another road runs on into it (lanes lie 4 m apart)
JUNCTION = 45.0         # ends of different roads this close share a junction (Driv3r ends lie ~20 m from its centre)
JUNCTION_HEIGHT = 2.0   # ends further apart in height lie on different decks
CONTROL = 0.4           # Bezier control points at this share of the chord from the lane ends (NYC: ~0.4)
ROAD_FLAGS = 0xB8       # the most common road flags byte in NYC
FAST_ROAD = 4           # road flags bit: traffic capped at 27 m/s instead of 18
RAIL_ROAD_FLAGS = 0xAB  # the flags byte of all 13 roads of NYC's network 2 (its two train tracks)
CONN_LENGTH = 0x3FFF    # conn length bits (cm); the two bits above are its speed class
TURN_CLASS = 2          # 5 m/s: the speed class of a turn without one of its own (NYC's most common for turns)
SEG_COUNT_FLAGS = 2 << 6
ROAD_X26 = 8
PAVEMENT_X02 = 10       # the most common value in NYC
PARKING_X06 = 0xFF00
STRAIGHT, RIGHT, LEFT = 2, 1, 0                         # conn turn bits (info bits 28-29)
TURN_PRIORITY = {STRAIGHT: 2, RIGHT: 0, LEFT: 0}        # info bits 26-27, the most common value per turn in NYC
STRAIGHT_DEGREES = 30.0
MAX_SEGMENTS = 0x3F
CURVE_SAMPLES = 16


@dataclass
class NewRoad:
    """A road to add, in DPL metres: its centre line as segments of points (x, y, z) and its driving lanes per direction
    (0: start to end, 1: back), its pavements and parking strips as offsets from the centre line, positive to the left of
    the start-to-end direction."""
    segments: list[list[np.ndarray]]
    lanes: tuple[list[float], list[float]]
    pavements: list[float] = field(default_factory=list)
    parking: list[float] = field(default_factory=list)
    fast: bool = False

    def end(self, at_end: int) -> tuple[np.ndarray, np.ndarray]:
        """(point, unit start-to-end direction in x z) at the road start (0) or end (1)."""
        a, b = (self.segments[-1][-2], self.segments[-1][-1]) if at_end else (self.segments[0][0], self.segments[0][1])
        direction = (b - a)[[0, 2]]
        return (b if at_end else a), direction / np.linalg.norm(direction)


@dataclass
class Link:
    """A junction curve from a lane to a lane of another road, each as (road, direction, lane), with the curve's two
    inner Bezier control points (x y z) and its speed class (None: by its turn)."""
    source: tuple[int, int, int]
    target: tuple[int, int, int]
    controls: tuple[np.ndarray, np.ndarray]
    speed_class: int | None = None


class Target(NamedTuple):
    """Where a conn leads: a lane of a road, entered at that road's end `end`, and the conn's inner Bezier control points
    (None: from the lane ends) and speed class (None: by its turn)."""
    road: int
    end: int
    lane: int
    controls: tuple[np.ndarray, np.ndarray] | None
    speed_class: int | None = None


def split_long(roads: list[NewRoad], links: list[Link]) -> tuple[list[NewRoad], list[Link]]:
    """Roads of more than MAX_SEGMENTS segments as consecutive roads of at most that many (their shared ends become
    plain nodes in add_roads), and the links moved to the pieces their lanes leave or enter."""
    pieces: list[NewRoad] = []
    first, last = [], []

    for road in roads:
        first.append(len(pieces))
        pieces += [replace(road, segments=road.segments[i:i + MAX_SEGMENTS]) for i in range(0, len(road.segments), MAX_SEGMENTS)]
        last.append(len(pieces) - 1)

    def moved(lane: tuple[int, int, int], leaving: bool) -> tuple[int, int, int]:
        """Direction 0 leaves its road at the last piece and enters at the first, direction 1 the other way round."""
        i, d, k = lane
        return last[i] if (d == 0) == leaving else first[i], d, k

    return pieces, [replace(link, source=moved(link.source, True), target=moved(link.target, False)) for link in links]


def _lane_point(road: NewRoad, offset: float, at_end: int) -> tuple[np.ndarray, np.ndarray]:
    point, direction = road.end(at_end)
    left = np.array([direction[1], 0.0, -direction[0]])
    return point + left * offset, direction


def _continuations(roads: list[NewRoad]) -> dict[tuple[int, int, int], list[tuple[int, int, int]]]:
    """Per lane (road, direction, lane): the nearest lane of another road that starts within LANE_JOIN of its end, heading
    on. Driv3r joins lanes this way, not road ends: a ramp's lanes run on as the outer lanes of a highway road whose
    centre line lies 22 m to the side (Miami: up to 2.0 m apart)."""
    def lane_ends(at_start: bool) -> list[tuple[tuple[int, int, int], np.ndarray, np.ndarray]]:
        result = []
        for i, road in enumerate(roads):
            for d in (0, 1):
                e = d if at_start else 1 - d        # direction 0 starts at the road start and ends at its end
                for k, offset in enumerate(road.lanes[d]):
                    point, heading = _lane_point(road, offset, e)
                    result.append(((i, d, k), point, heading if d == 0 else -heading))
        return result

    starts = lane_ends(True)
    points = np.array([p for _, p, _ in starts])
    found: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}

    for lane, point, heading in lane_ends(False):
        distance = np.linalg.norm(points - point, axis=1)
        for n in np.argsort(distance):
            other, _, other_heading = starts[n]
            if distance[n] >= LANE_JOIN:
                break
            if other[0] != lane[0] and heading @ other_heading > 0.5:
                found[lane] = [other]
                break

    return found


def _bezier(points: list[np.ndarray]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, CURVE_SAMPLES + 1)[:, None]
    p0, p1, p2, p3 = points
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _crosses(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two x z polylines intersect (shared end points do not count)."""
    for p, q in zip(a[:-1], a[1:]):
        r = q - p
        for u, v in zip(b[:-1], b[1:]):
            s = v - u
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-9:
                continue
            t = ((u[0] - p[0]) * s[1] - (u[1] - p[1]) * s[0]) / den
            w = ((u[0] - p[0]) * r[1] - (u[1] - p[1]) * r[0]) / den
            if 1e-3 < t < 1 - 1e-3 and 1e-3 < w < 1 - 1e-3:
                return True
    return False


def _nodes(roads: list[NewRoad], links: list[Link],
           continuing: dict[tuple[int, int, int], list[tuple[int, int, int]]]) -> list[list[tuple[int, int]]]:
    """Road ends grouped into nodes, nearest pairs first: two groups join only when every end of one lies within JUNCTION
    (x z) and JUNCTION_HEIGHT of every end of the other. Joining near pairs transitively chained Miami's ends into
    junctions up to 266 m wide and tied bridge decks to the roads under them: conns then ran through barriers. The two
    ends a link or a lane continuation joins are one node whatever their distance."""
    ends = [(i, e) for i in range(len(roads)) for e in (0, 1)]
    points = np.array([roads[i].end(e)[0] for i, e in ends])
    flat = np.linalg.norm(points[:, None, [0, 2]] - points[None, :, [0, 2]], axis=2)
    near = (flat < JUNCTION) & (np.abs(points[:, None, 1] - points[None, :, 1]) < JUNCTION_HEIGHT)
    road = np.array([i for i, _ in ends])
    group = list(range(len(ends)))
    members = {k: [k] for k in range(len(ends))}

    def join(p: int, q: int, linked: bool) -> None:
        a, b = group[p], group[q]
        if a != b and (linked or near[np.ix_(members[a], members[b])].all()):
            for k in members[b]:
                group[k] = a
            members[a] += members.pop(b)

    for p, q in sorted(zip(*np.nonzero(np.triu(near & (road[:, None] != road[None, :])))), key=lambda pq: flat[pq]):
        join(p, q, False)
    for link in links:      # end k of road i is item 2i + k; direction 0 leaves at the end and enters at the start
        join(2 * link.source[0] + 1 - link.source[1], 2 * link.target[0] + link.target[1], True)
    for (i, d, _), into in continuing.items():
        for j, f, _ in into:
            join(2 * i + 1 - d, 2 * j + f, True)

    return [[ends[k] for k in m] for m in members.values()]


def drop_dead_ends(roads: list[NewRoad], links: list[Link]) -> tuple[list[NewRoad], list[Link], int]:
    """The roads without the driving lanes that can only end at a dead end (a DPL car stops for good there; Miami: one-way
    roads at the west edge, partly over no ground), and the links without them; add_roads gives every other lane an exit
    (a link, a lane it runs on into, the other road of a plain node, or turning back). Returns the number dropped too."""
    roads, dropped = list(roads), 0

    while True:
        continuing = _continuations(roads)
        nodes = _nodes(roads, links, continuing)
        node_of = {end: node for node in nodes for end in node}
        linked = {link.source[:2] for link in links}
        dead = []

        for i, road in enumerate(roads):
            for d in (0, 1):
                node = node_of[i, 1 - d]
                points = [roads[j].end(f)[0] for j, f in node]
                is_plain = len(node) == 2 and np.linalg.norm(points[0] - points[1]) < JOIN
                if road.lanes[d] and not (
                        (i, d) in linked or road.lanes[1 - d] or any((i, d, k) in continuing for k in range(len(road.lanes[d])))
                        or ((is_plain or not links) and any(roads[j].lanes[f] for j, f in node if j != i))):
                    dead.append((i, d))

        if not dead:
            return roads, links, dropped
        for i, d in dead:
            dropped += len(roads[i].lanes[d])
            roads[i] = replace(roads[i], lanes=tuple([] if k == d else roads[i].lanes[k] for k in (0, 1)))
        links = [link for link in links if link.source[:2] not in dead and link.target[:2] not in dead]


def add_roads(ai: AiExport, roads: list[NewRoad], links: list[Link] | None = None, net: int = 0) -> str:
    """Appends the roads to the street network (net 0, network 1) as a network of their own: plain nodes where two ends
    meet, junctions with Bezier conns elsewhere, no traffic lights. At a junction the lanes of a road end with links take
    exactly those curves; the others get a conn to every lane leaving the node. Network 2 keeps its roads (their indices
    move up). Net 1 is network 2, the train tracks (0x435014: every road with two direction-0 lanes adds lane k to track
    k); its roads get NYC's rail road flags."""
    t = ai.tables
    first, added = sum(ai.road_counts[:net + 1]), len(roads)

    for name, mask in (('a', 0x7FFF), ('b', 0x7FFF), ('cell_roads', 0xFFFF)):
        refs = t[name] & mask
        t[name] = np.where(refs >= first, t[name] + added, t[name]).astype(t[name].dtype)
    for name in ('lane', 'c', 'd'):
        t[name]['road'] = np.where(t[name]['road'] >= first, t[name]['road'] + added, t[name]['road'])

    new_roads = np.zeros(added, ROAD)
    segs, pieces, lanes, conns, preds, conflicts, a_list, b_list = [], [], [], [], [], [], [], []
    seg_base, piece_base, lane_base = len(t['seg']), len(t['piece']), len(t['lane'])
    c_base, d_base = len(t['c']), len(t['d'])
    pavements, parking, h6c, h70 = [], [], [], []

    for i, road in enumerate(roads):
        rec = new_roads[i]
        if len(road.segments) > MAX_SEGMENTS:
            raise ValueError(f'road {i}: {len(road.segments)} segments, at most {MAX_SEGMENTS}')
        rec['seg'], rec['nseg'] = seg_base + len(segs), len(road.segments) | SEG_COUNT_FLAGS

        for points in road.segments:
            seg = np.zeros(1, SEG)[0]
            seg['x'], seg['z'] = pack_position(*points[0])
            seg['piece'], seg['npiece'] = piece_base + len(pieces), len(points) - 1
            for p, q in zip(points[:-1], points[1:]):
                length = float(np.linalg.norm(q - p))
                # turns +z to the piece: yaw, then pitch (a yaw-only piece keeps a ramp at its segment's start height)
                yaw, pitch = np.arctan2(q[0] - p[0], q[2] - p[2]), np.arcsin(np.clip((q[1] - p[1]) / max(length, 1e-9), -1, 1))
                sy, cy, sp, cp = np.sin(yaw / 2), np.cos(yaw / 2), np.sin(pitch / 2), np.cos(pitch / 2)
                pieces.append(((-cy * sp, sy * cp, sy * sp, cy * cp), length))
            seg['length'] = sum(length for _, length in pieces[-(len(points) - 1):])
            segs.append(seg)

        rec['length'] = sum(s['length'] for s in segs[-len(road.segments):])
        rec['light'] = 0xFF
        rec['flags'], rec['x26'] = (RAIL_ROAD_FLAGS if net else ROAD_FLAGS) | (FAST_ROAD if road.fast else 0), ROAD_X26
        rec['c'], rec['nc'] = c_base + len(pavements) if road.pavements else NONE, len(road.pavements)
        rec['d'], rec['nd'] = d_base + len(parking) if road.parking else NONE, len(road.parking)

        for offset in road.pavements:
            item = np.zeros(1, C_ITEM)[0]
            item['offset'], item['x02'], item['road'], item['x07'] = round(offset * 100), PAVEMENT_X02, first + i, 0xFF
            item['next'] = item['link'] = NONE
            pavements.append(item)

        for offset in road.parking:
            item = np.zeros(1, D_ITEM)[0]
            item['offset'], item['road'], item['x06'] = round(offset * 100), first + i, PARKING_X06
            parking.append(item)

        for d in (0, 1):
            rec['nlane'][d] = len(road.lanes[d])
            rec['lane'][d] = lane_base + len(lanes) if road.lanes[d] else NONE
            for offset in road.lanes[d]:
                lane = np.zeros(1, LANE)[0]
                lane['offset'], lane['road'], lane['x07'] = round(offset * 100), first + i, 0xFF
                lane['pred'] = lane['conn'] = NONE
                lanes.append(lane)

    def entering(j: int, end: int) -> list[int]:
        """Lanes of road j that leave the node at its end `end` (dir 0 leaves the start, dir 1 the end)."""
        d = end
        return [int(new_roads[j]['lane'][d]) + k for k in range(len(roads[j].lanes[d]))]

    links = links or []
    curves: dict[tuple[int, int], dict[int, list[Target]]] = {}
    for link in links:
        (i, d, k), (j, f, m) = link.source, link.target
        curves.setdefault((i, d), {}).setdefault(k, []).append(Target(j, f, m, link.controls, link.speed_class))
    continuing = _continuations(roads)
    partners: dict[tuple[int, int], set[tuple[int, int]]] = {}      # the road ends a lane continues across, both ways
    for (i, d, _), into in continuing.items():
        for j, f, _ in into:
            partners.setdefault((i, 1 - d), set()).add((j, f))
            partners.setdefault((j, f), set()).add((i, 1 - d))
    plain = junctions = u_turns = 0

    for node in _nodes(roads, links, continuing):
        points = [roads[i].end(e)[0] for i, e in node]
        is_plain = len(node) == 2 and np.linalg.norm(points[0] - points[1]) < JOIN
        plain += is_plain
        junctions += len(node) > 1 and not is_plain
        _link_pavements(node, roads, new_roads, pavements, c_base, is_plain, h6c, h70,
                        len(t['h6c']) // 2, len(t['h70']) // 4)
        node_conns: list[tuple[int, np.ndarray]] = []

        for i, e in node:
            others = [(j, f) for j, f in node if (j, f) != (i, e)]
            joined = set(others) if is_plain else partners.get((i, e), set())
            for name, target, ends in (('a', a_list, [o for o in others if o in joined]),
                                       ('b', b_list, [o for o in others if o not in joined])):
                new_roads[i][name][e], new_roads[i]['n' + name][e] = len(target) + len(t[name]), len(ends)
                target += [j + first | f << 15 for j, f in ends]
            d = 1 - e       # the direction whose lanes leave road i at this end
            for k, offset in enumerate(roads[i].lanes[d]):
                li = int(new_roads[i]['lane'][d]) + k
                lane = lanes[li - lane_base]
                into = continuing.get((i, d, k)) or [(j, f, min(k, len(entering(j, f)) - 1)) for j, f in others
                                                     if is_plain and entering(j, f) and j != i][:1]
                if into:
                    items = [entering(j, f)[m] | f << 15 for j, f, m in into]
                    lane['pred'], lane['npred'] = len(t['pred']) + len(preds), 2 * len(items)
                    preds += items + items
                    continue
                if curves.get((i, d)):
                    targets = _curves(curves[i, d], k)
                else:       # without links: every lane leaving the junction
                    targets = [Target(j, f, min(k, len(entering(j, f)) - 1), None) for j, f in others
                               if not links and entering(j, f) and j != i]
                if not targets and entering(i, e):      # a dead end: turn back (a DPL car stops there)
                    targets = [Target(i, e, min(k, len(entering(i, e)) - 1), None)]
                    u_turns += 1
                if not targets:
                    continue
                lane['conn'], lane['nconn'] = len(t['conn']) + len(conns), len(targets)
                p0, h0 = _lane_point(roads[i], offset, e)
                h0 = h0 if e else -h0
                for j, f, m, controls, speed_class in targets:
                    dst = entering(j, f)[m]
                    p3, h1 = _lane_point(roads[j], roads[j].lanes[f][m], f)
                    h1 = -h1 if f else h1
                    if controls:
                        p1, p2 = controls
                    else:
                        chord = np.linalg.norm((p3 - p0)[[0, 2]]) * CONTROL
                        p1 = p0 + np.array([h0[0], 0.0, h0[1]]) * chord
                        p2 = p3 - np.array([h1[0], 0.0, h1[1]]) * chord
                    curve = _bezier([p0, p1, p2, p3])
                    turn = np.degrees(np.arctan2(h0[0] * h1[1] - h0[1] * h1[0], h0 @ h1))
                    kind = STRAIGHT if abs(turn) < STRAIGHT_DEGREES else RIGHT if turn > 0 else LEFT
                    conn = np.zeros(1, CONN)[0]
                    conn['x'], conn['z'] = pack_position(*p1)
                    conn['dx'], conn['dz'] = round((p2[0] - p1[0]) * 100), round((p2[2] - p1[2]) * 100)
                    conn['y'] = round(p2[1] * 100)
                    if speed_class is None:
                        speed_class = 0 if kind == STRAIGHT else TURN_CLASS
                    length = round(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum() * 100)
                    conn['length'] = min(length, CONN_LENGTH) | speed_class << 14
                    conn['info'] = (d == 0) << 24 | (f == 1) << 25 | TURN_PRIORITY[kind] << 26 | kind << 28
                    conn['src'], conn['dst'] = li, dst
                    node_conns.append((len(conns), curve[:, [0, 2]]))
                    conns.append(conn)

        for n, (c, curve) in enumerate(node_conns):
            crossing = [len(t['conn']) + o for m, (o, other) in enumerate(node_conns)
                        if m != n and (conns[o]['src'] != conns[c]['src']) and
                        (conns[o]['dst'] == conns[c]['dst'] or _crosses(curve, other))]
            conns[c]['info'] |= len(crossing) | (len(t['conflict']) + len(conflicts)) << 8
            conflicts += crossing

    for name, rows, dtype in (('seg', segs, SEG), ('lane', lanes, LANE), ('conn', conns, CONN), ('c', pavements, C_ITEM),
                              ('d', parking, D_ITEM)):
        t[name] = np.concatenate([t[name], np.array(rows, dtype)])
    t['piece'] = np.concatenate([t['piece'], np.array([(q, length) for q, length in pieces], PIECE)])
    for name, rows in (('pred', preds), ('conflict', conflicts), ('a', a_list), ('b', b_list)):
        t[name] = np.concatenate([t[name], np.array(rows, '<u2')])
    for i in range(added):
        for name, count in (('a', 'na'), ('b', 'nb')):
            for e in (0, 1):
                if not new_roads[i][count][e]:
                    new_roads[i][name][e] = len(t[name])

    t['roads'] = np.concatenate([t['roads'][:first], new_roads, t['roads'][first:]])
    ai.road_counts[net] += added
    _add_to_cells(ai, roads, first, net)
    t['h6c'] += np.array(h6c, '<u2').tobytes()
    t['h70'] += np.array(h70, '<u2').tobytes()
    return (f'{added} roads, {len(lanes)} lanes ({len(continuing)} continuing on a lane of another road), {plain} plain '
            f'nodes, {junctions} junctions, {len(conns)} conns ({u_turns} turning back at a dead end), '
            f'{len(pavements)} pavements ({len(h70)} corner and crossing links), {len(parking)} parking strips')


def _curves(by_lane: dict[int, list[Target]], k: int) -> list[Target]:
    """The links leaving lane k of a road end. A lane without any (Driv3r drops a lane at a junction: 40 in Miami, such
    as outer highway lanes) takes those of the nearest lane of that end: without an exit a DPL car stops at its end."""
    return by_lane[min(by_lane, key=lambda m: abs(m - k))]


def _link_pavements(node: list[tuple[int, int]], roads: list[NewRoad], new_roads: np.ndarray, pavements: list,
                    c_base: int, is_plain: bool, h6c: list[int], h70: list[tuple[int, int]], h6c_base: int,
                    h70_base: int) -> None:
    """Links the pavement ends of a node: at a plain node each continues on the nearest pavement end of the other road
    (h6c); at a junction each goes round the corner to the nearest pavement end of another road and crosses that road to
    the nearest pavement end of a third (h70)."""
    ends = [(int(new_roads[i]['c']) + k, e, _lane_point(roads[i], offset, e)[0][[0, 2]], i)
            for i, e in node for k, offset in enumerate(roads[i].pavements)]

    for c, e, point, road in ends:
        others = sorted((float(np.linalg.norm(q - point)), o, f, j) for o, f, q, j in ends if j != road)
        if not others:
            continue
        item = pavements[c - c_base]
        if is_plain:
            item['next'][e], item['nnext'][e] = h6c_base + len(h6c), 1
            h6c.append(others[0][1] | others[0][2] << 15)
            continue
        crossing = [other for other in others if other[3] != others[0][3]]
        for k, (_, o, f, _) in enumerate(others[:1] + crossing[:1]):
            item['link'][k, e] = h70_base + len(h70)
            h70.append((c | e << 15, o | f << 15))


def _add_to_cells(ai: AiExport, roads: list[NewRoad], first: int, net: int) -> None:
    """Lists every new road in the cells of its network's grid that its box touches, grown by one cell on each side."""
    t, grid = ai.tables, ai.grids[net]
    base = sum(g.width * g.height for g in ai.grids[:net])     # the grids' cells lie one after the other
    lists = [list(t['cell_roads'][c >> 16:(c >> 16) + (c & 0xFFFF)]) for c in t['cells']]

    for i, road in enumerate(roads):
        xz = np.array([p[[0, 2]] for points in road.segments for p in points])
        lo = np.floor(xz.min(axis=0) / CELL).astype(int) - 1
        hi = np.floor(xz.max(axis=0) / CELL).astype(int) + 1
        for cx in range(max(lo[0], grid.min_x), min(hi[0], grid.min_x + grid.width - 1) + 1):
            for cz in range(max(lo[1], grid.min_z), min(hi[1], grid.min_z + grid.height - 1) + 1):
                lists[base + (cz - grid.min_z) * grid.width + cx - grid.min_x].append(first + i)

    starts = np.cumsum([0] + [len(items) for items in lists[:-1]])
    t['cells'] = np.array([s << 16 | len(items) for s, items in zip(starts, lists)], '<u4')
    t['cell_roads'] = np.array([r for items in lists for r in items], '<u2')
