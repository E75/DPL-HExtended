"""Reader for the road network of a Driv3r city: chunk AIEX v28 "AIExport" in GERD "Spline Road Data" of the first GESR
of <City>_DAY_PC.d3c. Only what the DPL converter needs: roads, their centre lines, driving lanes, pavements and parking.

Layout (found from the data and the loader 0x402A10 of the Driv3r exe; see docs/D3-TO-DPL-CONVERSION.md):
- The chunk starts with 11 (u32 count, u32 offset) pairs; pair k sits at 8k and its offset counts from there.
- Table 3, roads of 28 bytes: f32 length, f32 1 / length, u16 first lane per direction, u8 lanes per direction, u16,
  2 x u16 0, 4 bytes, u16 segment count, u8, u8 flags (bit 0: fast road, traffic capped at 27 m/s instead of 18 in
  the driving brain 0x40BED0). A road owns the next `segment count` segments of table 2, in order.
- Table 2, segments of 48 bytes: start x y z (f32), f32 length, f32 1 / length, u32 piece count, u32 offset of the pieces
  counted from the piece-count word; pieces of 16 bytes: unit direction x y z, f32 length.
- Table 4, lanes of 12 bytes: f32 offset from the centre line in metres, f32 height above the road (0.15 on a pavement),
  u16 kind, u16 road (low 9 bits; the high bits are flags). A road's lanes lie together in the table: its driving lanes
  per direction first (direction 0, then 1), then its other lanes. The road's own first-lane words agree in Miami and
  taken from the road word. The other lanes are pavements (height above PAVEMENT_HEIGHT) and parking strips (height 0). Driving lanes at height 1.0 are
  tram tracks (Miami's Metromover: the only 4 such lanes, on the rail-bed models). A road with neither driving lanes
  nor pavements is a route off the car network (Miami 355-390, up to 11 km, over water and bridges): no parking.
- Table 10, one 16-byte item per driving lane: u16 +0x0A its lane in table 4.
- Table 8, junction curves of 32 bytes: f32 x y z of the first inner Bezier control point, u16 target, u16 source (both
  items of table 10), f32 x y z of the second control point, 4 bytes (+0x1E bit 7: speed class 2, else +0x1F bit 7:
  class 1, else 0; traffic takes the curve at 5, 20 or 27 m/s, 0x40BED0). The first point lies straight ahead of the source
  lane's end, the second straight before the target lane's start (Miami: 7-22 m, all 1843 curves).
"""
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dplchunk import read_tree

TABLES = 11
ROAD_SIZE = 28
SEGMENT_SIZE = 48
PIECE_SIZE = 16
LANE_SIZE = 12
CURVE_SIZE = 32
DRIVING_LANE_SIZE = 16
PAVEMENT_HEIGHT = 0.1
RAIL_HEIGHT = 0.9
LANE_ROAD = 0x1FF
ROAD_FLAGS = 0x1B
FAST_ROAD = 1
BRIDGE_CONTROL = 0.4    # rail_line: Bezier control points at this share of the chord across a junction
BRIDGE_STEPS = 8


@dataclass
class Segment:
    start: np.ndarray                       # x y z
    pieces: list[tuple[np.ndarray, float]]  # (unit direction x y z, length)

    def points(self) -> list[np.ndarray]:
        """Start and the end of every piece."""
        result = [self.start]

        for direction, length in self.pieces:
            result.append(result[-1] + direction * length)

        return result


@dataclass
class Road:
    length: float
    segments: list[Segment]
    lanes: list[list[float]] = field(default_factory=list)    # per direction: offsets of the driving lanes, metres
    lane_ids: list[list[int]] = field(default_factory=list)   # per direction: their items in table 4
    pavements: list[float] = field(default_factory=list)      # offsets, metres
    parking: list[float] = field(default_factory=list)
    rail: bool = False                                        # a tram or train track, no car traffic
    fast: bool = False                                        # traffic capped at 27 m/s instead of 18
    rails: list[tuple[int, float, float]] = field(default_factory=list)  # rail lanes: (table 4 item, offset, height)

    def points(self) -> list[np.ndarray]:
        result: list[np.ndarray] = []

        for segment in self.segments:
            result += segment.points()[1 if result else 0:]

        return result


def _table(data: bytes, k: int) -> tuple[int, int]:
    count, offset = struct.unpack_from('<II', data, 8 * k)
    return count, 8 * k + offset


def read_roads(data: bytes) -> list[Road]:
    """Roads of an AIEX v28 chunk."""
    road_count, road_at = _table(data, 3)
    segment_count, segment_at = _table(data, 2)
    lane_count, lane_at = _table(data, 4)
    segments = []

    for s in range(segment_count):
        at = segment_at + s * SEGMENT_SIZE
        start = np.array(struct.unpack_from('<3f', data, at))
        pieces_count, pieces_offset = struct.unpack_from('<II', data, at + 20)
        pieces_at = at + 20 + pieces_offset
        pieces = []

        for k in range(pieces_count):
            *direction, length = struct.unpack_from('<4f', data, pieces_at + k * PIECE_SIZE)
            pieces.append((np.array(direction), length))

        segments.append(Segment(start, pieces))

    lanes = [struct.unpack_from('<2f2H', data, lane_at + i * LANE_SIZE) for i in range(lane_count)]
    road_lanes: list[list[int]] = [[] for _ in range(road_count)]

    for i, (_, _, _, owner) in enumerate(lanes):
        if owner & LANE_ROAD < road_count:
            road_lanes[owner & LANE_ROAD].append(i)

    roads = []
    driving = set()
    next_segment = 0

    for r in range(road_count):
        length, _, _, _, count0, count1 = struct.unpack_from('<2f2H2B', data, road_at + r * ROAD_SIZE)
        owned = struct.unpack_from('<H', data, road_at + r * ROAD_SIZE + 0x18)[0]
        road = Road(length, segments[next_segment:next_segment + owned],
                    fast=bool(data[road_at + r * ROAD_SIZE + ROAD_FLAGS] & FAST_ROAD))
        next_segment += owned

        if len(road_lanes[r]) < count0 + count1:
            raise ValueError(f'road {r}: {count0 + count1} driving lanes, {len(road_lanes[r])} lanes name it')

        for ids in (road_lanes[r][:count0], road_lanes[r][count0:count0 + count1]):
            road.lanes.append([lanes[i][0] for i in ids])
            road.lane_ids.append(ids)
            road.rail |= any(lanes[i][1] >= RAIL_HEIGHT for i in ids)
            driving.update(ids)

        roads.append(road)

    # curves: no parking there
    driven = driving | {lane for curve in read_curves(data) for lane in (curve.source, curve.target)}

    for i, (offset, height, _, owner) in enumerate(lanes):
        owner &= LANE_ROAD

        if owner >= road_count:
            continue
        if i in driven and (i not in driving or height >= RAIL_HEIGHT):
            roads[owner].rails.append((i, offset, height))
        elif i not in driven:
            (roads[owner].pavements if height > PAVEMENT_HEIGHT else roads[owner].parking).append(offset)

    for road in roads:
        if not any(road.lanes) and not road.pavements:
            road.parking.clear()

    if next_segment != segment_count:
        raise ValueError(f'roads own {next_segment} of {segment_count} segments')

    return roads


@dataclass
class Curve:
    source: int                                 # lane (table 4) it leaves
    target: int                                 # lane (table 4) it enters
    controls: tuple[np.ndarray, np.ndarray]     # the inner Bezier control points, x y z
    speed_class: int                            # 0: 27 m/s, 1: 20 m/s, 2: 5 m/s (DPL's conn classes are the same)


def read_curves(data: bytes) -> list[Curve]:
    """Junction curves of an AIEX v28 chunk."""
    lane_count, lane_at = _table(data, 10)
    driving = [struct.unpack_from('<H', data, lane_at + i * DRIVING_LANE_SIZE + 0x0A)[0] for i in range(lane_count)]
    curve_count, curve_at = _table(data, 8)
    curves = []

    for c in range(curve_count):
        *first, target, source = struct.unpack_from('<3f2H', data, curve_at + c * CURVE_SIZE)
        second = struct.unpack_from('<3f', data, curve_at + c * CURVE_SIZE + 0x10)
        slow, medium = data[curve_at + c * CURVE_SIZE + 0x1E] & 0x80, data[curve_at + c * CURVE_SIZE + 0x1F] & 0x80
        curves.append(Curve(driving[source], driving[target], (np.array(first), np.array(second)), 2 if slow else 1 if medium else 0))

    return curves


@dataclass
class RailLine:
    """A two-track line as pieces in order, each a centre line at rail height (x y z points, a piece starting where the
    last one ended) and the distance of its two tracks (one per direction) to either side of it."""
    pieces: list[tuple[list[np.ndarray], float]]

    @property
    def points(self) -> list[np.ndarray]:
        return [p for k, (points, _) in enumerate(self.pieces) for p in points[1 if k else 0:]]

    @property
    def length(self) -> float:
        points = self.points
        return float(sum(np.linalg.norm(q - p) for p, q in zip(points[:-1], points[1:])))


def rail_line(data: bytes, roads: list[Road]) -> RailLine | None:
    """The city's longest two-track rail line: its roads with two or more rail lanes, chained by the junction curves
    to run on from the last. A road's piece runs between its outermost rail lanes; the gaps through junctions are
    bridged by a Bezier along the end directions, one piece per step with the track distance going over evenly."""
    owner = {lane: r for r, road in enumerate(roads) if len(road.rails) >= 2 for lane, _, _ in road.rails}
    links: dict[int, set[int]] = {r: set() for r in set(owner.values())}

    for curve in read_curves(data):
        a, b = owner.get(curve.source), owner.get(curve.target)
        if a is not None and b is not None and a != b:
            links[a].add(b)
            links[b].add(a)

    def farthest(start: int) -> list[int]:
        """The longest chain (in roads) from start, by breadth-first search."""
        paths = {start: [start]}
        todo = [start]
        while todo:
            r = todo.pop(0)
            for n in sorted(links[r] - paths.keys()):
                paths[n] = paths[r] + [n]
                todo.append(n)
        return max(paths.values(), key=len)

    best: list[int] = []
    for r in links:
        chain = farthest(farthest(r)[-1])
        if sum(roads[c].length for c in chain) > sum(roads[c].length for c in best):
            best = chain
    if not best:
        return None

    pieces: list[tuple[list[np.ndarray], float]] = []
    for k, r in enumerate(best):
        line = _rail_centre(roads[r])
        gauge = (max(o for _, o, _ in roads[r].rails) - min(o for _, o, _ in roads[r].rails)) / 2
        if k == 0 and len(best) > 1:
            following = roads[best[1]].points()
            ends = (following[0], following[-1])
            if min(np.linalg.norm(line[0] - e) for e in ends) < min(np.linalg.norm(line[-1] - e) for e in ends):
                line = line[::-1]
        elif k > 0:
            end, last_gauge = pieces[-1][0][-1], pieces[-1][1]
            if np.linalg.norm(line[-1] - end) < np.linalg.norm(line[0] - end):
                line = line[::-1]
            bridge = [end] + _bridge(pieces[-1][0][-2], end, line[0], line[1]) + [line[0]]
            pieces += [([p, q], last_gauge + (gauge - last_gauge) * (s + 0.5) / (len(bridge) - 1))
                       for s, (p, q) in enumerate(zip(bridge[:-1], bridge[1:]))]
        pieces.append((line, gauge))

    return RailLine(pieces)


def _rail_centre(road: Road) -> list[np.ndarray]:
    """The road's centre line moved sideways to the middle of its outermost rail lanes and up to their height (Driv3r
    counts lane offsets to the right of the road direction)."""
    offsets = [offset for _, offset, _ in road.rails]
    shift = (max(offsets) + min(offsets)) / 2
    lift = float(np.mean([height for _, _, height in road.rails]))
    points = road.points()
    result = []
    for k, p in enumerate(points):
        a, b = points[max(k - 1, 0)], points[min(k + 1, len(points) - 1)]
        flat = np.array([b[0] - a[0], b[2] - a[2]]) / max(float(np.hypot(b[0] - a[0], b[2] - a[2])), 1e-9)
        result.append(p + np.array([-flat[1] * shift, lift, flat[0] * shift]))
    return result


def _bridge(before: np.ndarray, end: np.ndarray, start: np.ndarray, after: np.ndarray) -> list[np.ndarray]:
    """Points strictly between a road end and the next road's start: a cubic Bezier leaving along the first road and
    arriving along the second, its control points BRIDGE_CONTROL of the chord out."""
    reach = np.linalg.norm(start - end) * BRIDGE_CONTROL
    out = (end - before) / max(np.linalg.norm(end - before), 1e-9)
    into = (after - start) / max(np.linalg.norm(after - start), 1e-9)
    p1, p2 = end + out * reach, start - into * reach
    return [(1 - t) ** 3 * end + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * start
            for t in np.linspace(0, 1, BRIDGE_STEPS + 2)[1:-1]]


def aiex_chunk(path: Path) -> bytes:
    """The AIEX chunk of a Driv3r city file (<City>_DAY_PC.d3c)."""
    for top in read_tree(path.read_bytes(), 0, 3).chunks:
        for spline in (top.tree.chunks if top.tag == b'GESR' and top.tree else []):
            for chunk in (spline.tree.chunks if spline.tag == b'GERD' and spline.tree else []):
                if chunk.tag == b'AIEX':
                    return bytes(chunk.data)

    raise ValueError(f'{path}: no AIEX chunk')
