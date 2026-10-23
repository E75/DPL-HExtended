"""Builds a Driv3r city as a Driver Parallel Lines city file, on DPL's own region grid (docs/PLAN-CITY-IMPORT.md, part 8).

usage: d3_city_build.py <DPL game dir> <out.d4c> [--driv3r <Driv3r game dir>] [--city Miami] [--map <out.map>] [--level]

NYC Now is the template: its super regions, regions and lookup tables stay, their contents are replaced. The original
city file is only read.
- 8a, always: NYC without its city. Every (super) region and the global region keep an empty instance list (GEBI) and
  lose their broadphase (JNR!), occluders (OCGD), projected textures (PRJT) and props (PROP). Scene trees (SCND),
  terrain, roads and the lookup tables stay.
- 8b, --driv3r: the city's global page (Driv3r page 0: shared props) goes into DPL's global pack (page 0), Driv3r
  container k into DPL container k.
- 8c, --driv3r: blocks of 2 x 2 DPL regions become the super regions (GESI region map); each super region pack is
  rebuilt from scratch with the Driv3r containers its regions use. Instances that turn about more than y, and the global
  props of a region with more instances than a GEBI holds, are baked in place into containers around their region's
  centre. All go into the GEBI of the DPL region holding them, and each such region gets a new scene tree (SCND); a NYC
  water region gets both chunks.
- 8d, --driv3r: collision (Driv3r's records per model in the GEPM of its super region, instances with collision as GEBI
  kind 0 with a new broadphase JNR! per region, the rest kind 5), terrain (d3_terrain: Driv3r's GRTD cells per DPL
  region, NYC's terrain dropped elsewhere) and the road network (NYC's replaced by the city's, with pavements and
  parking strips).
- Sea, --driv3r: DPL has no water system; NYC draws its open sea as a flat 768 m tile (one global container) per
  region. That container is kept and placed in every region at the Driv3r city's water height (its GWTR chunk), so the
  land covers it and the sea reaches the grid's edge.
- --level: the city's LEVEL of levels.txt starts on the road nearest Driv3r's own start (StartPosition), and loads a
  copy of NYC Now's life event spool whose base life event puts the player there too (LifeEvents/<city>_missions.sp)
  and keeps NYC's mission starts and pickups out of reach.
"""
import argparse
import math
import mmap
import os
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d3_ai
import d3_lights
import d3_props
import d3_tanner
import d3_terrain
import d3_vehicle_swap
import d3city
import dpl_ai
import dpl_ground
import dplchunk as d
import dpl_poc

TEMPLATE = 'nyc_now_pc.d4c'
CITY_TAGS = {b'JNR!', b'OCGD', b'PRJT', b'PROP'}
# GEBI v10 header: u32 header size, i16 column, i16 row, u16 instance count, 6 x u16 running ends of the instance kinds
GEBI_HEADER = struct.Struct('<I2h7H')
GEBI_RECORD = struct.Struct('<4I')
ASRD_ENTRY = 8
KINDS = 6
# NYC's handles keep one kind each; kinds 0-4 have broadphase objects, kind 5 (13093 instances) none: drawn only.
DRAWN_ONLY = 5
PROP_KIND = 3           # NYC's props; the exe turns kinds 3-4 into moving props when hit (0x53125F)
W1_FLAGS = 0x800        # bits 20-31 of W1 on most NYC instances (meaning unknown)
MAX_REGION_INSTANCES = 1 << 12  # W3 bits 20-31 hold the instance's index in its region
ORIGINAL_MAPS = ('nyc_now.map', 'nyc_then.map')
MAP_CONTAINER = 0      # DPL draws it always, at the origin; both games: map x = -world x / 500, map y = world z / 500
D3_MAP_LAYERS = (1, 0)  # Driv3r's land and water, then its roads on top
MAP_OVERLAY = 0x44000      # substance word 1 flags of every material in DPL's maps: drawn as overlay, no depth fight
# word 1 low byte per layer (NYC's own: land 0x0C, road tiles 0x10 and 0x11); within a layer the next material
# takes the next of them. Values NYC never uses (0x0B, 0x0D, 0x0F) left the land undrawn
MAP_PRIORITIES = ((0x0C,), (0x10, 0x11))
MAP_TILES = range(16, 53)   # NYC's road tiles, drawn around the player from an 11 x 11 table in the exe (0x64B4A0)
BAKE_ROOM = 64          # GEBI records and broadphase objects kept free per region for its containers of baked models
MAX_REGION_OBJECTS = d.MAX_OBJECTS      # 12 bits of a broadphase node's first object
MIN_HEIGHT, MAX_HEIGHT = -0x3200 / 128, (0x10000 - 0x3200) / 128     # W2's 16 height bits (Driv3r hides some at -150)
GRID_ORIGIN = -5760.0
REGION_SIZE = 768.0
GLOBAL_PAGE = 0
NO_MODEL = 1023         # Driv3r's container index of a collision-only instance (bridge decks, ramps)
GRID_SIDE = 15          # regions per row of DPL's grid
# Regions per side of a new super region: Miami's 53 regions make 18 blocks, as many as NYC's super regions, each within
# MAX_CONTAINERS (at most 822 measured)
SUPER_REGION_BLOCK = 2
MAX_PACK = 0x2BC0000      # 43.75 MB: DPL's largest super region pack in every city, the ceiling they were built to
MAX_CONTAINERS = 1 << 10        # the container bits of a model handle
YAW_ONLY = 1e-4         # largest x and z of a yaw-only rotation quaternion
AT_ORIGIN = d3city.Instance((0.0, 0.0, 0.0), 0, (0.0, 0.0, 0.0, 1.0))
CLEAR_STEP = 0.25       # metres a lane or parking strip in a collision box moves towards the road's centre line per try
CLEAR_MAX = 1.0         # Miami's outer highway lanes clear the barriers they touch after 0.5-0.75 m
ROAD_STEP = 2.0         # metres between the points tested along a road
DECK_LIFT_MAX = 0.4     # metres a road point may rise onto the top of the deck box it lies in
DECK_MIN_THICKNESS = 0.9    # metres: thinner boxes are kerbs and rails, not decks
# NYC's ATRD lists that place things -> record size: 1 standing cars (0x410BAB), 2 no-parking zones, 3 lights on models
# at night (0x4CBE00), 7 particle emitters (0x4DD938), 8 litter (0x54188C), 10 player recovery points, 15 ambient cars
NYC_LISTS = {1: 10, 2: 8, 3: 12, 7: 22, 8: 16, 10: 8, 15: 10}
RECOVERY_LIST = 10
RECOVERY_CELL = 40.0    # metres: one recovery point per cell (NYC's lie 40-64 m apart along its streets)
RECOVERY_BAND = 5.0     # metres of height per cell, so a road under a viaduct keeps its own points
RECOVERY_LIFT = 0.5     # NYC's points stand this high over its streets
# SCND, the render quadtree per region: header 0x30 (u32 node offset, object offset, node count, object count, 4, then
# region size, min x, min z f32, column, row, 8 bytes of name), nodes and object list as JNR! (dplchunk); object = GEBI
# instance index.
SCND_HEADER = 0x30
SCND_MAX_DEPTH = 10     # NYC's deepest node
SEA_TILE_FLATNESS = 0.01    # height of NYC's sea tile box, metres
SEA_UV_REPEAT = 48      # texture repeats over a sea tile: one per 16 m
# NYC's shore pieces (global containers 280, 340, ...): water quad of material 174 at y -2, river floor of 64 at -10
SEA_WATER_MATERIAL = 174
SEA_BED_MATERIAL = 64
SEA_BED_DEPTH = 8.0
SEA_DRAW_DROP = 0.3
ROAD_MARGIN = 2.0       # metres beyond a road's outer lane or pavement (and between its sample points) kept dry when sunken
FLOOR_BELOW = 0.5      # a collision top this far under the water level or lower is a sea floor, not a floor to stand on
FLOOR_POINTS = 5       # of the cell's 9 sample points that must stand on that top (the centre one included)
FLOOR_FLAT = 0.3       # metres the tops under one terrain cell's sample points may differ for the cell to get a floor
QUATERNION_AS_STORED = set()     # cities whose tilted instances keep Driv3r's quaternion uninverted (C33)
PROP_TILT = 10.0
# x z of Driv3r's own Take A Ride start car, read in Driv3r (Miami: South Beach)
CITY_STARTS = {'miami': (1988.2, -1081.6)}
START_LIFT = 1.0        # metres above the road: the player drops onto it
NYC_LIFE_SPOOL = 'nyc_now_missions.sp'
# the base life event's player start (2570.034, 0.043, -231.333): the translation of two 4x4 matrix properties in the
# spool's first SSIC (EM__/LELD); it moves the player there about 2 s after the level start, StartPosition or not
NYC_PLAYER_START = bytes.fromhex('8da02045 4721303d 385567c3')
NYC_PLAYER_START_COUNT = 2
# base life event actors (LEAC: LEAD records of 16 bytes, kind in byte 0; LEPR property lists) that bring NYC's game
# into the city: 0x6A starts a mission or activity (its event loads within 300-500 m of it), 0x67 is a pickup
NYC_GAME_ACTORS = {0x6A: 'mission and activity starts', 0x67: 'pickups'}
LIFE_ACTOR_SIZE = 16
LIFE_NODE_SIZE = 12
LIFE_PROPERTY = struct.Struct('<BxHI')     # kind, string index, value size; the value follows, padded to 4 bytes
MATRIX_PROPERTY = 0x17                      # 4 x 4 floats, translation at +48
PLAYER_ACTOR = 2                            # the base life event's player character
SKIN_PROPERTY = b'Character Skin Type'      # its character id (TK = 4)
OUT_OF_REACH = (5500.0, -500.0, 5500.0)    # under the grid's far corner, over 3.5 km from any Driv3r city road
# Trains (docs/D3-PORT-KEYPOINTS.md "Trains and trams"): Driv3r's only train, the train-chase mission's locomotive and
# carriage, runs as DPL's train cars (life actors 0x74, model GLUT key 0x1C, 0x47BD24) on the city's rail line
TRAIN_FILE = 'mission31.vvv'
TRAIN_CARS = (0, 1)             # its hierarchies, front to back
TRAIN_CITY = 'miami'            # the mission plays on Miami's train territory: its cars use Miami's CarGlobals
TRAIN_GAP = 0.8                 # metres between the coupled cars
TRAIN_KEY = 0x1C
TRAIN_PAGE = 0x1000             # write_into's page key for the train's materials (no city page has it)
TRAIN_ACTOR = 0x74
TRAIN_MOVE = 0xBE               # life node MNTYPE_TRAIN_MOVE: drives one car to a target share of its track
TRAIN_STOP = 20.0               # metres from a line end to a stopped train's middle (the train is 31 m long)
MIN_RAIL_LINE = 1000.0
POSITION_PROPERTIES = (b'Initial Position', b'Target Position')     # shares of the track length (0x5B918B)
TRACK_PROPERTY = b'Track'
TRACK_HEIGHT = -300.0   # the hidden train track's height: as low as a road position packs (dpl_ai.pack_position)
TRACK_LENGTH = 400.0
TRACK_GAUGE = 2.0
SEGMENT_POINTS = 8      # points per segment of a train track road
# The base life event's district boxes (actors 69-112, kind 4): the start wires a check per borough box, a borough the
# player is in checks its district boxes, and a district hands over NYC's traffic and pedestrian lists (nodes 0x64, 0x69).
# A start outside every district (Miami's) got neither: no cars or people until the hook's fallback, two loading icons.
VOLUME_ACTOR = 4
DISTRICT_BOXES = range(69, 113)
CITY_DISTRICT = (72, 104)
PRIMITIVE_PROPERTY = 0x18       # u32 shape (2 = box), then f32 sizes along x, z and the height
BOX_SHAPE = 2
CITY_DISTRICT_SIZE = (12000.0, 12000.0, 4000.0)     # the 11.5 km grid, every city's heights, centred on the origin
DISTRICT_AWAY = (100000.0, -10000.0, 100000.0)      # far past the grid: NYC's boxes are up to 8.7 km long


def empty_gebi(gebi: bytes) -> bytes:
    """The GEBI with its header (place) kept and no instances. SCND must stay with it: without SCND the game finds no
    per-region object for the column and row and crashes (0x4DAF15)."""
    size, column, row = struct.unpack_from('<I2h', gebi, 0)
    return GEBI_HEADER.pack(size, column, row, *[0] * 7) + bytes(size - GEBI_HEADER.size)


def drop_effects(asrd: bytes, containers: set[int] | None) -> bytes:
    """The ASRD without the entries of these containers (None: all). ASRD (loader 0x4542D1): u32 entry table offset,
    u16 entry count, u16 0, u32 version 27, entries of u32 item offset, u16 item count, u16 container (sorted, then
    one more offset that ends the last entry's items); the items (effects such as NYC's barrel fires) follow."""
    table, count, version = struct.unpack_from('<IHxxI', asrd, 0)
    entries = [struct.unpack_from('<IHH', asrd, table + ASRD_ENTRY * i) for i in range(count + 1)]
    kept = [(entries[i], entries[i + 1][0]) for i in range(count) if containers is not None and entries[i][2] not in containers]
    at = table + ASRD_ENTRY * (len(kept) + 1)
    head, items = b'', b''

    for (start, n, container), end in kept:
        head += struct.pack('<IHH', at + len(items), n, container)
        items += asrd[start:end]

    return struct.pack('<IHxxI', table, len(kept), version) + head + struct.pack('<IHH', at + len(items), 0, 0) + items


def empty_city(tree: d.ChunkTree, world: set[int]) -> Counter:
    """Drops the city chunks (CITY_TAGS) from every region, super region and the global region, empties their GEBI and
    drops the effects of the models Driv3r's replace (world: NYC's placed global containers; every super region model).
    Returns the changed tags counted."""
    changed: Counter = Counter()

    for top in tree.chunks:
        if top.tree and top.tag in (b'GEIR', b'GESR', b'GEGR'):
            changed.update(c.tag.decode() for c in top.tree.chunks if c.tag in CITY_TAGS | {b'GEBI', b'ASRD'})
            # the global region's PROP stays: Driv3r's props take NYC prop slots and their records (d3_props)
            top.tree.chunks = [c for c in top.tree.chunks if c.tag not in CITY_TAGS or (top.tag, c.tag) == (b'GEGR', b'PROP')]

            for chunk in top.tree.chunks:
                if chunk.tag == b'GEBI':
                    chunk.data = empty_gebi(bytes(chunk.data))
                elif chunk.tag == b'ASRD':
                    chunk.data = drop_effects(bytes(chunk.data), world if top.tag == b'GEGR' else None)

    return changed


def write_global_pack(tree: d.ChunkTree, pages: dict[int, d3city.Page], instances: list[d3city.Instance],
                      world: set[int], props: dict[int, int], prop_slots: set[int]) -> tuple[dict[int, float], dict[int, int], int]:
    """DPL's global pack (page 0) keeps NYC's containers that game code draws by index (the weapon in the player's hand,
    muzzle flash) and its material table; Driv3r page 0's placed containers take the slots of NYC's world props (world:
    the global containers NYC's regions place). A Driv3r prop takes its NYC prop slot (props), every other model a world
    slot without a NYC prop record (prop_slots). Returns the horizontal radius per Driv3r container, the DPL slot per
    Driv3r container and a slot that draws nothing (for collision-only instances)."""
    chunk = d.find_pack(tree, GLOBAL_PAGE)
    raw = d.split_pack(chunk.data, 0, len(chunk.data))
    page = pages[GLOBAL_PAGE]
    free = sorted(world)
    plain = iter(s for s in free if s not in prop_slots)
    textures: dict[bytes, int] = {}
    materials: dict[int, int | None] = {}
    radii: dict[int, float] = {}
    slots: dict[int, int] = {}

    placed = {i.handle & 1023 for i in instances if i.handle >> 10 == GLOBAL_PAGE}

    for container in sorted(placed & set(range(len(page.containers)))):
        # props keep every near LOD entry: Driv3r's light posts carry their span arm and wire in entries 1 and 3
        baked = d3city.bake_instance(page, container, AT_ORIGIN, all_near=True)

        if baked is not None:
            slots[container] = props[container] if container in props else next(plain, None)

            if slots[container] is None:
                raise SystemExit(f"global page: more Driv3r containers than NYC's {len(free)} world prop slots")

            dpl_poc.write_into(raw, slots[container], baked, pages, textures, materials, share_stream=True)
            radii[container] = radius(baked)

    # NYC's leftover world props draw nothing
    leftover = [s for s in free if s not in slots.values()]

    for container in leftover:
        raw.lod_groups[container] = [None] * len(raw.lod_groups[container])

        for k in range(len(raw.lod_groups[container])):
            struct.pack_into('<I', raw.containers[container], 0xA8 + k * 24, 0)

    chunk.data = d.join_pack(raw)
    print(f"global pack: {len(radii)} of {len(page.containers)} containers into NYC's world prop slots, "
          f"{len(raw.containers) - len(free)} NYC code containers kept, {len(materials)} materials "
          f'({sum(v is not None for v in materials.values())} with a DPL texture), {len(raw.streams)} streams, '
          f'{len(chunk.data) >> 20} MB')
    return radii, slots, next(s for s in leftover if s not in prop_slots)


def train_model(driv3r: Path) -> tuple[d3city.Baked, dict[int, SimpleNamespace]]:
    """The train of TRAIN_FILE as one model: its cars front to back along +z, TRAIN_GAP apart, the middle at the origin
    and the wheels' bottom at y 0 (DPL's train cars stand on their track there). Lamp glows are left out (DPL draws them
    as plain quads). Returns it with write_into's page table for its materials (TRAIN_PAGE)."""
    data, cars = d3_vehicle_swap.driv3r_cars(driv3r, TRAIN_CITY, TRAIN_FILE)
    shared = d3_vehicle_swap.driv3r_globals(driv3r, TRAIN_CITY)
    parts, front, page = [], 0.0, None

    for car in TRAIN_CARS:
        _, pack, upvh = cars[car]
        page = d3_vehicle_swap.driv3r_page(data, pack)
        baked = [b for c in sorted(d3_vehicle_swap.drawn_containers(data, upvh)) if c < len(page.containers)
                 for b in [d3city.bake_instance(page, c, AT_ORIGIN)] if b is not None]
        whole = merge(baked)
        kept = np.array([not d3_vehicle_swap.is_light(m) and m is not None
                         for m in (d3_vehicle_swap.d3_material(page, shared, k & 0xFFFF) for k in whole.materials.tolist())])
        whole = d3city.Baked(whole.vertices, whole.triangles[kept], whole.materials[kept] & 0xFFFF | TRAIN_PAGE << 16)
        whole.vertices[:, 2] += front - whole.vertices[:, 2].max()
        front = float(whole.vertices[:, 2].min()) - TRAIN_GAP
        parts.append(whole)

    train = merge(parts)
    low, high = train.vertices[:, 0:3].min(axis=0), train.vertices[:, 0:3].max(axis=0)
    train.vertices[:, 0:3] -= [0.0, low[1], (low[2] + high[2]) / 2]
    # the cars of one .vvv share one pack (driv3r_cars), so one page serves both
    return train, {TRAIN_PAGE: SimpleNamespace(material=lambda k: d3_vehicle_swap.d3_material(page, shared, k))}


def glut_models(tree: d.ChunkTree) -> dict[int, int]:
    """GLUT of the global region (u32 count, then u16 key, u16 container): the global container per game-code model key.
    Life actors (gadgets, mission objects, train cars) draw their model through it (0x455C16)."""
    glut = next(c for top in tree.chunks if top.tree and top.tag == b'GEGR' for c in top.tree.chunks if c.tag == b'GLUT')
    count = struct.unpack_from('<I', glut.data, 0)[0]
    pairs = struct.unpack_from(f'<{2 * count}H', glut.data, 4)
    return dict(zip(pairs[0::2], pairs[1::2]))


def write_train_model(tree: d.ChunkTree, driv3r: Path) -> str:
    """Puts the Driv3r train into the global container the game draws DPL's train cars with: GLUT (the global region's
    u32 count, then u16 key, u16 container) under TRAIN_KEY. NYC's regions place no model there (write_global_pack keeps it)."""
    glut = next(c for top in tree.chunks if top.tree and top.tag == b'GEGR' for c in top.tree.chunks if c.tag == b'GLUT')
    count = struct.unpack_from('<I', glut.data, 0)[0]
    pairs = struct.unpack_from(f'<{2 * count}H', glut.data, 4)
    container = dict(zip(pairs[0::2], pairs[1::2]))[TRAIN_KEY]
    chunk = d.find_pack(tree, GLOBAL_PAGE)
    raw = d.split_pack(chunk.data, 0, len(chunk.data))
    train, pages = train_model(driv3r)
    converted = dpl_poc.write_into(raw, container, train, pages, share_stream=True)
    chunk.data = d.join_pack(raw)
    size = train.vertices[:, 0:3].max(axis=0) - train.vertices[:, 0:3].min(axis=0)
    return f'{TRAIN_FILE} cars {TRAIN_CARS} into global container {container:#x}: {np.round(size, 1)} m, ' \
           f'{len(train.triangles)} triangles, {converted} materials'


def radius(baked: d3city.Baked) -> float:
    """Horizontal reach of the vertices from their origin."""
    return float(np.hypot(baked.vertices[:, 0], baked.vertices[:, 2]).max())


def merge(parts: list[d3city.Baked]) -> d3city.Baked:
    offsets = np.cumsum([0] + [len(b.vertices) for b in parts])[:-1]
    return d3city.Baked(np.concatenate([b.vertices for b in parts]), np.concatenate([b.triangles + o for b, o in zip(parts, offsets)]),
                        np.concatenate([b.materials for b in parts]))


def region_key(x: float, z: float) -> tuple[int, int]:
    """(column, row) of the DPL region holding the point: columns count from x = -5760, rows from z = +5760 down (the
    min corners in SCND); region index = row * GRID_SIDE + column."""
    return int((x - GRID_ORIGIN) // REGION_SIZE), int((-GRID_ORIGIN - z) // REGION_SIZE)


def yaw(inst: d3city.Instance) -> tuple[float, float] | None:
    """DPL's (cos, sin) of a yaw-only instance, None for any other turn. Driv3r's turn about y by a (v' = M v) is DPL's
    row-vector yaw with cos a and -sin a."""
    if abs(inst.rotation[0]) > YAW_ONLY or abs(inst.rotation[2]) > YAW_ONLY:
        return None

    a = 2 * math.atan2(inst.rotation[1], inst.rotation[3])
    return math.cos(a), -math.sin(a)


def stand_props(instances: list[d3city.Instance], props: dict[int, int]) -> int:
    """Keeps only the yaw of each global prop tilted less than PROP_TILT, so it is placed (and topples) instead of
    baked static. Returns the count."""
    count = 0

    for k, i in enumerate(instances):
        x, y, z, w = i.rotation
        if (i.handle >> 10 == GLOBAL_PAGE and i.handle & 1023 in props and yaw(i) is None
                and 1 - 2 * (x * x + z * z) > math.cos(math.radians(PROP_TILT))):
            n = math.hypot(y, w)
            instances[k] = replace(i, rotation=(0.0, y / n, 0.0, w / n))
            count += 1

    return count


@dataclass
class Placement:
    x: float
    y: float
    z: float
    yaw_cos: float
    yaw_sin: float
    handle: int
    radius: float
    objects: list[tuple[int, float, float, float, float]] = field(default_factory=list)  # (GEPM id << 11 | record, box)
    region: tuple[int, int] | None = None       # set on a border copy: the neighbour region it stands in
    gebi: tuple[int, int, int] | None = None    # set by write_regions: (column, row, instance index)
    prop: bool = False                          # topples when hit: GEBI kind PROP_KIND


def merge_records(records: list[d.PdlRecord]) -> list[d.PdlRecord]:
    """The records' parts regrouped into as few records as DPL's own have parts (nearest together), each with its
    bounding part: one broadphase object and GEPM record per group instead of per Driv3r record."""
    parts = sorted((p for r in records for p in (r.parts[1:] if len(r.parts) > 1 else r.parts)),
                   key=lambda p: struct.unpack_from('<3f', p, 64)[0::2])
    n = dpl_poc.MAX_RECORD_PARTS - 1
    return [dpl_poc.bounded(parts[i:i + n]) for i in range(0, len(parts), n)]


Shape = tuple[int, tuple[tuple[int, int], ...]]     # Driv3r handle, the instance's own (GEPM id, record) pairs


class CollisionBuilder:
    """Driv3r collision as DPL records, refilling the GEPM of the global region (id 0) and of each super region, and the
    broadphase objects of each placement. Records are in their model's space, as DPL's own: the game turns and moves
    them with the instance."""

    def __init__(self, tree: d.ChunkTree, collision: d3city.Collision) -> None:
        self.collision = collision
        self.gepms = {struct.unpack_from('<I', c.data, 0)[0]: c for top in tree.chunks if top.tree and top.tag in (b'GEGR', b'GESR')
                      for c in top.tree.chunks if c.tag == b'GEPM'}
        self.records: dict[int, list[d.PdlRecord]] = defaultdict(list)
        self.converted: dict[Shape, list[d.PdlRecord]] = {}     # shape -> model-space records
        self.added: dict[tuple[int, Shape], list[int]] = {}     # (GEPM id, shape) -> codes

    @staticmethod
    def shape(inst: d3city.Instance) -> Shape:
        """What decides an instance's collision: its model, and its own broadphase records where it has them (one
        model, such as a road piece, can stand in many places with a different surface each)."""
        return inst.handle, tuple(inst.collision)

    def model(self, inst: d3city.Instance) -> list[d.PdlRecord]:
        key = self.shape(inst)

        if key not in self.converted:
            parts = [d3city.place_parts(r, AT_ORIGIN) for r in self.collision.records(inst)]
            self.converted[key] = merge_records(dpl_poc.collision_records(parts, np.zeros(3)))

        return self.converted[key]

    def add(self, gepm_id: int, records: list[d.PdlRecord]) -> list[int]:
        store = self.records[gepm_id]

        if len(store) + len(records) > d.MAX_PDL_RECORDS:
            raise SystemExit(f'GEPM {gepm_id}: more than {d.MAX_PDL_RECORDS} records')

        store += records
        return [gepm_id << 11 | i for i in range(len(store) - len(records), len(store))]

    def objects(self, codes: list[int], records: list[d.PdlRecord], x: float, z: float, yaw_cos: float,
                yaw_sin: float) -> list[tuple[int, float, float, float, float]]:
        """(code, world box x0, z0, x1, z1) per record of an instance at (x, z) with this yaw (DPL row-vector matrix)."""
        out = []

        for code, record in zip(codes, records):
            corners = np.concatenate([dpl_poc.part_corners(p) for p in record.parts])
            wx = x + corners[:, 0] * yaw_cos - corners[:, 2] * yaw_sin
            wz = z + corners[:, 0] * yaw_sin + corners[:, 2] * yaw_cos
            out.append((code, float(wx.min()), float(wz.min()), float(wx.max()), float(wz.max())))

        return out

    def placed(self, gepm_id: int, inst: d3city.Instance, turn: tuple[float, float]) -> list[tuple[int, float, float, float, float]]:
        """The broadphase objects of a yaw-only instance; its records go into the GEPM once per shape."""
        records = self.model(inst)
        key = gepm_id, self.shape(inst)

        if key not in self.added:
            self.added[key] = self.add(gepm_id, records)

        return self.objects(self.added[key], records, inst.position[0], inst.position[2], *turn)

    def baked(self, gepm_id: int, instances: list[d3city.Instance], centre: np.ndarray) -> list[tuple[int, float, float, float, float]]:
        """The broadphase objects of a container of these instances baked around centre (yaw 0)."""
        parts = [d3city.place_parts(r, i) for i in instances for r in self.collision.records(i)]
        records = merge_records(dpl_poc.collision_records(parts, centre)) if parts else []
        return self.objects(self.add(gepm_id, records), records, centre[0], centre[2], 1.0, 0.0)

    def write(self) -> Counter:
        """Refills every GEPM with its new records only (NYC's are dropped)."""
        counts: Counter = Counter()

        for gepm_id, chunk in self.gepms.items():
            data = bytes(chunk.data)
            table = struct.unpack_from('<I', data, d.GEPM_HEADER + 8)[0]
            empty = bytearray(data[:d.GEPM_HEADER + table])
            struct.pack_into('<4I', empty, d.GEPM_HEADER, 0, 0, table, table)
            chunk.data = d.append_pdl_records(bytes(empty), self.records.get(gepm_id, []))
            counts['records'] += len(self.records.get(gepm_id, []))

        counts['GEPMs'] = len(self.gepms)
        return counts


def excess_props(instances: list[d3city.Instance], radii: dict[int, float], collision: CollisionBuilder,
                 props: dict[int, int]) -> set[int]:
    """Identities (id()) of the global props that do not fit their region's GEBI or broadphase, the smallest static
    ones first (a baked prop no longer topples): write_super_regions bakes them in place. BAKE_ROOM records and objects
    stay free for the region's baked containers."""
    placed = defaultdict(list)

    for i in instances:
        if yaw(i) is not None:
            placed[region_key(i.position[0], i.position[2])].append(i)

    excess = set()

    for group in placed.values():
        count, objects = len(group), sum(len(collision.model(i)) for i in group)
        small = sorted((i for i in group if i.handle >> 10 == GLOBAL_PAGE and i.handle & 1023 in radii),
                       key=lambda i: (i.handle & 1023 in props, radii[i.handle & 1023]))

        while small and (count > MAX_REGION_INSTANCES - BAKE_ROOM or objects > MAX_REGION_OBJECTS - BAKE_ROOM):
            prop = small.pop(0)
            excess.add(id(prop))
            count -= 1
            objects -= len(collision.model(prop))

    return excess


def reach(x: float, z: float, objects: list[tuple[int, float, float, float, float]]) -> float:
    """Horizontal radius around (x, z) that holds these broadphase boxes."""
    return max((math.hypot(max(abs(x0 - x), abs(x1 - x)), max(abs(z0 - z), abs(z1 - z))) for _, x0, z0, x1, z1 in objects),
               default=0.0)


def global_placements(instances: list[d3city.Instance], radii: dict[int, float], slots: dict[int, int], hidden: int,
                      excess: set[int], collision: CollisionBuilder, props: dict[int, int]) -> list[Placement]:
    """The yaw-only instances of the written global containers (at their DPL slot), but the excess ones
    (write_super_regions bakes those and the turned ones), with their collision in the global region's GEPM.
    Collision-only instances stand on the hidden handle, a global container that draws nothing."""
    placements = [Placement(*i.position, *turn, GLOBAL_PAGE << 10 | slots[i.handle & 1023], radii[i.handle & 1023],
                            collision.placed(GLOBAL_PAGE, i, turn), prop=i.handle & 1023 in props)
                  for i in instances if i.handle >> 10 == GLOBAL_PAGE and i.handle & 1023 in radii
                  and (turn := yaw(i)) is not None and id(i) not in excess]

    for i in instances:
        if i.handle & 1023 == NO_MODEL and (turn := yaw(i)) is not None:
            objects = collision.placed(GLOBAL_PAGE, i, turn)
            placements.append(Placement(*i.position, *turn, hidden, reach(i.position[0], i.position[2], objects), objects))

    return placements


def sea_containers(tree: d.ChunkTree, world: set[int]) -> tuple[int, int]:
    """Two of NYC's open-sea tile slots (water, then sea bed): placed global containers whose box is a flat
    region-sized square (NYC places 37 of them, a 4-vertex quad of global material 70 each, at y -3)."""
    pack = d.load_pack(d.find_pack(tree, GLOBAL_PAGE).data, 0)
    tiles = [k for k in sorted(world) if (box := list(zip(*pack.containers[k].box))) and np.ptp(box[1]) < SEA_TILE_FLATNESS
             and min(np.ptp(box[0]), np.ptp(box[2])) >= REGION_SIZE]

    if len(tiles) < 2:
        raise SystemExit('global page: fewer than two NYC sea tiles found')

    return tiles[0], tiles[1]


def write_sea_tiles(tree: d.ChunkTree, water: int, bed: int) -> None:
    """Rewrites two NYC sea quads with tiled UVs (one texture repeat per 16 m, as DPL's coastal water: about 17 m), both
    faces. The water takes DPL's near water material (SEA_WATER_MATERIAL; NYC's open-sea material 70 stayed invisible
    near the camera). DPL's water blends over a bed: with nothing under it the clear colour showed through, so the bed
    gets NYC's river floor texture (SEA_BED_MATERIAL) as an opaque material."""
    chunk = d.find_pack(tree, GLOBAL_PAGE)
    pack = d.load_pack(chunk.data, 0)
    mud = d3city.Material((0, dpl_poc.DRIV3R_ENV_MAP), d.material_texture(chunk.data, pack, SEA_BED_MATERIAL))
    half, repeat = REGION_SIZE / 2, SEA_UV_REPEAT
    corners = [(-half, -half, 0, 0), (half, -half, repeat, 0), (-half, half, 0, repeat), (half, half, repeat, repeat)]
    baked = d3city.Baked(np.array([(x, 0.0, z, 0.0, 1.0, 0.0, u, v) for x, z, u, v in corners]),
                         np.array([(0, 1, 2), (2, 1, 3), (0, 2, 1), (2, 3, 1)]), np.zeros(4, dtype=np.int64))
    raw = d.split_pack(chunk.data, 0, len(chunk.data))
    # no material found: write_into keeps the container's mesh material word (70); its index becomes the near water's
    dpl_poc.write_into(raw, water, baked, {GLOBAL_PAGE: SimpleNamespace(material=lambda _: None)}, share_stream=True)
    struct.pack_into('<H', raw.meshes[-1], 48, SEA_WATER_MATERIAL)
    dpl_poc.write_into(raw, bed, baked, {GLOBAL_PAGE: SimpleNamespace(material=lambda _: mud)}, share_stream=True)
    chunk.data = d.join_pack(raw)


def water_height(d3c: Path) -> float:
    """The Driv3r city's water level: GWTR (in a GESR) starts with its vertex count, a count and the height (Miami -2.5;
    every water vertex after it has that y)."""
    data = d3c.read_bytes()
    gwtr = next(c for top in d.read_tree(data).chunks if top.tree and top.tag == b'GESR'
                for c in top.tree.chunks if c.tag == b'GWTR')
    return struct.unpack_from('<f', gwtr.data, 8)[0]


def write_sea(tree: d.ChunkTree, water: int, bed: int, height: float) -> int:
    """The water tile SEA_DRAW_DROP under the water height and the bed tile SEA_BED_DEPTH under that, under every region
    of the grid, as NYC places its sea tiles: in the global region's GEBI (all drawn only) with its scene tree. A
    region's scene tree cannot hold a region-sized box: it went in by its centre and was culled with that small cell.
    Where the city stands its ground covers both: Driv3r draws its water only in its water area (GWRD), so its roads may
    lie at the water height, and a tile at that height fought them (docs/D3-SOLVED-CASES.md M10)."""
    chunks = {c.tag: c for top in tree.chunks if top.tree and top.tag == b'GEGR' for c in top.tree.chunks}
    half = REGION_SIZE / 2
    height -= SEA_DRAW_DROP
    centres = [(GRID_ORIGIN + column * REGION_SIZE + half, -GRID_ORIGIN - row * REGION_SIZE - half)
               for row in range(GRID_SIDE) for column in range(GRID_SIDE)]
    tiles = [(x, y, z, k) for x, z in centres for y, k in ((height - SEA_BED_DEPTH, bed), (height, water))]
    gebi = bytes(chunks[b'GEBI'].data)
    size, column, row = struct.unpack_from('<I2h', gebi, 0)
    records = [region_record(i, x, y, z, 1.0, 0.0, GLOBAL_PAGE << 10 | k) for i, (x, y, z, k) in enumerate(tiles)]
    ends = [0] * DRAWN_ONLY + [len(records)] * (KINDS - DRAWN_ONLY)
    chunks[b'GEBI'].data = GEBI_HEADER.pack(size, column, row, len(records), *ends) + gebi[GEBI_HEADER.size:size] + b''.join(records)
    chunks[b'SCND'].data = scene_tree(bytes(chunks[b'SCND'].data), [(x - half, z - half, x + half, z + half) for x, _, z, _ in tiles])
    return len(centres)


def border_copies(placements: list[Placement], hidden: int) -> list[Placement]:
    """A region's broadphase only holds boxes inside the region, so collision reaching into a neighbour is missed there
    (a bridge deck 108 m over the edge): per neighbour, a copy of the placement on the hidden handle with those boxes."""
    copies = []

    for p in placements:
        home = region_key(p.x, p.z)
        per_region: dict[tuple[int, int], list[tuple[int, float, float, float, float]]] = defaultdict(list)

        for obj in p.objects:
            (c0, r1), (c1, r0) = region_key(obj[1], obj[2]), region_key(obj[3], obj[4])

            for key in ((c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)):
                if key != home and 0 <= key[0] < GRID_SIDE and 0 <= key[1] < GRID_SIDE:
                    per_region[key].append(obj)

        for key, objects in per_region.items():
            copies.append(Placement(p.x, p.y, p.z, p.yaw_cos, p.yaw_sin, hidden, reach(p.x, p.z, objects), objects, key))

    return copies


def fresh_pack(raw: d.RawPack) -> d.RawPack:
    """An empty copy of the pack for new models: its header and its first container with that container's first group,
    mesh and one vertex as the templates write_into copies; no NYC models, materials or textures."""
    container = next(i for i, lods in enumerate(raw.lod_groups) if any(g is not None for g in lods))
    lods = raw.lod_groups[container]
    entry = next(k for k, g in enumerate(lods) if g is not None)
    stream = struct.unpack_from('<H', raw.containers[container], 0x18)[0]
    record = bytearray(raw.containers[container])
    struct.pack_into('<H', record, 0x18, 0)

    for k in range(len(lods)):
        if k != entry:
            struct.pack_into('<I', record, 0xA8 + k * 24, 0)

    group = bytearray(raw.groups[lods[entry]])
    struct.pack_into('<H', group, 0x44, 1)
    mesh = bytearray(raw.meshes[raw.group_meshes[lods[entry]]])
    struct.pack_into('<I', mesh, 12, 0)
    # its NYC material may lie past the new table: the loader checks every mesh (0x5EE51E)
    struct.pack_into('<Hh', mesh, 48, 0, struct.unpack_from('<I', raw.header, 4)[0])
    pcmp = replace(raw.pcmp, materials=[], material_pointers=[], pointers=[], substances=[], substance_slots=[], slots=[],
                   textures=[], dds=[])
    return d.RawPack(raw.header, [record], [[0 if k == entry else None for k in range(len(lods))]], [group], [0], [mesh], [0],
                     [bytearray(raw.descriptors[stream])], bytearray(), [bytearray(raw.streams[stream][:d.VERTEX_STRIDE])], pcmp)


def super_region_block(c: int, r: int) -> tuple[int, int]:
    """The block of a region. Every other block column is shifted by one region, like bricks: DPL wants every super region
    with a region within 210 m of the camera resident at once and has 3 slots, so where 4 aligned blocks met the game
    waited for ever with its loading icon. Bricks meet 3 at most (the radius is far under half a region)."""
    column = c // SUPER_REGION_BLOCK
    return column, (r + column % 2) // SUPER_REGION_BLOCK


def super_region_slots(keys: set[tuple[int, int]], available: int) -> dict[tuple[int, int], int]:
    """Super region slot (1.., GSIL index) per region: blocks of SUPER_REGION_BLOCK x SUPER_REGION_BLOCK regions."""
    blocks = sorted({super_region_block(c, r) for c, r in keys})

    if len(blocks) > available:
        raise SystemExit(f'{len(blocks)} super regions needed, the template has {available}')

    return {(c, r): blocks.index(super_region_block(c, r)) + 1 for c, r in keys}


def write_super_regions(tree: d.ChunkTree, pages: dict[int, d3city.Page], instances: list[d3city.Instance],
                        excess: set[int], collision: CollisionBuilder) -> list[Placement]:
    """Each block of regions becomes one DPL super region (GESI region map): its pack holds the Driv3r containers of its
    yaw-only instances, one copy per block, and per region one or more containers of its other instances (turned ones,
    and the excess global props) baked in place around the region centre. Returns the placements (DPL handles)."""
    per_region: dict[tuple[int, int], list[d3city.Instance]] = defaultdict(list)

    for inst in instances:
        page = pages.get(inst.handle >> 10)

        if page and inst.handle & 1023 < len(page.containers) and \
                (inst.handle >> 10 != GLOBAL_PAGE or yaw(inst) is None or id(inst) in excess):
            per_region[region_key(inst.position[0], inst.position[2])].append(inst)

    gegl = next(t for t in tree.chunks if t.tag == b'GEGL')
    gesi_chunk = next(c for c in gegl.tree.chunks if c.tag == b'GESI')
    gesi = bytearray(gesi_chunk.data)
    # GESI: u32 super region count (the global region included), region count, f32 region size, offset of a byte per
    # region: its super region (GSIL index, 0 = the global region)
    super_count, regions, _, map_off = struct.unpack_from('<2IfI', gesi, 0)
    slots = super_region_slots(set(per_region), super_count - 1)
    gesi[map_off:map_off + regions] = bytes(regions)

    for (column, row), slot in slots.items():
        gesi[map_off + row * GRID_SIDE + column] = slot

    gesi_chunk.data = bytes(gesi)
    gsil = bytes(next(c for c in gegl.tree.chunks if c.tag == b'GSIL').data)
    placements: list[Placement] = []

    for slot in sorted(set(slots.values())):
        super_region = tree.chunks[struct.unpack_from('<I', gsil, slot * 4)[0]]
        chunk = next(c for c in super_region.tree.chunks if c.tag == b'MDXN')
        page_id = struct.unpack_from('<I', chunk.data, 4)[0]
        gepm_id = next(struct.unpack_from('<I', c.data, 0)[0] for c in super_region.tree.chunks if c.tag == b'GEPM')
        keys = [k for k, s in slots.items() if s == slot]
        # every near LOD entry, as for props: Driv3r's entries add up (a bridge's deck and the strips over its seams sit
        # in entries 1-3); a pack that would grow past DPL's ceiling that way keeps only each model's first entry
        all_near = len(d.join_pack(fill_super_region(chunk, page_id, keys, per_region, pages, True))) <= MAX_PACK
        raw = fill_super_region(chunk, page_id, keys, per_region, pages, all_near, (gepm_id, collision, placements))
        chunk.data = d.join_pack(raw)
        print(f'super region {slot} (page {page_id}): {len(raw.containers)} containers, {len(raw.pcmp.materials)} '
              f'materials, {len(raw.streams)} streams, {len(chunk.data) >> 20} MB'
              + ('' if all_near else ', first LOD entry only (every entry would pass the pack ceiling)'))

    # Unused slots still load: their NYC meshes point at global materials past the new global pack's table (crash
    # 0x5EE4F0). They get the same layout as the used packs: one container, here a centimetre triangle nobody places.
    material = next(m for m in range(0x10000) if pages[GLOBAL_PAGE].material(m))
    stub = d3city.Baked(np.array([[0, 0, 0, 0, 1, 0, 0, 0], [0.01, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0.01, 0, 1, 0, 0, 1]]),
                        np.array([[0, 1, 2]]), np.array([GLOBAL_PAGE << 16 | material]))

    for slot in sorted(set(range(1, super_count)) - set(slots.values())):
        chunk = next(c for c in tree.chunks[struct.unpack_from('<I', gsil, slot * 4)[0]].tree.chunks if c.tag == b'MDXN')
        raw = fresh_pack(d.split_pack(chunk.data, 0, len(chunk.data)))
        dpl_poc.write_into(raw, 0, stub, pages, share_stream=True)
        chunk.data = d.join_pack(raw)
        print(f'super region {slot}: unused, stub pack')

    return placements


def fill_super_region(chunk: d.Chunk, page_id: int, keys: list[tuple[int, int]],
                      per_region: dict[tuple[int, int], list[d3city.Instance]], pages: dict[int, d3city.Page],
                      all_near: bool, place: tuple[int, CollisionBuilder, list[Placement]] | None = None) -> d.RawPack:
    """The super region's pack with the models of its regions' instances: yaw-only ones once per Driv3r container,
    turned ones baked in place around their region centre. With place (GEPM id, collision, placements) the instances
    are also placed and get their collision; without it the pack is only built (to measure it)."""
    raw = fresh_pack(d.split_pack(chunk.data, 0, len(chunk.data)))
    template = (bytes(raw.containers[0]), list(raw.lod_groups[0]))
    textures: dict[bytes, int] = {}
    materials: dict[int, int | None] = {}
    handles: dict[int, tuple[int, float] | None] = {}      # Driv3r handle -> (DPL handle, radius), None: nothing to draw
    written: list[int] = []

    def add(baked: d3city.Baked) -> int:
        """Writes the model into the next container (the template's own first) and returns its DPL handle."""
        if written:
            raw.containers.append(bytearray(template[0]))
            raw.lod_groups.append(list(template[1]))

        if len(raw.containers) > MAX_CONTAINERS:
            raise SystemExit(f'page {page_id}: more than {MAX_CONTAINERS} containers')

        written.append(len(raw.containers) - 1)
        dpl_poc.write_into(raw, written[-1], baked, pages, textures, materials, share_stream=True)
        return page_id << 10 | written[-1]

    for key in keys:
        turned = []

        for inst in per_region[key]:
            turn = yaw(inst)

            if turn is None or inst.handle >> 10 == GLOBAL_PAGE:
                turned.append(inst)
                continue

            if inst.handle not in handles:
                baked = d3city.bake_instance(pages[inst.handle >> 10], inst.handle & 1023, AT_ORIGIN, all_near)
                handles[inst.handle] = None if baked is None else (add(baked), radius(baked))

            if handles[inst.handle] and place:
                gepm_id, collision, placements = place
                placements.append(Placement(*inst.position, *turn, *handles[inst.handle],
                                            collision.placed(gepm_id, inst, turn)))

        centre = np.array([GRID_ORIGIN + (key[0] + 0.5) * REGION_SIZE, 0.0, -GRID_ORIGIN - (key[1] + 0.5) * REGION_SIZE])
        parts = [(i, b) for i in turned if (b := d3city.bake_instance(pages[i.handle >> 10], i.handle & 1023, i,
                                                                     all_near or i.handle >> 10 == GLOBAL_PAGE)) is not None]

        while parts:
            take, count = [], 0

            while parts and count + len(parts[0][1].vertices) <= d.MAX_STREAM_VERTICES:
                count += len(parts[0][1].vertices)
                take.append(parts.pop(0))

            if not take:
                raise SystemExit(f'region {key}: a turned model of {len(parts[0][1].vertices)} vertices')

            baked = merge([b for _, b in take])
            baked.vertices[:, 0:3] -= centre
            handle = add(baked)

            if place:
                gepm_id, collision, placements = place
                placements.append(Placement(*centre, 1.0, 0.0, handle, radius(baked),
                                            collision.baked(gepm_id, [i for i, _ in take], centre)))

    return raw


def region_record(index: int, x: float, y: float, z: float, yaw_cos: float, yaw_sin: float, handle: int) -> bytes:
    """One GEBI v10 record, as dplchunk.region_instances decodes it. x and z round down: rounding up could put a point
    just inside a region's west or south edge on that edge, in the next region."""
    return GEBI_RECORD.pack((round(yaw_cos * 32767) + 0x7FFF) << 16 | (round(yaw_sin * 32767) + 0x7FFF),
                            math.floor((x - GRID_ORIGIN) * 64) | W1_FLAGS << 20,
                            (round(y * 128) + 0x3200) | handle << 16,
                            math.floor((z - GRID_ORIGIN) * 64) | index << 20)


def scene_tree(scnd: bytes, boxes: list[tuple[float, float, float, float]]) -> bytes:
    """The region's SCND rebuilt for these instance boxes (x0, z0, x1, z1; box k = instance k), header kept."""
    size, min_x, min_z = struct.unpack_from('<3f', scnd, 0x14)
    # broadphase_insert reads the cell size and min corner at +0x20 and the maximum depth at +0x34 of a JNR! header
    header = bytearray(0x38)
    struct.pack_into('<3f', header, 0x20, size, min_x, min_z)
    struct.pack_into('<I', header, 0x34, SCND_MAX_DEPTH)
    bp = d.Broadphase(bytes(header), [[0] * 8], [[]], b'')

    # largest first: they need the shallow nodes, which small boxes would otherwise fill (a baked region piece went in by
    # its centre and vanished whenever that small cell left the view)
    for index in sorted(range(len(boxes)), key=lambda k: (boxes[k][0] - boxes[k][2]) + (boxes[k][1] - boxes[k][3])):
        x0, z0, x1, z1 = boxes[index]
        mid_x, mid_z = (x0 + x1) / 2, (z0 + z1) / 2

        # limit: more than 15 large boxes around one spot fill their node; the rest go in by their centre, so only the
        # loose cell (half a cell around) keeps them from being culled, upgrade: when props vanish at the screen edge
        if not (d.broadphase_insert(bp, index, x0, z0, x1, z1) or d.broadphase_insert(bp, index, mid_x, mid_z, mid_x, mid_z)):
            raise SystemExit(f'scene tree: no room for instance {index}')

    out = bytearray(scnd[:SCND_HEADER])
    first = 0

    for node, objects in zip(bp.nodes, bp.objects):
        out += struct.pack('<8H', len(objects) << 12 | first, *node[1:])
        first += len(objects)

    objects_off = len(out)

    for objects in bp.objects:
        out += struct.pack(f'<{len(objects)}I', *objects)

    struct.pack_into('<4I', out, 0, SCND_HEADER, objects_off, len(bp.nodes), first)
    return bytes(out)


def add_region_chunks(region: d.Chunk, template: dict[bytes, d.Chunk], column: int, row: int) -> None:
    """Gives a region without models (NYC water: only GEBR and ATRD) an empty GEBI and a SCND header in front, as the
    template region has them."""
    scnd = bytearray(bytes(template[b'SCND'].data)[:SCND_HEADER])
    struct.pack_into('<3f2i', scnd, 0x14, REGION_SIZE, GRID_ORIGIN + column * REGION_SIZE,
                     -GRID_ORIGIN - (row + 1) * REGION_SIZE, column, row)
    new = {b'GEBI': GEBI_HEADER.pack(GEBI_HEADER.size, column, row, *[0] * 7), b'SCND': bytes(scnd)}

    for tag in (b'SCND', b'GEBI'):
        region.tree.chunks.insert(0, replace(template[tag], data=new[tag], tree=None))


def broadphase(template: d.Chunk, column: int, row: int, placed: list[Placement]) -> bytes:
    """The region's JNR! for the placements' collision objects (instance index << 16 | code), from a NYC template's
    header with this region's min corner."""
    data = bytes(template.data)
    header = bytearray(data[:0x10 + struct.unpack_from('<I', data, 0x10)[0]])
    struct.pack_into('<2f2I', header, 0x24, GRID_ORIGIN + column * REGION_SIZE, -GRID_ORIGIN - (row + 1) * REGION_SIZE, column, row)
    bp = d.Broadphase(bytes(header), [[0] * 8], [[]], b'')
    lo_x, lo_z = struct.unpack_from('<2f', header, 0x24)
    hi_x, hi_z = lo_x + REGION_SIZE - 0.01, lo_z + REGION_SIZE - 0.01

    # a box reaching over the region's edge keeps to the region: only cells inside it can hold it
    boxes = [(index << 16 | code, *sorted(min(max(v, lo_x), hi_x) for v in box[0::2]),
              *sorted(min(max(v, lo_z), hi_z) for v in box[1::2]))
             for index, p in enumerate(placed) for code, *box in p.objects]

    # largest first, as in scene_tree
    for obj, x0, x1, z0, z1 in sorted(boxes, key=lambda b: (b[1] - b[2]) + (b[3] - b[4])):
        mid_x, mid_z = (x0 + x1) / 2, (z0 + z1) / 2

        # limit: as in scene_tree, a crowd of large boxes goes in by its centre, upgrade: when collision misses
        if not (d.broadphase_insert(bp, obj, x0, z0, x1, z1) or d.broadphase_insert(bp, obj, mid_x, mid_z, mid_x, mid_z)):
            raise SystemExit(f'region ({column}, {row}): no broadphase room for instance {obj >> 16}')

    return d.join_broadphase(bp)


def write_regions(tree: d.ChunkTree, placements: list[Placement], jnr_template: d.Chunk) -> Counter:
    """Puts the placements into the GEBI of the DPL region holding them, those with collision first (kind 0, NYC's
    buildings) and the others drawn only (kind 5), with a new SCND and JNR!."""
    regions = {top.name: top for top in tree.chunks if top.tree and top.tag == b'GEIR'}
    template = next(chunks for top in regions.values() if b'GEBI' in (chunks := {c.tag: c for c in top.tree.chunks}))
    per_region: dict[tuple[int, int], list[Placement]] = defaultdict(list)
    counts: Counter = Counter()

    for p in placements:
        if MIN_HEIGHT <= p.y < MAX_HEIGHT:
            per_region[p.region or region_key(p.x, p.z)].append(p)
        else:
            counts['outside the GEBI height range'] += 1

    for key, placed in per_region.items():
        region = regions[b'R_%03d' % (key[1] * GRID_SIDE + key[0])]

        if not any(c.tag == b'GEBI' for c in region.tree.chunks):
            add_region_chunks(region, template, *key)
            counts['regions given GEBI and SCND'] += 1

        chunks = {c.tag: c for c in region.tree.chunks}

        if len(placed) > MAX_REGION_INSTANCES:
            raise SystemExit(f'region {key}: {len(placed)} instances, at most {MAX_REGION_INSTANCES}')

        # kind 0: solid, PROP_KIND: props (they topple when hit), DRAWN_ONLY: no collision
        placed.sort(key=lambda p: 1 if p.prop else 0 if p.objects else 2)
        solid = sum(1 for p in placed if p.objects and not p.prop)
        props = sum(1 for p in placed if p.prop)
        records, boxes = [], []

        lo_x, hi_z = GRID_ORIGIN + key[0] * REGION_SIZE, -GRID_ORIGIN - key[1] * REGION_SIZE

        for index, p in enumerate(placed):
            records.append(region_record(index, p.x, p.y, p.z, p.yaw_cos, p.yaw_sin, p.handle))
            p.gebi = (*key, index)

            if p.region:    # a border copy stands outside its region and draws nothing: its box keeps to the region
                boxes.append((max(lo_x, min(o[1] for o in p.objects)), max(hi_z - REGION_SIZE, min(o[2] for o in p.objects)),
                              min(lo_x + REGION_SIZE - 0.01, max(o[3] for o in p.objects)), min(hi_z - 0.01, max(o[4] for o in p.objects))))
            else:
                boxes.append((p.x - p.radius, p.z - p.radius, p.x + p.radius, p.z + p.radius))

        gebi = bytes(chunks[b'GEBI'].data)
        size, column, row = struct.unpack_from('<I2h', gebi, 0)
        ends = [solid] * PROP_KIND + [solid + props] * (DRAWN_ONLY - PROP_KIND) + [len(placed)] * (KINDS - DRAWN_ONLY)
        chunks[b'GEBI'].data = GEBI_HEADER.pack(size, column, row, len(placed), *ends) + gebi[GEBI_HEADER.size:size] + b''.join(records)
        chunks[b'SCND'].data = scene_tree(bytes(chunks[b'SCND'].data), boxes)
        jnr = replace(jnr_template, data=broadphase(jnr_template, column, row, placed), tree=None)

        if b'JNR!' in chunks:
            chunks[b'JNR!'].data = jnr.data
        else:
            region.tree.chunks.insert(region.tree.chunks.index(chunks[b'SCND']) + 1, jnr)

        counts['placed'] += len(placed)
        counts['with collision'] += solid
        counts['props'] += props
        counts['broadphase objects'] += sum(len(p.objects) for p in placed)
        counts['regions'] += 1

    # every other scene tree (the empty regions' and the global region's) still lists NYC's instances: empty them
    filled = {b'R_%03d' % (key[1] * GRID_SIDE + key[0]) for key in per_region}

    for top in (t for t in tree.chunks if t.tree and t.tag in (b'GEIR', b'GEGR') and t.name not in filled):
        for chunk in (c for c in top.tree.chunks if c.tag == b'SCND'):
            chunk.data = scene_tree(bytes(chunk.data), [])
            counts['scene trees emptied'] += 1

    return counts


def write_terrain(tree: d.ChunkTree, d3c: Path) -> Counter:
    """Every DPL region overlapping the Driv3r terrain gets it as its GRTD (d3_terrain), with DPL water at the city's
    water height in Driv3r's water and where it has no ground, but not in a terrain cell a Driv3r road or pavement
    crosses under the water level (its sunken roads are built from models, its sea floor too, so collision cannot tell
    them apart); the other regions lose NYC's."""
    chunks = [c for top in d.read_tree(d3c.read_bytes()).chunks for c in (top.tree.chunks if top.tree else [])]
    sources = [d3_terrain.Driv3rTerrain(bytes(c.data)) for c in chunks if c.tag == b'GRTD']
    sea = d3_terrain.Driv3rWater([bytes(c.data) for c in chunks if c.tag == b'GWRD'], water_height(d3c))
    cell = sources[0].cell
    sunk: set[tuple[int, int]] = set()

    for road in d3_ai.read_roads(d3_ai.aiex_chunk(d3c)):
        reach = max([abs(o) for side in road.lanes for o in side] + [abs(o) for o in road.pavements + road.parking] + [0.0])
        reach += ROAD_MARGIN
        points = road.points()
        for a, b in zip(points, points[1:]):
            for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / ROAD_MARGIN) + 1)):
                x, y, z = a + (b - a) * t
                if y < sea.height - d3_terrain.SUNK:
                    sunk.update((cx, cz) for cx in range(math.floor((x - reach) / cell), math.floor((x + reach) / cell) + 1)
                                for cz in range(math.floor((z - reach) / cell), math.floor((z + reach) / cell) + 1))

    def sunken(x: float, z: float) -> bool:
        return (math.floor(x / cell), math.floor(z / cell)) in sunk

    ground = dpl_ground.read_ground(tree)

    def floors(points: list[tuple[float, float]]) -> float | None:
        """The height of the flat collision top (a jetty, quay or pier) that the centre and at least four of the nine
        sample points of a cell without ground stand on; None otherwise. A cell a pier only half covers gets a floor
        over its whole width: a walker can stand on the water up to 10 m beside it."""
        tops = [[top for _, top in ground.boxes(x, z) if sea.height - FLOOR_BELOW < top < sea.height + d3_terrain.ROOF] for x, z in points]

        for top in tops[len(tops) // 2]:
            found = [t for t in (min(ts, key=lambda v: abs(v - top), default=None) for ts in tops) if t is not None and abs(t - top) <= FLOOR_FLAT]

            if len(found) >= FLOOR_POINTS:
                return float(np.mean(found))

        return None

    lo_x = min(s.min_x for s in sources)
    lo_z = min(s.min_z for s in sources)
    hi_x = max(s.min_x + s.width * s.cell for s in sources)
    hi_z = max(s.min_z + s.height * s.cell for s in sources)
    regions = [top for top in tree.chunks if top.tree and top.tag == b'GEIR']
    template = next(c for top in regions for c in top.tree.chunks if c.tag == b'GRTD')
    counts: Counter = Counter()

    for top in regions:
        column, row = int(top.name[2:]) % GRID_SIDE, int(top.name[2:]) // GRID_SIDE
        x0, z1 = GRID_ORIGIN + column * REGION_SIZE, -GRID_ORIGIN - row * REGION_SIZE
        top.tree.chunks = [c for c in top.tree.chunks if c.tag != b'GRTD']

        if x0 < hi_x and lo_x < x0 + REGION_SIZE and z1 - REGION_SIZE < hi_z and lo_z < z1:
            terrain = d3_terrain.region_terrain(sources, x0, z1 - REGION_SIZE, x0 + REGION_SIZE, z1, sea, sunken, floors)
            at = next((i for i, c in enumerate(top.tree.chunks) if c.tag == b'GEBR'), len(top.tree.chunks))
            top.tree.chunks.insert(at, replace(template, data=terrain.join(), tree=None))
            counts['regions with Driv3r terrain'] += 1
            counts['clamped planes'] += terrain.clamped
            counts['cells floored on collision'] += terrain.floored
        else:
            counts['regions without terrain'] += 1

    return counts


def write_roads(tree: d.ChunkTree, d3c: Path, heads: list[d3_lights.Head],
                phases: list[int]) -> tuple[str, list[dpl_ai.NewRoad], d3_ai.RailLine]:
    """DPL's road network (AIEX of the global region) becomes the Driv3r city's, with its pavements and parking strips
    and the traffic lights of its junctions; NYC's roads go, and Driv3r's tram tracks become DPL's train tracks
    (train_tracks) instead of car roads. Driv3r's own junction curves link the lanes (docs/D3-SOLVED-CASES.md C16),
    clear of the city's collision. Also returns the train line."""
    aiex = next(c for top in tree.chunks if top.tree and top.tag == b'GEGR' for c in top.tree.chunks if c.tag == b'AIEX')
    ai = dpl_ai.read(bytes(aiex.data))
    dpl_ai.clear(ai)
    d3_aiex = d3_ai.aiex_chunk(d3c)
    kept = [road for road in d3_ai.read_roads(d3_aiex) if not road.rail]
    lane_at = {lane: (i, d, k) for i, road in enumerate(kept) for d, ids in enumerate(road.lane_ids) for k, lane in enumerate(ids)}
    links = [dpl_ai.Link(lane_at[c.source], lane_at[c.target], c.controls, c.speed_class) for c in d3_ai.read_curves(d3_aiex)
             if c.source in lane_at and c.target in lane_at]
    roads, links = dpl_ai.split_long([dpl_poc.new_road(road, np.zeros(3)) for road in kept], links)
    ground = dpl_ground.read_ground(tree)
    roads, moved = clear_of_collision(roads, ground)
    roads, lifted = lift_onto_decks(roads, ground)
    moved += lifted
    roads, links, dropped = dpl_ai.drop_dead_ends(roads, links)
    summary = f'{dpl_ai.add_roads(ai, roads, links)}; {dropped} lanes dropped (dead ends); {dict(moved)}'
    lights = d3_lights.assign_lights(ai, roads, 0, heads, phases)
    line, trains = train_tracks(ai, d3_aiex)
    aiex.data = dpl_ai.write(ai)
    return f'{summary}; traffic lights {dict(lights)}; train tracks: {trains}', roads, line


def train_tracks(ai: dpl_ai.AiExport, d3_aiex: bytes) -> tuple[d3_ai.RailLine, str]:
    """Network 2 (DPL's train tracks, emptied with NYC's roads) becomes the city's rail line (d3_ai.rail_line), one road
    per piece with a lane per track to either side. A city without a line of MIN_RAIL_LINE gets one road under
    OUT_OF_REACH instead: NYC's train cars (life actors 0x74) stay in the life spool, and on an empty track the game
    puts them at the origin (0x5B94B1) and moves their model there once the player comes within 300 m (0x47BF1F)."""
    line = d3_ai.rail_line(d3_aiex, d3_ai.read_roads(d3_aiex))

    if line is None or line.length < MIN_RAIL_LINE:
        start = np.array([OUT_OF_REACH[0], TRACK_HEIGHT, OUT_OF_REACH[2]])
        line = d3_ai.RailLine([([start, start + [0.0, 0.0, TRACK_LENGTH]], TRACK_GAUGE)])

    points = np.array(line.points)
    first = np.floor(points[:, [0, 2]].min(axis=0) / dpl_ai.CELL).astype(int) - 1
    last = np.floor(points[:, [0, 2]].max(axis=0) / dpl_ai.CELL).astype(int) + 1
    ai.grids[1] = dpl_ai.Grid(int(first[0]), int(first[1]), int(last[0] - first[0] + 1), int(last[1] - first[1] + 1))
    street_cells = ai.grids[0].width * ai.grids[0].height
    ai.tables['cells'] = np.concatenate([ai.tables['cells'][:street_cells],
                                         np.zeros(ai.grids[1].width * ai.grids[1].height, '<u4')])
    # direction 0 lane k becomes train track k (0x435014); NYC's trains use tracks 0 and 1
    tracks = [dpl_ai.NewRoad([piece[i:i + SEGMENT_POINTS + 1] for i in range(0, len(piece) - 1, SEGMENT_POINTS)],
                             ([-gauge, gauge], [])) for piece, gauge in line.pieces]
    tracks, _ = dpl_ai.split_long(tracks, [])
    return line, f'{line.length:.0f} m in {len(line.pieces)} pieces; {dpl_ai.add_roads(ai, tracks, net=1)}'


def road_line(road: dpl_ai.NewRoad, offset: float) -> list[np.ndarray]:
    """Points at most ROAD_STEP apart along the road, offset metres to the left of its start-to-end direction."""
    points = [p for segment in road.segments for p in segment]
    result = []

    for p, q in zip(points[:-1], points[1:]):
        flat = (q - p)[[0, 2]]
        if not flat.any():
            continue
        left = np.array([flat[1], 0.0, -flat[0]]) / np.linalg.norm(flat) * offset
        n = max(1, int(np.linalg.norm(q - p) // ROAD_STEP))
        result += [p + (q - p) * k / n + left for k in range(n + 1)]

    return result


def lift_onto_decks(roads: list[dpl_ai.NewRoad], ground: dpl_ground.Ground) -> tuple[list[dpl_ai.NewRoad], Counter]:
    """A centre line point whose lane, pavement or parking strip lies inside a deck box (a slab DECK_MIN_THICKNESS or
    more thick holding the point, its top at most DECK_LIFT_MAX above it) rises to the highest such top, so what drives
    or walks there starts on the deck, not in it: DPL's wheel and foot probes start at the lane height and sink to the
    surface below a box that holds their start (Miami's off-ramp at 1068.5, -429.4: lanes 0.14 m under the deck top)."""
    counts: Counter = Counter()
    result = []

    for road in roads:
        offsets = [0.0, *road.lanes[0], *road.lanes[1], *road.pavements, *road.parking]
        segments = []

        for segment in road.segments:
            lifted = []

            for i, p in enumerate(segment):
                a, b = (segment[i], segment[i + 1]) if i + 1 < len(segment) else (segment[i - 1], segment[i])
                flat = (b - a)[[0, 2]]
                left = np.array([flat[1], 0.0, -flat[0]]) / max(np.linalg.norm(flat), 1e-9)
                tops = [hi for o in offsets for lo, hi in ground.boxes(*(p + left * o)[[0, 2]])
                        if lo < p[1] < hi <= p[1] + DECK_LIFT_MAX and hi - lo >= DECK_MIN_THICKNESS]
                counts['road points lifted'] += bool(tops)
                lifted.append(p + [0.0, max(tops) - p[1], 0.0] if tops else p)

            segments.append(lifted)

        result.append(replace(road, segments=segments))

    return result, counts


def clear_of_collision(roads: list[dpl_ai.NewRoad], ground: dpl_ground.Ground) -> tuple[list[dpl_ai.NewRoad], Counter]:
    """Lanes and parking strips whose line meets a collision box in the height of a car move towards the road's centre
    line, CLEAR_STEP at a time up to CLEAR_MAX, until clear: Driv3r keeps one offset per road where its barriers bow in
    (outer highway lanes ran through them for 6-16 m). A parking strip that stays blocked goes (cars spawn on it into a
    lamp post or wall); a lane stays and dpl_road_check reports it."""
    counts: Counter = Counter()

    def cleared(road: dpl_ai.NewRoad, offset: float) -> float | None:
        for step in np.arange(0.0, CLEAR_MAX + 1e-9, CLEAR_STEP):
            moved = offset - math.copysign(step, offset)
            if not any(ground.blocked(*p, dpl_ground.CAR_BAND) for p in road_line(road, moved)):
                return moved
        return None

    result = []
    for road in roads:
        lanes = tuple([o if (c := cleared(road, o)) is None else c for o in road.lanes[d]] for d in (0, 1))
        parking = [c for c in (cleared(road, o) for o in road.parking) if c is not None]
        counts['lanes moved'] += sum(a != b for d in (0, 1) for a, b in zip(lanes[d], road.lanes[d]))
        counts['parking strips moved'] += len(set(parking) - set(road.parking))
        counts['parking strips dropped'] += len(road.parking) - len(parking)
        result.append(replace(road, lanes=lanes, parking=parking))

    return result, counts


def write_lamps(tree: d.ChunkTree, heads: list[d3_lights.Head], phases: list[int], placements: list[Placement],
                slots: dict[int, int], instances: list[d3city.Instance], pages: dict[int, d3city.Page]) -> Counter:
    """The lamps of the Driv3r light heads as DPL's type-5 attractors in the ATRD of the region holding each head's
    instance, lit by the light slot of its phase, and a type-4 street light per lamp lens of every placed model
    (d3_lights). A baked instance names its region's first instance (a building), whose destroyed flag never comes
    on."""
    aiex = next(c for top in tree.chunks if top.tree and top.tag == b'GEGR' for c in top.tree.chunks if c.tag == b'AIEX')
    models = d3_lights.head_models(pages[GLOBAL_PAGE], heads, instances)
    head_slots = {slots[c]: c for c in models if c in slots}
    placed = {d3_lights.position_key((p.x, p.y, p.z)): (head_slots[p.handle & 1023], p.gebi[:2], p.gebi[2])
              for p in placements if p.region is None and p.gebi and p.handle >> 10 == GLOBAL_PAGE
              and p.handle & 1023 in head_slots}
    filled = {p.gebi[:2] for p in placements if p.gebi}

    for i in instances:
        key, region = d3_lights.position_key(i.position), region_key(i.position[0], i.position[2])

        if i.handle in models and key not in placed and region in filled:
            placed[key] = (i.handle, region, 0)

    regions = {struct.unpack_from('<2h', c.data, 8): c for top in tree.chunks if top.tree and top.tag == b'GEIR'
               for c in top.tree.chunks if c.tag == b'ATRD'}
    lamps, counts = d3_lights.lamps(heads, phases, dpl_ai.read(bytes(aiex.data)), placed, models)
    every = {d3_lights.position_key((p.x, p.y, p.z)): (p.gebi[:2], p.gebi[2]) for p in placements
             if p.region is None and p.gebi}
    lights, light_counts = d3_lights.street_lights(pages, instances, every,
                                                   lambda x, z: key if (key := region_key(x, z)) in filled else None)
    return counts + light_counts + d3_lights.write_lamps(regions, lamps + lights)


def write_attractors(tree: d.ChunkTree, roads: list[dpl_ai.NewRoad]) -> Counter:
    """Empties NYC's attractor lists that place things (NYC_LISTS) in every region's ATRD: at NYC's places and heights
    they put standing cars, lights, emitters and litter in Miami's streets and buildings. An empty list is safe: every
    query stops at once, every user checks its hits. List 10 gets a point on a lane of Miami's roads per RECOVERY_CELL:
    the game puts a player who sank 0.6-2.5 m under the ground on the nearest within 400 m (0x4A25AB)."""
    counts: Counter = Counter()
    cells: dict[tuple[int, int, int], np.ndarray] = {}

    for road in (r for r in roads if r.lanes[0] or r.lanes[1]):
        for p in road_line(road, (road.lanes[0] + road.lanes[1])[0]):
            cell = (int(p[0] // RECOVERY_CELL), int(p[2] // RECOVERY_CELL), int(p[1] // RECOVERY_BAND))
            cells.setdefault(cell, p + np.array([0.0, RECOVERY_LIFT, 0.0]))

    points: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    for p in cells.values():
        points[region_key(p[0], p[2])].append(p)

    atrds = {struct.unpack_from('<2h', c.data, 8): c for top in tree.chunks if top.tree and top.tag == b'GEIR'
             for c in top.tree.chunks if c.tag == b'ATRD'}

    for key, c in atrds.items():
        for k, stride in NYC_LISTS.items():
            table = struct.unpack_from('<I', c.data, 0)[0]
            start, end = struct.unpack_from('<2I', c.data, table + 4 * k)
            counts[f'NYC list {k} records removed'] += (end - start) // stride
            items = d3_lights.point_list(points.get(key, []), k) if k == RECOVERY_LIST else b''
            c.data = d3_lights.replace_list(bytes(c.data), k, items)

        counts['recovery points'] += len(points.get(key, []))

    counts['recovery points without an ATRD'] = sum(len(v) for k, v in points.items() if k not in atrds)
    return counts


def write_map(dpl_map: Path, d3_map: Path, out: Path) -> str:
    """The overhead map (Overlays\\*.map: a model pack of the map page): DPL's own map with its city map (container 0)
    replaced by the Driv3r city's land, water and roads, in the same map units, and NYC's road tiles emptied. DPL's
    other containers (icons) stay."""
    data = d3_map.read_bytes()
    inner = d.entries(data, 0)[0]
    entry = d.entries(data, inner.offset)[0]
    page = d3city.Page(data, entry.offset, entry.size)
    layers = [d3city.bake_instance(page, c, AT_ORIGIN) for c in D3_MAP_LAYERS]
    baked = merge(layers)
    tree = d.read_tree(dpl_map.read_bytes(), 0, 2)
    chunk = next(c for c in tree.chunks[0].tree.chunks if c.tag == b'MDXN')
    raw = d.split_pack(chunk.data, 0, len(chunk.data))
    first_material = len(raw.pcmp.materials)
    # the map's meshes name their own page or 0 for their material table: both are this page
    dpl_poc.write_into(raw, MAP_CONTAINER, baked, {0: page, page.id: page})

    # materials were added in drawing order, layer by layer; overlapping materials of one priority swap order from
    # frame to frame (flicker)
    priorities = [values[min(k, len(values) - 1)] for layer, values in zip(layers, MAP_PRIORITIES)
                  for k in range(len(dict.fromkeys(layer.materials.tolist())))]

    for m, priority in zip(range(first_material, len(raw.pcmp.materials)), priorities, strict=True):
        substance = raw.pcmp.substances[raw.pcmp.pointers[raw.pcmp.material_pointers[m]]]
        word = struct.unpack_from('<I', substance, 4)[0] & ~0xFF
        struct.pack_into('<I', substance, 4, word | MAP_OVERLAY | priority)

    for tile in MAP_TILES:
        raw.lod_groups[tile] = [None] * len(raw.lod_groups[tile])

        for k in range(len(raw.lod_groups[tile])):
            struct.pack_into('<I', raw.containers[tile], 0xA8 + k * 24, 0)

    chunk.data = d.join_pack(raw)

    with out.open('wb') as f:
        for piece in d.tree_pieces(tree):
            f.write(piece)

    return (f'{len(baked.vertices)} vertices, {len(np.unique(baked.materials))} materials into container {MAP_CONTAINER}, '
            f'{len(MAP_TILES)} NYC road tiles emptied')


def start_position(d3c: Path, city: str) -> tuple[float, float, float]:
    """The point of the city's car roads nearest Driv3r's own start there (CITY_STARTS, else the middle of the roads),
    lifted a little."""
    points = np.array([s.start for r in d3_ai.read_roads(d3_ai.aiex_chunk(d3c)) if any(r.lanes) and not r.rail for s in r.segments])
    centre = np.array(CITY_STARTS.get(city.lower(), (points[:, 0].mean(), points[:, 2].mean())))
    x, y, z = points[np.argmin(np.hypot(*(points[:, 0::2] - centre).T))]
    return round(float(x), 1), round(float(y) + START_LIFT, 1), round(float(z), 1)


def set_start(game: Path, city: str, position: tuple[float, float, float]) -> None:
    """Sets StartPosition of the city's LEVEL in levels.txt, keeping its indent and line end."""
    def edit(line: bytes) -> bytes:
        indent = line[:len(line) - len(line.lstrip())]
        return indent + b'StartPosition\t\t' + ', '.join(f'{v:g}' for v in position).encode() + (b'\r' if line.endswith(b'\r') else b'')

    d3_vehicle_swap.edit_level(game, b'StartPosition', edit, city)
    print(f'{city} LEVEL starts at {position}')


def child(data: bytes | bytearray, parent: int, tag: str) -> d.Entry:
    return next(e for e in d.entries(data, parent) if e.tag == tag)


def edit_base_actors(data: bytearray, player_skin: int) -> Counter:
    """Moves the base life event's NYC mission starts and pickups out of reach (NYC's places lie inside the Driv3r
    city, where they would load NYC's missions with their shops and markers, and pickups), makes one NYC district
    (CITY_DISTRICT) the whole city and moves the other district boxes away, and gives its player character the skin
    player_skin. Returns the count per kind."""
    leld = child(data, child(data, child(data, 0, 'SSIC').offset, 'EM__').offset, 'LELD')
    lesc = child(data, leld.offset, 'LESC')
    offsets = struct.unpack_from(f'<{struct.unpack_from("<I", data, lesc.offset)[0]}I', data, lesc.offset + 4)
    names = [bytes(data[lesc.offset + o:data.index(0, lesc.offset + o)]) for o in offsets]
    skin_name = names.index(SKIN_PROPERTY)
    lead, lepr = d.entries(data, child(data, leld.offset, 'LEAC').offset)
    assert (lead.tag, lepr.tag) == ('LEAD', 'LEPR'), (lead.tag, lepr.tag)
    actors = struct.unpack_from('<I', data, lead.offset)[0]
    at = lepr.offset + 4
    moved = Counter()

    for i in range(actors):
        kind = data[lead.offset + 4 + i * LIFE_ACTOR_SIZE]
        properties = struct.unpack_from('<I', data, at)[0]
        at += 4

        for _ in range(properties):
            prop, name, size = LIFE_PROPERTY.unpack_from(data, at)

            if prop == MATRIX_PROPERTY and kind in NYC_GAME_ACTORS:
                struct.pack_into('<3f', data, at + LIFE_PROPERTY.size + 48, *OUT_OF_REACH)
                moved[NYC_GAME_ACTORS[kind]] += 1
            elif kind == PLAYER_ACTOR and name == skin_name and size == 4:
                struct.pack_into('<i', data, at + LIFE_PROPERTY.size, player_skin)
                moved['player skin'] += 1
            elif kind == VOLUME_ACTOR and i in DISTRICT_BOXES and prop == MATRIX_PROPERTY:
                where = (0.0, 0.0, 0.0) if i in CITY_DISTRICT else DISTRICT_AWAY
                struct.pack_into('<16f', data, at + LIFE_PROPERTY.size, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, *where, 1)
                moved['district boxes over the city' if i in CITY_DISTRICT else 'district boxes away'] += 1
            elif kind == VOLUME_ACTOR and i in CITY_DISTRICT and prop == PRIMITIVE_PROPERTY:
                if struct.unpack_from('<I', data, at + LIFE_PROPERTY.size)[0] != BOX_SHAPE:
                    raise SystemExit(f'{NYC_LIFE_SPOOL}: district actor {i} is not a box')
                struct.pack_into('<3f', data, at + LIFE_PROPERTY.size + 4, *CITY_DISTRICT_SIZE)

            at += LIFE_PROPERTY.size + (size + 3 & ~3)

    if at > lepr.offset + lepr.size:
        raise SystemExit(f'{NYC_LIFE_SPOOL}: base life event properties run past their chunk')
    if moved['player skin'] != 1:
        raise SystemExit(f'{NYC_LIFE_SPOOL}: {moved["player skin"]} player skins in the base life event, expected 1')
    if moved['district boxes over the city'] + moved['district boxes away'] != len(DISTRICT_BOXES):
        raise SystemExit(f'{NYC_LIFE_SPOOL}: {moved["district boxes over the city"] + moved["district boxes away"]} '
                         f'district boxes in the base life event, expected {len(DISTRICT_BOXES)}')

    return moved


def life_properties(data: bytes | bytearray, container: d.Entry, record_size: int) -> Iterator[tuple[int, int, int, int, int]]:
    """(item, kind, property kind, name index, value offset) of every property of a base life event list: LEAC (actors,
    LEAD records of 16 bytes) or LENC (nodes, LEND records of 12), kind in a record's byte 0, then the LEPR lists."""
    records, lepr = d.entries(data, container.offset)
    at = lepr.offset + 4

    for i in range(struct.unpack_from('<I', data, records.offset)[0]):
        kind = data[records.offset + 4 + i * record_size]
        properties = struct.unpack_from('<I', data, at)[0]
        at += 4

        for _ in range(properties):
            prop, name, size = LIFE_PROPERTY.unpack_from(data, at)
            yield i, kind, prop, name, at + LIFE_PROPERTY.size
            at += LIFE_PROPERTY.size + (size + 3 & ~3)


def edit_trains(data: bytearray, line: d3_ai.RailLine) -> Counter:
    """Puts NYC's train cars (actors TRAIN_ACTOR, five per track, each shuttling between its own shares near the two
    track ends on TRAIN_MOVE nodes) on the line: every car of a track gets the same shares, TRAIN_STOP from either end, so
    the five drive as one train (they all draw the one train model), and its matrix there on its track."""
    leld = child(data, child(data, child(data, 0, 'SSIC').offset, 'EM__').offset, 'LELD')
    lesc = child(data, leld.offset, 'LESC')
    offsets = struct.unpack_from(f'<{struct.unpack_from("<I", data, lesc.offset)[0]}I', data, lesc.offset + 4)
    names = [bytes(data[lesc.offset + o:data.index(0, lesc.offset + o)]) for o in offsets]
    shares = {names.index(name) for name in POSITION_PROPERTIES}
    track_name = names.index(TRACK_PROPERTY)
    stop = TRAIN_STOP / line.length
    counts = Counter()
    actors = list(life_properties(data, child(data, leld.offset, 'LEAC'), LIFE_ACTOR_SIZE))
    tracks = {i: struct.unpack_from('<i', data, at)[0] for i, kind, _, name, at in actors if kind == TRAIN_ACTOR and name == track_name}

    for i, kind, prop, name, at in actors + list(life_properties(data, child(data, leld.offset, 'LENC'), LIFE_NODE_SIZE)):
        if kind not in (TRAIN_ACTOR, TRAIN_MOVE) or (kind == TRAIN_ACTOR and i not in tracks):
            continue
        if name in shares:
            share = stop if struct.unpack_from('<f', data, at)[0] < 0.5 else 1 - stop
            struct.pack_into('<f', data, at, share)
            counts['train shares' if kind == TRAIN_ACTOR else 'train move targets'] += 1

    # the matrices last: they follow the cars' new shares
    for i, kind, prop, name, at in actors:
        if prop == MATRIX_PROPERTY and i in tracks:
            share = next(struct.unpack_from('<f', data, a)[0] for j, _, _, n, a in actors if j == i and n in shares)
            position, forward = track_place(line, share, tracks[i])
            right = np.array([forward[2], 0.0, -forward[0]])
            struct.pack_into('<16f', data, at, *right, 0, 0, 1, 0, 0, *forward, 0, *position, 1)
            counts[f'train cars on track {tracks[i]}'] += 1

    return counts


def track_place(line: d3_ai.RailLine, share: float, track: int) -> tuple[np.ndarray, np.ndarray]:
    """(position, unit forward along the line) at that share of the line's length on train track 0 or 1: the lane
    of the piece there, to the right (0) or left (1) of the line (DPL counts lane offsets to the left)."""
    left = share * line.length

    for points, gauge in line.pieces:
        for p, q in zip(points[:-1], points[1:]):
            step = float(np.linalg.norm(q - p))
            if left <= step or (points is line.pieces[-1][0] and q is points[-1]):
                forward = (q - p) / max(step, 1e-9)
                side = np.array([forward[2], 0.0, -forward[0]]) / max(float(np.hypot(forward[0], forward[2])), 1e-9)
                return p + forward * min(left, step) + side * (gauge if track else -gauge), forward
            left -= step

    raise ValueError('empty rail line')


def write_life_spool(game: Path, city: str, position: tuple[float, float, float], line: d3_ai.RailLine) -> None:
    """A copy of NYC Now's life event spool with the base life event's player start moved to the city's start, the
    player Tanner (d3_tanner), its NYC mission starts and pickups out of reach and its train cars on the city's train
    line (edit_trains), loaded by the city's LEVEL (LifeSpoolFile); DPL's own spool is only read."""
    data = (game / 'LifeEvents' / NYC_LIFE_SPOOL).read_bytes()

    if data.count(NYC_PLAYER_START) != NYC_PLAYER_START_COUNT:
        raise SystemExit(f'{NYC_LIFE_SPOOL}: the player start is there {data.count(NYC_PLAYER_START)} times, '
                         f'expected {NYC_PLAYER_START_COUNT}')

    data = bytearray(data.replace(NYC_PLAYER_START, struct.pack('<3f', *position)))
    print(f'life events: {dict(edit_base_actors(data, d3_tanner.TANNER_ID))}; {dict(edit_trains(data, line))}')
    name = f'{city.lower()}_missions.sp'
    (game / 'LifeEvents' / name).write_bytes(data)
    d3_vehicle_swap.set_level(game, name, b'LifeSpoolFile', 'LifeEvents', city)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('game', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--driv3r', type=Path, help='Driv3r game dir: adds the city (parts 8b-8c: its models)')
    ap.add_argument('--city', default='Miami')
    ap.add_argument('--map', type=Path, help='with --driv3r: also write the overhead map here (e.g. Overlays/miami.map)')
    ap.add_argument('--level', action='store_true', help="with --driv3r: set the city LEVEL's StartPosition in levels.txt")
    args = ap.parse_args()
    d.require_copy(args.out)
    template = args.game / 'Cities' / TEMPLATE
    new = args.out.with_suffix('.d4c.new')

    with template.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        tree = d.read_tree(data)
        jnr_template = next(replace(c, data=bytes(c.data)) for top in tree.chunks if top.tree and top.tag == b'GEIR'
                            for c in top.tree.chunks if c.tag == b'JNR!')       # a copy: no view outlives the map
        world = {i.handle & 1023 for top in tree.chunks if top.tree and top.tag in (b'GEIR', b'GEGR') for c in top.tree.chunks
                 if c.tag == b'GEBI' for i in d.region_instances(bytes(c.data)) if i.handle >> 10 == GLOBAL_PAGE}
        # NYC places some game-code models too (barriers, crates): they keep NYC's model for the missions that spawn them
        world -= set(glut_models(tree).values())
        changed = empty_city(tree, world)
        print(f'empty city: emptied or dropped {dict(sorted(changed.items()))}')

        if args.driv3r:
            city = args.driv3r / 'Cities'
            pages = d3city.read_pages(city / f'{args.city}_DAY_PC' / 'SuperRegions.pcs')
            pages.update(d3city.read_interior_pages(city / f'{args.city}_DAY_PC' / 'Interiors.pcs'))     # C34
            instances = d3city.read_instances(city / f'{args.city}_DAY_PC.d3c', args.city.lower() in QUATERNION_AS_STORED,
                                              interiors=True)
            collision = CollisionBuilder(tree, d3city.read_collision(city / f'{args.city}_DAY_PC.d3c', interiors=True))
            water, bed = sea_containers(tree, world)
            moving, prop_slots = d3_props.templates(tree)
            masses = d3_props.prop_models(city / f'{args.city}_DAY_PC.d3c')
            props = d3_props.assign(pages[GLOBAL_PAGE], masses, {s: n for s, n in moving.items() if s in world - {water, bed}}, tree)
            print(f'props: {stand_props(instances, props)} slightly tilted props stand upright')
            radii, slots, empty = write_global_pack(tree, pages, instances, world - {water, bed}, props, prop_slots)
            print(f'train: {write_train_model(tree, args.driv3r)}')
            write_sea_tiles(tree, water, bed)
            excess = excess_props(instances, radii, collision, props)
            hidden = GLOBAL_PAGE << 10 | empty
            placements = global_placements(instances, radii, slots, hidden, excess, collision, props)
            shapes = Counter((p.handle & 1023, code & 2047) for p in placements if p.prop for code, *_ in p.objects[:1])
            print(f'props: {d3_props.write_records(tree, pages[GLOBAL_PAGE], props, masses, shapes)} PROP records from Driv3r props')
            placements += write_super_regions(tree, pages, instances, excess, collision)
            placements += border_copies(placements, hidden)
            print(f'instances: {dict(write_regions(tree, placements, jnr_template))}')
            height = water_height(city / f'{args.city}_DAY_PC.d3c')
            print(f'sea: NYC tiles {water} (water) and {bed} (bed) under {write_sea(tree, water, bed, height)} regions, '
                  f'water at y {height:g}')
            print(f'collision: {dict(collision.write())}')
            print(f'terrain: {dict(write_terrain(tree, city / f"{args.city}_DAY_PC.d3c"))}')
            heads, phases = d3_lights.read_heads(d3_ai.aiex_chunk(city / f'{args.city}_DAY_PC.d3c'))
            summary, roads, line = write_roads(tree, city / f'{args.city}_DAY_PC.d3c', heads, phases)
            print(f'roads: {summary}')
            print(f'lamps: {dict(write_lamps(tree, heads, phases, placements, slots, instances, pages))}')
            print(f'attractors: {dict(write_attractors(tree, roads))}')

            if args.map:
                if args.map.name.lower() in ORIGINAL_MAPS:
                    raise SystemExit(f'refusing to overwrite DPL\'s own {args.map.name}')

                print(f'map: {write_map(args.game / "Overlays" / ORIGINAL_MAPS[0], args.driv3r / "Overlays" / f"{args.city.lower()}.map", args.map)}')

            if args.level:
                start = start_position(city / f'{args.city}_DAY_PC.d3c', args.city)
                set_start(args.game, args.city, start)
                write_life_spool(args.game, args.city, start, line)

        with new.open('wb') as out:
            for piece in d.tree_pieces(tree):
                out.write(piece)

        del tree        # release the views into the template before the map closes

    os.replace(new, args.out)
    print(f'file {os.path.getsize(args.out):#x} bytes')


if __name__ == '__main__':
    main()
