"""Converts every Driv3r car of a city into Driver Parallel Lines: the vehicles of the Driv3r DLC.


plan  lists the city's cars (traffic <city>.vvs, specials <city>.vvv, the artic truck and its trailer, bikes; duplicates
      across those files once) and gives each the DPL vehicle type it replaces (the donor): same wheel layout (hub nodes),
      hubs nearest, each donor once per city; racer, punk and turbo types only when no plain type is left. The plan is
      written to <DPL dir>/Mods/Cars/D3/cars.json; entries already there (a donor changed by hand, "pinned": true) are
      kept, so edit the file and run plan again to see the rest re-fitted.
build converts the city's planned cars with d3_vehicle_swap.convert into every variant of their donor type (the donor
      read from the original nyc_vehicles.sp), writes each car's converted VEHC to Mods/Cars/D3/<city>/<name>.vehc and
      the city's vehicle spool to Vehicles/<spool> (from cars.json; the level of that city loads it via levels.txt
      SpooledVehicles). With the game running the spool waits as <spool>.new.
      --add keeps every DPL vehicle and adds each car as a new vehicle instead: a new variant (5-15) of its donor type
      (id = type | variant << 12, recorded as "added_id" in cars.json), a new VEHC before BINF and a new SSLP record.
      With --city all every city's cars go into one spool (the first city's), each with its own id, so any level that
      loads it can spawn all of them. Spawn them from the Extended window ("Driv3r cars", reads
      Mods/Cars/D3/<city>/added.txt).
traffic reads the city's moving-traffic list (model, weight) out of Driv3r.exe (v1.20) and writes it with the added
      ids to Mods/Cars/D3/<city>/traffic.txt, which the Extended hook hands DPL as the city's traffic list. Driv3r has
      one such list per city, no list per district (docs/D3-PORT-KEYPOINTS.md, "Traffic models"). It also writes
      Mods/Cars/D3/<city>/police.txt: the added id of the car whose model is the city's police car in the exe.

Boats: the hierarchies of <city>_boat.vvv without wheel hubs whose model has a BO3 block with boat keys (167-180) are
planned as cars on BOAT_DONORS[city] (the donor's wheels stay, invisible: on land the boat rolls on them). build writes
the boat's keys (BOAT_TABLE_KEYS, Driv3r's defaults where its block sets none) into its VO3D; the Extended hook
(BoatFrame) knows a boat by its "-boat-" name in added.txt (DPL's default table sets every key, so none can mark one)
and sails it with Driv3r's boat physics on DPL's water (docs/PLAN-BOATS.md).
Left out: other vehicles without wheel hubs (the city's helicopter-like props).
"""
import argparse
import dataclasses
import hashlib
import json
import mmap
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d3_vehicle_swap as s
import dplchunk as d
import dpl_vehicles as dv
import vehchunk as v

CITIES = ('miami',)
# (file suffix after the city name, name part); bikes and the artic truck repeat cars of the other files
SOURCES = (('.vvs', 'traffic'), ('.vvv', 'special'), ('_artic_truck.vvv', 'artic'), ('_bike.vvv', 'bike'))
LIBRARY = Path('Mods') / 'Cars' / 'D3'
MANIFEST = 'cars.json'
# the level that shows a city loads this spool; Miami's level (levels.txt "NYC Now") already loads the first one
SPOOLS = {'miami': 'nyc_d3_vehicles.sp'}
# donors checked in game by the user (2026-10-19): taxi -> Colonna, police car -> Citation, Tanner's car -> Teramo,
# six-wheel truck -> Challenger
PINNED = {('miami', 'miami.vvs', 1): 0x9, ('miami', 'miami.vvv', 1): 0x2D, ('miami', 'miami.vvs', 6): 0x20,
          ('miami', 'miami.vvs', 5): 0x5}
SPECIAL_DONOR = ('Racer', 'Punk', 'Turbo', 'Specia', 'SpyderR')
SPECIAL_COST = 1.0      # metres added to a racer/punk/turbo donor, so plain types are used first
VARIANT_SHIFT = 12
TYPE_MASK = 0xFFF
FIRST_ADDED_VARIANT = 5     # DPL's variants 1-4 are garage upgrades; the id keeps 4 bits for the variant
LAST_VARIANT = 14           # 15 is the tail guard's (add_tail_guard)
PAGE_BASE = 0x2001          # model pack page = PAGE_BASE + PAGE_STRIDE * type + variant (0x58035f)
PAGE_STRIDE = 124
PAGE_HANDLE_SHIFT = 10      # hierarchy handle = (page & 63) << 10 | container
HANDLE_CONTAINER = 0x3FF
NODE_HANDLE = 18            # byte offset of the handle in a hierarchy node
UPVH_ID = 0x14              # UPVH holds the vehicle id here, VO3D at 0
MESH_WORD = 48              # MDXN mesh record: material word = page << 16 | material
SSLP_HEADER = 0x18          # u16 0, u16 count, ...; then (u16 VEHC ordinal, u16 0, u16 vehicle id) per vehicle
SSLP_COUNT = 2
SSLP_RECORD = 6
TAIL_GUARD_VARIANT = 15
MAX_VEHC = 0x2A8880         # DPL's largest VEHC; its streaming slots are sized from the biggest one (SSLP header)
# Driv3r.exe v1.20: InitialisationData's traffic list (vtable +0x84, 0x4648B0) per city, (u32 model, f32 weight) pairs
TRAFFIC_LISTS = {'miami': 0x71B780}
TRAFFIC_END = 0x67          # Driv3r's "no vehicle" id ends the list
TRAFFIC_MAX = 32
TRAFFIC_FILE = 'traffic.txt'
WEIGHT_SCALE = 30           # Driv3r weight 1.0 = u8 weight 30: DPL's spawn list (0x471C18) counts weights under 3 as 0
# Driv3r.exe v1.20: the felony system (constructor 0x4313E0) takes its police car model from u32[city] here
POLICE_MODELS = 0x71B584
BO3_DIR = Path('Configs') / 'Vehicles' / 'BigVO3'    # <city>.BO3: handling per Driv3r model, its paint colours too
POLICE_FILE = 'police.txt'
# siren light halo (size, falloff) of DPL's own police car; a non-police donor's values draw one big blob
POLICE_HALO_DONOR = 0xB
POLICE_HALO_KEYS = range(282, 291)
BOAT_SUFFIX = '_boat.vvv'
BOAT_KEYS = range(167, 181)     # only Driv3r's boat blocks set these (Driv3r 0x522300: buoyancy, thrust, rudder, damping)
# what Driv3r's boat code reads (driv3r 0x522200, 0x522300, 0x5243F0): steer smoothing, planing speeds, hull size, the
# boat keys and the drag keys 181-183; DPL's VO3Ds set none of them
BOAT_TABLE_KEYS = (*range(62, 70), *range(167, 184))
# one unused 4-wheel DPL type per city (11 free variants each, 4 boats per city); plain vans and a limousine, no sirens.
# The donor needs door points (0x3C/0x3D): the SwiftVan (0x59) has none, and the walk-in to a boat on it never ended
BOAT_DONORS = {'miami': 0x2B}
DRIV3R_DEFAULTS = 0x7A3570      # Driv3r.exe v1.20: the default handling table every BO3 block is laid over (FUN_005142a0)
DRIV3R_KEYS = 190


@dataclasses.dataclass
class Car:
    city: str
    file: str
    car: int
    name: str
    hubs: dict[str, list[float]]
    donor: int | None = None
    donor_name: str = ''
    cost: float = 0.0
    pinned: bool = False


def hub_places(data: bytes, offset: int) -> dict[str, list[float]]:
    """{'27': [x, y, z], ...}: s.hub_places as JSON-ready values (cars.json keeps no hubs, plan compares them)."""
    return {str(part): place.round(4).tolist() for part, place in s.hub_places(data, offset).items()}


def city_cars(game: Path, city: str) -> list[Car]:
    """The city's vehicles with wheel hubs, each model once (same drawn container boxes = same model)."""
    seen: set[str] = set()
    found: list[Car] = []

    for suffix, kind in SOURCES:
        name = city + suffix

        if not (game / 'Vehicles' / name).is_file():
            continue

        data, cars = s.driv3r_cars(game, city, name)

        for i, (_, pack, upvh) in enumerate(cars):
            hubs = hub_places(data, upvh.offset)
            count = struct.unpack_from('<I', data, pack.offset + 12)[0]
            boxes = s.container_boxes(data, pack.offset + count, len(s.driv3r_page(data, pack).containers), 0x188, 0x28)
            key = hashlib.md5(b''.join(boxes[c].round(3).tobytes() for c in sorted(s.drawn_containers(data, upvh))
                                       if c < len(boxes))).hexdigest()

            if hubs and key not in seen:
                seen.add(key)
                found.append(Car(city, name, i, f'{city}-{kind}-{i:02d}', hubs))

    return found


def city_boats(game: Path, city: str) -> list[Car]:
    """The boats of <city>_boat.vvv: hierarchies without wheel hubs whose model's BO3 block sets boat keys, each pinned
    to the city's boat donor."""
    bo3 = v.bo3_blocks((game / BO3_DIR / f'{city}.BO3').read_bytes())
    name = city + BOAT_SUFFIX
    data, cars = s.driv3r_cars(game, city, name)
    found = []

    for i, (_, _, upvh) in enumerate(cars):
        model = struct.unpack_from('<I', data, upvh.offset + s.UPVH_MODEL)[0]

        if not s.hub_places(data, upvh.offset) and any(r.key in BOAT_KEYS for r in bo3.get(model, [])):
            found.append(Car(city, name, i, f'{city}-boat-{i:02d}', {}, BOAT_DONORS[city], pinned=True))

    return found


def boat_table(exe: bytes, bo3: dict[int, list[v.HandlingRecord]], data: bytes, upvh: d.Entry) -> dict[int, int]:
    """{key: raw u32} of the boat's BOAT_TABLE_KEYS (its BO3 block over Driv3r's defaults); empty for a model without
    boat keys."""
    model = struct.unpack_from('<I', data, upvh.offset + s.UPVH_MODEL)[0]
    block = {r.key: r.raw for r in bo3.get(model, [])}

    if not block.keys() & set(BOAT_KEYS):
        return {}

    defaults = struct.unpack_from(f'<{DRIV3R_KEYS}I', exe, exe_offset(exe, DRIV3R_DEFAULTS))
    return {k: block.get(k, defaults[k]) for k in BOAT_TABLE_KEYS}


def donor_types(dpl: Path) -> dict[int, tuple[str, dict[str, list[float]]]]:
    """{base vehicle id: (type name, hubs)} of DPL's original vehicles (variant 0)."""
    found = {}

    with (dpl / 'Vehicles' / s.ORIGINAL).open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        names, vehicles = dv.vehicles(mm)

        for parts in vehicles:
            vid = struct.unpack_from('<I', mm, parts['VO3D'].offset)[0]

            if vid >> VARIANT_SHIFT == 0:
                upvh = bytes(mm[parts['UPVH'].offset:parts['UPVH'].offset + parts['UPVH'].size])
                found[vid] = (names[vid] if vid < len(names) else f'type {vid:#x}', hub_places(upvh, 0))

        del names, vehicles

    return found


def hub_cost(car: dict[str, list[float]], donor: dict[str, list[float]]) -> float | None:
    """Summed distance of the hubs in metres; None for another wheel layout."""
    if car.keys() != donor.keys():
        return None

    return float(sum(np.linalg.norm(np.subtract(car[k], donor[k])) for k in car))


def fit(cars: list[Car], donors: dict[int, tuple[str, dict[str, list[float]]]]) -> None:
    """Gives every car without a donor the free donor of its wheel layout with the nearest hubs, cheapest pairs first."""
    taken = {c.donor for c in cars if c.donor is not None}
    pairs = []

    for car in cars:
        if car.donor is not None:
            continue

        for vid, (name, hubs) in donors.items():
            cost = hub_cost(car.hubs, hubs)

            if cost is not None:
                pairs.append((cost + SPECIAL_COST * any(t in name for t in SPECIAL_DONOR), cost, car.name, vid))

    by_name = {c.name: c for c in cars}

    for _, cost, name, vid in sorted(pairs):
        car = by_name[name]

        if car.donor is None and vid not in taken:
            car.donor, car.donor_name, car.cost = vid, donors[vid][0], round(cost, 3)
            taken.add(vid)


def load_manifest(path: Path) -> dict[str, dict]:
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))

    return {'cities': {}}


def plan(dpl: Path, game: Path, cities: list[str]) -> None:
    path = dpl / LIBRARY / MANIFEST
    manifest = load_manifest(path)
    donors = donor_types(dpl)

    for city in cities:
        old = {(c['file'], c['car']): c for c in manifest['cities'].get(city, {}).get('cars', [])}
        cars = city_cars(game, city) + city_boats(game, city)

        for car in cars:
            kept = old.get((car.file, car.car))

            # a boat's donor comes from BOAT_DONORS, so a changed one there replaces the plan's
            if kept and kept.get('pinned') and not car.pinned:
                car.donor, car.pinned = kept['donor'], True
            elif (city, car.file, car.car) in PINNED:
                car.donor, car.pinned = PINNED[city, car.file, car.car], True

            if car.donor is not None:
                car.donor_name = donors.get(car.donor, ('?', {}))[0]
                car.cost = round(hub_cost(car.hubs, donors.get(car.donor, ('', {}))[1]) or 0.0, 3)

        fit(cars, donors)
        manifest['cities'][city] = {'spool': manifest['cities'].get(city, {}).get('spool', SPOOLS[city]),
                                    'cars': [{k: value for k, value in dataclasses.asdict(c).items() if k not in ('city', 'hubs')}
                                             for c in cars]}

        for c in cars:
            donor = f'{c.donor_name} ({c.donor:#x}), hubs {c.cost:.2f} m off' if c.donor is not None else 'no free donor'
            print(f'{c.name:22s} {len(c.hubs)} wheels -> {donor}{" (pinned)" if c.pinned else ""}')

        print(f'{city}: {len(cars)} cars, {sum(c.donor is None for c in cars)} without a donor')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    print(f'plan written to {path}')


def vehicle_id(vehicle: d.Chunk) -> int:
    return struct.unpack_from('<I', next(c for c in vehicle.tree.chunks if c.tag == b'VO3D').data, 0)[0]


def renumbered(vehicle: d.Chunk, vid: int) -> d.Chunk:
    """A copy of the VEHC as vehicle `vid`: the id in VO3D and UPVH, its model page in the MDXN header, in its own mesh
    materials and in the hierarchy handles. The sound bank names the type, as on DPL's own variants, and stays."""
    parts = {c.tag: dataclasses.replace(c) for c in vehicle.tree.chunks}
    page = PAGE_BASE + PAGE_STRIDE * (vid & TYPE_MASK) + (vid >> VARIANT_SHIFT)
    vo3d = bytearray(parts[b'VO3D'].data)
    struct.pack_into('<I', vo3d, 0, vid)
    upvh = bytearray(parts[b'UPVH'].data)
    struct.pack_into('<I', upvh, UPVH_ID, vid)

    for i in range(struct.unpack_from('<H', upvh, v.UPVH_NODE_COUNT)[0] - 1):
        at = v.UPVH_NODES + i * v.UPVH_NODE_SIZE + NODE_HANDLE
        handle = struct.unpack_from('<H', upvh, at)[0]

        if handle != v.NO_HANDLE:
            struct.pack_into('<H', upvh, at, (page & 63) << PAGE_HANDLE_SHIFT | handle & HANDLE_CONTAINER)

    raw = d.split_pack(parts[b'MDXN'].data, 0, len(parts[b'MDXN'].data))
    old_page = struct.unpack_from('<I', raw.header, 4)[0]
    header = bytearray(raw.header)
    struct.pack_into('<I', header, 4, page)
    raw.header = bytes(header)

    for mesh in raw.meshes:
        word = struct.unpack_from('<I', mesh, MESH_WORD)[0]

        if word >> 16 == old_page:
            struct.pack_into('<I', mesh, MESH_WORD, page << 16 | word & 0xFFFF)

    parts[b'VO3D'].data, parts[b'UPVH'].data, parts[b'MDXN'].data = bytes(vo3d), bytes(upvh), d.join_pack(raw)
    return dataclasses.replace(vehicle, tree=dataclasses.replace(vehicle.tree, chunks=list(parts.values())), gap_extra=0)


def add_lookups(ssic: d.Chunk, added: list[tuple[int, int]]) -> None:
    """Appends (VEHC ordinal, vehicle id) records to the SSLP lookup in the SSIC chunk; the bytes after the records stay."""
    sslp = next(c for c in ssic.tree.chunks if c.tag == b'SSLP')
    data = bytearray(sslp.data)
    count = struct.unpack_from('<H', data, SSLP_COUNT)[0]
    end = SSLP_HEADER + count * SSLP_RECORD
    struct.pack_into('<H', data, SSLP_COUNT, count + len(added))
    sslp.data = bytes(data[:end] + b''.join(struct.pack('<3H', ordinal, 0, vid) for ordinal, vid in added) + data[end:])


def tail_guard_id(tree: d.ChunkTree) -> int:
    """The id of the tail guard: the spool's first vehicle type at variant 15."""
    return vehicle_id(next(c for c in tree.chunks if c.tag == b'VEHC' and c.tree)) & TYPE_MASK | TAIL_GUARD_VARIANT << VARIANT_SHIFT


def add_tail_guard(tree: d.ChunkTree, ssic: d.Chunk) -> None:
    """Puts a copy of the first vehicle last, as a placeholder nothing spawns: the game never loads the spool's last
    VEHC (DPL's own 0x1033 neither), so every real car must come before it."""
    first = next(c for c in tree.chunks if c.tag == b'VEHC' and c.tree)
    at = next(i for i, c in enumerate(tree.chunks) if c.tag == b'BINF')
    tree.chunks.insert(at, renumbered(first, tail_guard_id(tree)))
    add_lookups(ssic, [(at - 1, tail_guard_id(tree))])


def free_variants(vehicles: list[d.Chunk]) -> dict[int, list[int]]:
    """{type: variants still free for added cars}."""
    used: dict[int, set[int]] = {}

    for vehicle in vehicles:
        vid = vehicle_id(vehicle)
        used.setdefault(vid & TYPE_MASK, set()).add(vid >> VARIANT_SHIFT)

    return {t: [k for k in range(FIRST_ADDED_VARIANT, LAST_VARIANT + 1) if k not in u] for t, u in used.items()}


def build(dpl: Path, game: Path, cities: list[str], spool: bool, add: bool = False) -> None:
    """Builds the cities' planned cars into one spool: one city (replace), or with `add` any number of cities, so every
    city's cars get their own ids and can be spawned in any level that loads it (the first city's spool)."""
    manifest = load_manifest(dpl / LIBRARY / MANIFEST)

    for city in cities:
        if city not in manifest['cities']:
            raise SystemExit(f'{city} has no plan yet: run "plan --city {city}" first')

    planned = {city: [c for c in manifest['cities'][city]['cars'] if c['donor'] is not None] for city in cities}

    for city, cars_of_city in planned.items():
        if not cars_of_city:
            raise SystemExit(f'{city}: no car has a donor in the plan')

    exe = (game / 'Driv3r.exe').read_bytes()       # Driv3r's default handling table, for the boats
    spool_name = manifest['cities'][cities[0]]['spool']
    target = dpl / 'Vehicles' / spool_name
    new = target.with_suffix('.sp.new')

    with (dpl / 'Vehicles' / s.ORIGINAL).open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as original:
        tree = d.read_tree(original, 0, 2)
        ssic = next(c for c in tree.chunks if c.tag == b'SSIC')
        ssic_pack = bytes(next(c for c in ssic.tree.chunks if c.tag == b'MDXN').data)
        vehicles = [c for c in tree.chunks if c.tag == b'VEHC' and c.tree]
        free = free_variants(vehicles)
        added: list[tuple[int, int]] = []
        tail = next(i for i, c in enumerate(tree.chunks) if c.tag == b'BINF')
        halo_vo3d = bytes(next(c for c in next(x for x in vehicles if vehicle_id(x) == POLICE_HALO_DONOR).tree.chunks
                               if c.tag == b'VO3D').data)
        halo = {r.key: r.raw for r in v.read_dvo3(halo_vo3d, s.VO3D_DVO3) if r.key in POLICE_HALO_KEYS}

        for city, cars_of_city in planned.items():
            police = police_model(exe, city)
            shared = s.driv3r_globals(game, city)
            bo3 = v.bo3_blocks((game / BO3_DIR / f'{city}.BO3').read_bytes())
            (dpl / LIBRARY / city).mkdir(parents=True, exist_ok=True)
            files: dict[str, tuple[bytes, list]] = {}

            for car in cars_of_city:
                if car['file'] not in files:
                    files[car['file']] = s.driv3r_cars(game, city, car['file'])

                data, cars = files[car['file']]
                base = next(x for x in vehicles if vehicle_id(x) == car['donor'])
                car.pop('added_id', None)

                if add:
                    if not free.get(car['donor']):
                        print(f'{car["name"]}: no free variant left on {car["donor_name"]}, not added')
                        continue

                    car['added_id'] = car['donor'] | free[car['donor']].pop(0) << VARIANT_SHIFT
                    base = renumbered(base, car['added_id'])
                    variants = [base]
                else:
                    variants = [x for x in vehicles if vehicle_id(x) & TYPE_MASK == car['donor']]

                colours = s.palette(bo3, data, cars[car['car']][2])
                boat = boat_table(exe, bo3, data, cars[car['car']][2]) if car['file'].endswith(BOAT_SUFFIX) else {}
                is_police = struct.unpack_from('<I', data, cars[car['car']][2].offset + s.UPVH_MODEL)[0] == police

                for vehicle in variants:
                    parts = {c.tag: c for c in vehicle.tree.chunks}
                    parts[b'MDXN'].data, parts[b'UPVH'].data, report = s.convert(data, cars[car['car']], shared, parts, ssic_pack,
                                                                                 tinted=bool(colours))

                    if colours:
                        parts[b'VO3D'].data = s.with_palette(bytes(parts[b'VO3D'].data), colours)

                    if boat:
                        parts[b'VO3D'].data = s.with_palette(bytes(parts[b'VO3D'].data), boat, set(boat))

                    if is_police:
                        parts[b'VO3D'].data = s.with_palette(bytes(parts[b'VO3D'].data), halo, set(halo))

                if add:
                    size = d.tree_layout(base.tree)[2]

                    if size > MAX_VEHC:
                        raise SystemExit(f'{car["name"]}: VEHC of {size} bytes, more than the {MAX_VEHC} of DPL\'s largest')

                    tree.chunks.insert(tail, base)
                    added.append((tail - 1, car['added_id']))       # the VEHC ordinal: SSIC is entry 0
                    tail += 1

                with (dpl / LIBRARY / city / f'{car["name"]}.vehc').open('wb') as out:
                    for piece in d.tree_pieces(d.ChunkTree(tree.version, [dataclasses.replace(base, gap_extra=0)])):
                        out.write(piece)

                label = f'new id {car["added_id"]:#x}' if add else f'{len(variants)} variant(s)'
                print(f'{car["name"]} -> {car["donor_name"]} ({car["donor"]:#x}), {label}: {report.splitlines()[0]}')

        if added:
            add_lookups(ssic, added)

        if spool:
            add_tail_guard(tree, ssic)
            with new.open('wb') as out:
                for piece in d.tree_pieces(tree):
                    out.write(piece)

        del tree, ssic, vehicles, variants, vehicle, base, parts     # release the views into the original

    for city, cars_of_city in planned.items():
        out_dir = dpl / LIBRARY / city
        # what to spawn in game: DPLHook's Vehicle tab lists vehicles by the donor id, added cars only the Extended window
        (out_dir / 'cars.txt').write_text(''.join(f'{c["name"]:22s} Driv3r {c["file"]} car {c["car"]:2d} -> spawn {c["donor_name"]} '
                                                  f'(id {c.get("added_id", c["donor"]):#x})\n' for c in cars_of_city), encoding='utf-8')
        # the Extended window's list: one "id name" line per added car
        (out_dir / 'added.txt').write_text(''.join(f'{c["added_id"]:#x} {c["name"]} ({c["donor_name"]})\n' for c in cars_of_city
                                                   if 'added_id' in c), encoding='utf-8')
        print(f'{len(cars_of_city)} cars written to {out_dir}')

    (dpl / LIBRARY / MANIFEST).write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')

    if not spool:
        return

    try:
        os.replace(new, target)
    except PermissionError as error:
        raise SystemExit(f'{target} is in use: close the game and move {new.name} over it') from error

    print(f'spool {target} (levels.txt must load vehicles\\{spool_name} in the level that should show these cars)')


def exe_offset(exe: bytes, va: int) -> int:
    """File offset of virtual address va in a 32-bit PE image."""
    pe = struct.unpack_from('<I', exe, 0x3C)[0]
    sections, optional = struct.unpack_from('<H', exe, pe + 6)[0], struct.unpack_from('<H', exe, pe + 20)[0]
    rva = va - struct.unpack_from('<I', exe, pe + 52)[0]

    for i in range(sections):
        start, size, raw = struct.unpack_from('<III', exe, pe + 24 + optional + 40 * i + 12)

        if start <= rva < start + size:
            return raw + rva - start

    raise SystemExit(f'{va:#x} is not in the exe: pass the folder of Driv3r v1.20')


def driv3r_traffic(exe: bytes, city: str) -> list[tuple[int, float]]:
    """(Driv3r model id = .vvs chunk tag, weight) of the city's moving traffic."""
    offset = exe_offset(exe, TRAFFIC_LISTS[city])
    found: list[tuple[int, float]] = []

    for model, weight in struct.iter_unpack('<If', exe[offset:offset + 8 * TRAFFIC_MAX]):
        if model == TRAFFIC_END:
            return found

        found.append((model, weight))

    raise SystemExit(f'no end to the {city} traffic list: pass the folder of Driv3r v1.20')


def police_model(exe: bytes, city: str) -> int:
    """Driv3r model id of the city's police car."""
    return struct.unpack_from('<I', exe, exe_offset(exe, POLICE_MODELS) + 4 * CITIES.index(city))[0]


def police_car(game: Path, exe: bytes, city: str, cars: list[dict]) -> dict | None:
    """The planned car whose Driv3r model is the city's police car (any of the city's files), None when not added."""
    model = police_model(exe, city)
    files: dict[str, tuple[bytes, list]] = {}

    for car in cars:
        if car['file'] not in files:
            files[car['file']] = s.driv3r_cars(game, city, car['file'])

        data, found = files[car['file']]

        if 'added_id' in car and struct.unpack_from('<I', data, found[car['car']][2].offset + s.UPVH_MODEL)[0] == model:
            return car

    return None


def write_traffic(dpl: Path, game: Path, cities: list[str]) -> None:
    manifest = load_manifest(dpl / LIBRARY / MANIFEST)
    exe = (game / 'Driv3r.exe').read_bytes()

    for city in cities:
        police = police_car(game, exe, city, manifest['cities'].get(city, {}).get('cars', []))
        path = dpl / LIBRARY / city / POLICE_FILE
        path.write_text(f'{police["added_id"]:#x} {police["name"]}\n' if police else '', encoding='utf-8')
        print(f'{path}: ' + (f'{police["name"]} ({police["added_id"]:#x})' if police else 'Driv3r\'s police car is not added'))
        types = [t for t, _, _ in s.driv3r_cars(game, city)[1]]
        added = {types[c['car']]: c['added_id'] for c in manifest['cities'].get(city, {}).get('cars', [])
                 if c['file'] == f'{city}.vvs' and 'added_id' in c}
        listed = driv3r_traffic(exe, city)
        lines = [f'{added[m]:#x} {max(1, round(w * WEIGHT_SCALE))}\n' for m, w in listed if m in added]
        path = dpl / LIBRARY / city / TRAFFIC_FILE
        path.write_text(''.join(lines), encoding='utf-8')
        missing = [f'{m:#x}' for m, _ in listed if m not in added]
        print(f'{path}: {len(lines)} of {len(listed)} Driv3r traffic models' + (f', not added: {", ".join(missing)}' if missing else ''))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('dpl', type=Path)
    ap.add_argument('driv3r', type=Path)
    ap.add_argument('action', choices=('plan', 'build', 'traffic'))
    ap.add_argument('--city', default='miami', choices=(*CITIES, 'all'))
    ap.add_argument('--no-spool', action='store_true', help='build: only the per-car VEHC files, no vehicle spool')
    ap.add_argument('--add', action='store_true', help='build: add the cars as new vehicles, keep every DPL vehicle')
    args = ap.parse_args()

    if not (args.dpl / 'Vehicles' / s.ORIGINAL).is_file():
        raise SystemExit(f'{args.dpl}: no Vehicles/{s.ORIGINAL}, pass your Driver Parallel Lines folder')

    cities = list(CITIES) if args.city == 'all' else [args.city]

    if args.action == 'plan':
        plan(args.dpl, args.driv3r, cities)
    elif args.action == 'traffic':
        write_traffic(args.dpl, args.driv3r, cities)
    elif args.add:      # one spool with every city's cars, spawnable in any level that loads it
        build(args.dpl, args.driv3r, cities, not args.no_spool, True)
    else:
        for city in cities:
            build(args.dpl, args.driv3r, [city], not args.no_spool)


if __name__ == '__main__':
    main()
