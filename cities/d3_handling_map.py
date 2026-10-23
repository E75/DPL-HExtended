"""Writes Mods/Cars/D3/handling-map.txt: per DPL vehicle id the Driv3r car whose handling the Extended hook applies.

- Converted Driv3r cars (cars.json from d3_vehicle_dlc.py): their own Driv3r car, by identity (city + vehicle type).
- DPL's own vehicles (nyc_vehicles.sp, every variant): the Driv3r car with the same wheel layout and the nearest hubs, plus
  a term for the top speed (key 160), so a sports car is not matched to a van of the same size.
- Boats (their own boat step) and trailers (no front wheels, never driven): "none",
  the hook leaves DPL's handling.

A line: `<id> <city> <car> <source>  # <why>`; <car> is the hook's index (block order in <city>.BO3), <source> identity,
map or none. Edit a line by hand to choose another car; the hook reads the file when Driv3r's cars are loaded.

usage: python cities/d3_handling_map.py [--dpl "C:/GOG Games/Driver Parallel Lines"] [--game "C:/GOG Games/DRIVERMAP/DRIV3R"]
"""
import argparse
import json
import math
import mmap
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import d3_vehicle_dlc as dlc  # noqa: E402
import d3_vehicle_swap as s  # noqa: E402
import dpl_vehicles as dv  # noqa: E402
import vehchunk as v  # noqa: E402

CITIES = ('miami',)
MAP_FILE = 'handling-map.txt'
KEY_TOP_SPEED = 160
SPEED_COST = 2.0        # metres of hub distance per unit of |ln(top speed ratio)|
FRONT_HUB = '27'        # a hub layout without it is a trailer
BIKE_WHEELS = 2         # a bike took Driv3r's car handling and DPL crashed in its ground lookup right after the player mounted it
BIKE_DONORS = ('mutsumi1000r', 'AntilliVO3', 'AntilliVO3Specia', 'Chopper', 'Wombat', 'Yamashita900')


@dataclass
class Driv3rCar:
    city: str
    index: int          # block order in <city>.BO3, the hook's car index
    vtype: int
    name: str
    hubs: dict[str, list[float]]
    top_speed: float


def block_order(bo3: bytes) -> dict[int, int]:
    """{Driv3r vehicle type: the hook's car index}: blocks are numbered in file order, the slot table points into them."""
    table, base, slots = struct.unpack_from('<3I', bo3, 16)
    offsets = {}

    for vtype in range(slots):
        slot = struct.unpack_from('<H', bo3, table + vtype * 2)[0]

        if slot != 0xFFFF:
            offsets[vtype] = base + slot

    rank = {offset: i for i, offset in enumerate(sorted(set(offsets.values())))}
    return {vtype: rank[offset] for vtype, offset in offsets.items()}


def car_types(game: Path, city: str, file: str) -> list[int]:
    """Driv3r vehicle type per car of a Vehicles file: the chunk tag of a .vvs car, the UPVH's model id in a .vvv."""
    data, cars = s.driv3r_cars(game, city, file)
    return [t if t >= 0 else struct.unpack_from('<I', data, upvh.offset + s.UPVH_MODEL)[0] for t, _, upvh in cars]


def driv3r_cars(game: Path) -> tuple[list[Driv3rCar], dict[tuple[str, str, int], tuple[int, int]]]:
    """Driv3r cars with wheel hubs and a handling block, and {(city, file, car): (type, hook index)} for every car."""
    found: list[Driv3rCar] = []
    identity: dict[tuple[str, str, int], tuple[int, int]] = {}
    exe = (game / 'Driv3r.exe').read_bytes()
    defaults = struct.unpack_from(f'<{dlc.DRIV3R_KEYS}f', exe, dlc.exe_offset(exe, dlc.DRIV3R_DEFAULTS))

    for city in CITIES:
        bo3 = (game / dlc.BO3_DIR / f'{city}.BO3').read_bytes()
        order, blocks = block_order(bo3), v.bo3_blocks(bo3)
        types: dict[str, list[int]] = {}

        for path in sorted((game / 'Vehicles').glob(f'{city}*.vv?')):
            if path.name.startswith(city + '_') or path.suffix == '.vvs' or path.name == f'{city}.vvv':
                types[path.name] = car_types(game, city, path.name)

                for i, vtype in enumerate(types[path.name]):
                    if vtype in order:
                        identity[(city, path.name, i)] = (vtype, order[vtype])

        for car in dlc.city_cars(game, city):
            vtype = types[car.file][car.car]

            if vtype in order:
                block = {r.key: r.value for r in blocks[vtype]}
                found.append(Driv3rCar(city, order[vtype], vtype, car.name, car.hubs, block.get(KEY_TOP_SPEED, defaults[KEY_TOP_SPEED])))

    return found, identity


def wheel_cost(hubs: dict[str, list[float]], other: dict[str, list[float]]) -> float | None:
    """Summed hub distance with the hubs paired front to back and left to right (the part ids of a bike or truck differ
    between the games); None for another wheel count."""
    if len(hubs) != len(other):
        return None

    def order(h: dict[str, list[float]]) -> list[list[float]]:
        return sorted(h.values(), key=lambda p: (round(p[2], 1), p[0]))

    return sum(math.dist(a, b) for a, b in zip(order(hubs), order(other)))


def nearest(hubs: dict[str, list[float]], top_speed: float, cars: list[Driv3rCar]) -> tuple[Driv3rCar, float, float] | None:
    """The Driv3r car of the same wheel layout (else the same wheel count) with the least hub distance plus speed term:
    (car, hub metres, cost)."""
    best = None
    same_layout = any(dlc.hub_cost(hubs, car.hubs) is not None for car in cars)

    for car in cars:
        hub = dlc.hub_cost(hubs, car.hubs) if same_layout else wheel_cost(hubs, car.hubs)

        if hub is None:
            continue

        # a VO3D without key 160 (DPL's bikes) is compared by its hubs alone
        cost = hub + (SPEED_COST * abs(math.log(top_speed / car.top_speed)) if top_speed > 0 and car.top_speed > 0 else 0.0)

        if best is None or cost < best[2]:
            best = (car, hub, cost)

    return best


def dpl_lines(dpl: Path, cars: list[Driv3rCar]) -> list[str]:
    lines = []

    with (dpl / 'Vehicles' / s.ORIGINAL).open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        names, vehicles = dv.vehicles(mm)

        for parts in vehicles:
            vid = struct.unpack_from('<I', mm, parts['VO3D'].offset)[0]
            vtype = vid & dlc.TYPE_MASK
            name = names[vtype] if vtype < len(names) else f'type {vtype:#x}'
            name += f' v{vid >> dlc.VARIANT_SHIFT}' if vid >> dlc.VARIANT_SHIFT else ''
            upvh = bytes(mm[parts['UPVH'].offset:parts['UPVH'].offset + parts['UPVH'].size])
            hubs = dlc.hub_places(upvh, 0)
            keys = {r.key: r.value for r in v.read_dvo3(mm, parts['VO3D'].offset + 4)}
            top = keys.get(KEY_TOP_SPEED, 0.0)
            bike = len(hubs) == BIKE_WHEELS
            match = nearest(hubs, top, cars) if FRONT_HUB in hubs and not bike else None

            if match is None:
                why = 'bike' if bike else 'no Driv3r car with this wheel layout' if FRONT_HUB in hubs else 'trailer' if hubs else 'no wheels'
                lines.append(f'{vid:#06x} - -1 none      # {name}: {why}, keeps DPL handling')
            else:
                car, hub, _ = match
                lines.append(f'{vid:#06x} {car.city} {car.index} map       # {name} -> {car.name} (type {car.vtype}): {len(hubs)} wheels, '
                             f'hubs {hub:.2f} m apart, top speed {top:.0f} / {car.top_speed:.0f}' + ('' if dlc.hub_cost(hubs, car.hubs) is not None else ', hub ids differ'))

        del names, vehicles

    return lines


def converted_lines(dpl: Path, identity: dict[tuple[str, str, int], tuple[int, int]]) -> list[str]:
    manifest = json.loads((dpl / dlc.LIBRARY / dlc.MANIFEST).read_text(encoding='utf-8'))
    lines = []

    for city, entry in manifest['cities'].items():
        for car in entry['cars']:
            if 'added_id' not in car:
                continue

            vid, name = car['added_id'], car['name']

            if car['file'].endswith(dlc.BOAT_SUFFIX):
                lines.append(f'{vid:#06x} - -1 none      # {name}: boat, its own boat step')
            elif car['donor_name'] in BIKE_DONORS:
                lines.append(f'{vid:#06x} - -1 none      # {name}: bike, keeps DPL handling')
            elif (city, car['file'], car['car']) in identity:
                vtype, index = identity[(city, car['file'], car['car'])]
                lines.append(f'{vid:#06x} {city} {index} identity  # {name} ({car["file"]} car {car["car"]}, type {vtype}) on the {car["donor_name"]}')
            else:
                lines.append(f'{vid:#06x} - -1 none      # {name}: no handling block for its type in {city}.BO3')

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dpl', type=Path, default=Path(r'C:\GOG Games\Driver Parallel Lines'))
    parser.add_argument('--game', type=Path, default=Path(r'C:\GOG Games\DRIVERMAP\DRIV3R'))
    args = parser.parse_args()
    cars, identity = driv3r_cars(args.game)
    head = ['# Driv3r handling per DPL vehicle id, read by DPLHookExtended (Driv3r handling on, "Nearest Driv3r car" on).',
            '# <id> <city> <car> <source>: car = the hook\'s index in <city>.BO3; source identity (a converted Driv3r car),',
            '# map (the nearest Driv3r car by wheel hubs and top speed) or none (DPL\'s own handling). Edit freely;',
            '# written by cities/d3_handling_map.py, which overwrites hand edits.']
    lines = head + converted_lines(args.dpl, identity) + dpl_lines(args.dpl, cars)
    out = args.dpl / dlc.LIBRARY / MAP_FILE
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{out}: {len(lines) - len(head)} vehicles')


if __name__ == '__main__':
    main()
