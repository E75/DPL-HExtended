"""Puts a Driv3r car into a COPY of Driver Parallel Lines' vehicle file, in place of one DPL vehicle (the donor).

usage: d3_vehicle_swap.py <DPL dir> <Driv3r dir> --list [--city miami] [--file miami.vvv]
       d3_vehicle_swap.py <DPL dir> <Driv3r dir> --donor 9 --car 1 [--city miami] [--fresh] [--level] [--flat-tail]
       d3_vehicle_swap.py <DPL dir> <Driv3r dir> --restore-level

The copy is Vehicles/nyc_d3_vehicles.sp (made from nyc_vehicles.sp on first use, again with --fresh); the original is
never written. --level lets the Miami LEVEL of Cities/levels.txt load the copy (SpooledVehicles), --restore-level points
it back; levels.txt is backed up to <DPL dir>/_backup/<date>/ first.

The donor keeps its handling (VO3D), sound bank, container records and hierarchy (UPVH, with the wheel hubs, wheel instance matrices and
locator points such as seats and exhausts moved to the Driv3r car's); its model pack gets the
Driv3r car's geometry and textures. Each Driv3r container goes into the donor container nearest in place and size
(part ids differ between the games); donor containers without one are emptied, Driv3r light glow, flares and beams are
left out. LOD slots carry over by slot (kinds 30/28/24 -> 2/4/8). Spawn the donor in game (DPLHook's Vehicle tab lists
vehicles by the same id) to see the car.
"""
import argparse
import collections
import datetime
import io
import mmap
import os
import shutil
import struct
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import d3city
import dplchunk as d
import vehchunk as v
from dpl_poc import add_material, check_strip, strip_of, strip_of_greedy

REORDER_STRIPS = False      # set by gta5_vehicle.py: a decimated model has no strip order worth keeping

ORIGINAL = 'nyc_vehicles.sp'
COPY = 'nyc_d3_vehicles.sp'
LEVEL_KEY = b'SpooledVehicles'
CITY_LEVEL = 'Miami'
CITY_FILE = b'Cities\\miami_pc.d4c'
# what DPL's first LEVEL (NYC Now) loads as shipped, for the keys the Driv3r tools changed in it before Miami had a LEVEL
NYC_NOW_FILE = b'Cities\\NYC_NOW_pc.d4c'
NYC_NOW_VALUES = {b'LifeSpoolFile': b'LifeEvents\\nyc_now_missions.sp', b'Map': b'Overlays\\NYC_NOW.map',
                  b'SpooledVehicles': b'vehicles\\NYC_vehicles.sp', b'CharacterModels': b'Characters\\nyc.sp',
                  b'Hud': b'Overlays\\HUD-NOW'}
SCALE = 1.0 / 8192          # raw position * SCALE = metres; DPL's own cars use 1/4096 .. 1/131072
D3_STRIDE_FLOATS = 15
KIND_TO_DPL = {30: 2, 28: 4, 24: 8, 1: 1, 0: 0}
DDS_HEADER = 128            # 'DDS ' + 124-byte header; the compressed blocks of every mip level follow
SHADOW_KIND = 0            # LOD kind of the shadow hull (slot 5) in both games: no UVs, drawn without a material
SHADOW_WORD = 0xCCCCCCCC    # DPL's material word on shadow hull meshes; a real material draws the hull visibly
LOD_RECORD = 24
SHADE_SCALE = 255.0         # Driv3r's last vertex float is 0..1, DPL's 0..255
SHADER_TYPE_MASK = 0xFF
# light shader type in both games. Driv3r: 0x41E is its dark/lens shader (drawn dark while the lamps are off): tyre
# treads, the inner shell (floor, boot, parcel shelf, wheel arches) and lenses; 0x1801E (glow bit) the glow quads and
# beams of its light containers, drawn only while the lamps are on. 0x41E parts stay with darkened textures; the light
# containers' quads and lenses go into the donor's lamp containers, which DPL switches with the lamps
LIGHT_SHADER = 0x1E
D3_GLOW_BIT = 0x10000
D3_GLASS_SHADER = 0x19      # windows: the only Driv3r car shader whose alpha is transparency
# DPL substance words per Driv3r car shader type, taken from DPL's shared SSIC car materials: interior (0x17 in both
# games, SSIC material 3) and glass (DPL 0x25, SSIC material 2). Generic glass or body words draw glass and interior in
# the wrong order, so the interior flickers
DPL_CAR_SUBSTANCES = {0x17: (0, 0x4417), D3_GLASS_SHADER: (0xDC0100, 0x5425)}
# the rest takes DPL's most common car words; their alpha blend (0x4000) needs full texture alpha (opaque_dds)
DPL_CAR_SUBSTANCE = (0, 0x4406)
# paint and wheels as on DPL's own cars (tallied over all 329 vehicle packs): paint has a clean and a damaged texture
# (0x4DC0101, 252 times; the Colonna's 0xCDC0101 also tints), wheels one (0xDC0101); the clean texture's alpha is gloss
D3_BODY_SHADER = 0x06
D3_PAINT_BIT = 0x1000       # Driv3r body shader 0x1406: clean texture alpha = gloss, texture 1 (if any) = colour mask
D3_PAINT_MASK = 1
DPL_PAINT_SUBSTANCE = (0x4DC0101, 0x4406)
# DPL's colourable cars (Colonna, Teramo ...): grey paint times lerp(car colour, 1, damaged texture alpha), the same rule
# as Driv3r's colour mask (texture 1, else the clean one); 0x4DC0101 above draws the texture's own colour (racers, police)
DPL_TINT_SUBSTANCE = (0xCDC0101, 0x4406)
# handling keys of a model's paint colours (u32: low byte 0xFF, colour above it): Driv3r's BO3 sets 142-157 for every
# model it colours, DPL's VO3D 142-150; a model without them keeps its texture's colour in both games
PALETTE_KEYS = range(142, 158)
VO3D_DVO3 = 4               # the DVO3 block in a VO3D chunk (after the vehicle id): magic, version, record count
UPVH_MODEL = 0x14           # a Driv3r UPVH holds its model id here, also in a .vvv whose chunk tags give none
DPL_WHEEL_SUBSTANCE = (0xDC0101, 0x4406)
DAMAGE_SHADE = (0.7, 0.7)   # damaged paint texture: darkened evenly; a random factor per 4x4 block drew a blocky camouflage
DARK_SHADE = (0.2, 0.2)     # Driv3r's lens shader parts with the lamps off
DPL_LAMP_SUBSTANCE = (0, 0xC01E)    # DPL's shared lamp material (SSIC material 1)
LAMP_LIFT = 0.004           # metres the lamp glow sits in front of its lens; on the lens plane the body hides it
GLOBALS_PAGE = 29           # Driv3r mesh material page of the city's CarGlobals table
GLOBAL_MATERIAL = 0x10000   # added to a CarGlobals material index, so own and shared materials stay apart
STATIC_PART = (False, False)    # part class (wheel, moving) of a container no node draws
SIDE_MIN = 0.1             # metres off the centre line before a lamp counts as left or right
BEAM_SIZE = 1.0           # metres: a lamp container longer than this is a head light beam, not a lamp
PLACE_WEIGHT = 0.1        # lamp matching: colour distance (chromaticity) + this x distance in metres
TEMPLATE_CENTRE = 0.05      # metres
TEMPLATE_SIZE = 1.0
WHEEL_NODE = 2                  # hierarchy node type of the wheel hubs (parts 27-30, trucks 42/43): they hold the transform
INSTANCE_MATCH = 0.002          # metres: a UPVH matrix whose translation is this close to a wheel's instance offset is one
LAMP_NODE = 8                   # node type DPL switches as a lamp (ActivateLamp 0x5AE599); part id = lamp id, same in both games
LOCATOR_NODE = 3                # node type of the points DPL reads from the locators: seats 10/11 (0x521437), exhausts 6 (0x5B4E0A)
HINGE_NODE = 1                  # node type of the parts with a transform: doors 0x1F/0x20, bonnet 4, boot 5, bumpers 0x23/0x24
SEAT_DOORS = {10: 0x3C, 11: 0x3D}   # seat locator -> its door entry point (0x5212EB: seat 0 -> 0x3C, seat 1 -> 0x3D)
BIKE_HUBS = 2
GRIP_BAND = 0.04                # metres inside the widest point that still count as the grip end
GRIP_SHARE = 0.8                # part of the grip height difference the seat goes up: half left the hands below the grips, all a gap at the seat
SEAT_LIFT_MAX = 0.15            # metres
# indicator lamp id -> head lamp on its side (same x sign in both games) and the other indicator: a car without front
# indicators (Driv3r's taxi) gets the other rear indicator's glow turned round onto the head lamp of that side
INDICATOR_HEAD = {12: 50, 13: 51}
INDICATOR_TWIN = {12: 13, 13: 12}
INDICATOR_LIFT = 0.01           # metres the turned indicator glow sits in front of the head lamp glow
TURN_ROUND = np.array([-1.0, 1.0, -1.0])    # 180 degrees about the up axis
# PDL collision shape codes: the low byte (0 box, 1 cylinder) is the same in both games, the surface byte is not
# (Driv3r body 0x0B / hinged panel 0x19 / wheel 0x0C, DPL 0x09 / 0x17 / 0x0A); DPL uses one cylinder code for all
D3_TO_DPL_SHAPE = {0: 0, 0xB00: 0x900, 0x1900: 0x1700, 0xC01: 0xA01, 0xB01: 0xA01}
SHAPE_CODE = 0x50               # in a PDL part; DPL's 3 floats after it are 0, Driv3r's 0.7 0.6 1
PART_CENTRE = 0x40              # in a PDL part, after the half sizes and the 3 axis rows


def driv3r_cars(game: Path, city: str, name: str | None = None) -> tuple[bytes, list[tuple[int, d.Entry, d.Entry]]]:
    """(file data, per car (Driv3r type, MDPC entry, UPVH entry)) of Vehicles/<city>.vvs, or of Vehicles/<name>. A .vvv
    (the city's special cars, mission cars) holds one MDPC that all its hierarchies share; their type is given as -1."""
    path = game / 'Vehicles' / (name or f'{city}.vvs')

    if not path.is_file():
        raise SystemExit(f'{path} not found: pass your Driv3r folder and a city (miami) or a Vehicles file')

    data = path.read_bytes()
    packs, hierarchies = v.find_all(data, 'MDPC'), v.find_all(data, 'UPVH')

    if len(packs) == 1 < len(hierarchies):
        return data, [(-1, packs[0], h) for h in hierarchies]

    tops = [e for e in d.entries(data, 0) if e.tag != 'MDPC']
    types = [struct.unpack('<I', t.tag.encode('latin1'))[0] for t in tops]
    return data, list(zip(types, packs, hierarchies))


def wheel_containers(data: bytes, offset: int) -> dict[int, set[int]]:
    """{container: hub part ids of the hub nodes drawing it}. A car's wheel is drawn once per hub in both games; a DPL
    bike has one wheel container per hub, a Driv3r bike one for both. (Counting any two nodes also took a truck's lamp
    housing that two lamp nodes draw.)"""
    hubs: dict[int, set[int]] = {}

    for n in v.hierarchy_nodes(data, offset):
        if n.handle != v.NO_HANDLE and n.node_type == WHEEL_NODE:
            hubs.setdefault(n.handle & 0x3FF, set()).add(n.part)

    return hubs


def hub_places(data: bytes, offset: int) -> dict[int, np.ndarray]:
    """{hub part id: hub position in metres}."""
    return {part: np.frombuffer(data, '<f4', 3, at).astype(np.float64)
            for (node, part), at in node_positions(data, offset).items() if node == WHEEL_NODE}


def wheel_mapping(car_wheels: dict[int, set[int]], car_centres: dict[int, np.ndarray], car_hubs: dict[int, np.ndarray],
                  donor_wheels: dict[int, set[int]], donor_centres: dict[int, np.ndarray],
                  donor_hubs: dict[int, np.ndarray]) -> tuple[dict[int, list[int]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """({donor wheel container: Driv3r wheel containers}, {donor container: (Driv3r hub it is stored at, the car's hub
    at the donor's stored hub)}) by hub, not by box: each container sits at one hub (nearest its centre) and goes to the
    donor wheel stored at that hub, moved there (see rebuild). Equal hub sets first (tyre and rim at one hub: Driv3r's
    front right, some DPL cars' front left); a Driv3r bike's one wheel for both hubs goes into both DPL bike wheels."""
    def stored(centre: np.ndarray, hubs: dict[int, np.ndarray], parts: set[int]) -> int:
        return min((p for p in parts if p in hubs), key=lambda p: float(np.linalg.norm(hubs[p] - centre)))

    mapping: dict[int, list[int]] = {}
    placements: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for k, parts in donor_wheels.items():
        if not parts & donor_hubs.keys():
            continue

        hub = stored(donor_centres[k], donor_hubs, parts)
        users = [c for c, hw in car_wheels.items() if hw == parts] or [c for c, hw in car_wheels.items() if hub in hw]
        users = [c for c in users if hub in car_hubs and car_wheels[c] & car_hubs.keys()]

        if users:
            mapping[k] = users
            # limit: one placement per donor container, from its first source; Driv3r keeps a wheel's parts at one hub
            placements[k] = car_hubs[stored(car_centres[users[0]], car_hubs, car_wheels[users[0]])], car_hubs[hub]

    return mapping, placements


def container_points(page: d3city.Page, container: int) -> np.ndarray:
    """Vertex positions (n x 3) of the container's top LOD."""
    groups = next((g for _, g in page.containers[container] if g), [])
    used = [page.mesh_triangles(m).reshape(-1) for g in groups for m in g.meshes]
    used = [u for u in used if len(u)]
    return page.vertices[np.unique(np.concatenate(used)), 0:3].astype(np.float64) if used else np.zeros((0, 3))


def container_centre(page: d3city.Page, container: int) -> np.ndarray:
    """Mean vertex position of the container's top LOD (Driv3r container boxes of small parts are unreliable)."""
    points = container_points(page, container)
    return points.mean(0) if len(points) else np.zeros(3)


def pack_points(mdxn: bytes, container: int) -> np.ndarray:
    """Top-LOD vertex positions (n x 3) in metres of a DPL pack's container (raw position x the container's scale)."""
    pack = d.load_pack(mdxn, 0)
    record = pack.containers[container]
    stride = struct.unpack_from('<4I', mdxn, pack.header[16] + record.stream * 32)[3] // 4
    count, offset = pack.streams[record.stream]
    rows = np.frombuffer(mdxn, '<f4', count * stride, offset).reshape(count, stride)
    used = np.unique(d.container_indices(mdxn, pack, record, range(1))).astype(np.int64)
    return rows[used, 0:3].astype(np.float64) * np.array(record.scale)


def pack_centre(mdxn: bytes, container: int) -> np.ndarray:
    """Mean top-LOD vertex position of a DPL pack's container. Its box does not tell: the SanMarinoSpyder's wheel,
    stored at the front left hub, has its box on the right."""
    return pack_points(mdxn, container).mean(0)


def wheel_frame(group: bytes, hub: np.ndarray) -> list[float]:
    """The donor wheel group's matrix with its pivot at `hub`: DPL steers and spins the wheel about it, in the group's
    frame (turned 180 degrees about the up axis for a wheel stored on the left). Group translations are (x, -z, y); a
    zero pivot stays (most DPL wheels' shadow hull groups)."""
    matrix = list(struct.unpack_from('<16f', group, 4))

    if any(round(x, 3) for x in matrix[12:15]):
        matrix[12:15] = [float(hub[0]), float(-hub[2]), float(hub[1])]

    return matrix


def part_classes(data: bytes, offset: int) -> dict[int, tuple[bool, bool]]:
    """{container: (wheel, moving)}: drawn by a hub node; its first node moves (node word 0 = 1: doors, bonnet, boot
    lid, wheels), the same in both games."""
    wheels = wheel_containers(data, offset)
    classes: dict[int, tuple[bool, bool]] = {}

    for n in v.hierarchy_nodes(data, offset):
        if n.handle != v.NO_HANDLE:
            c = n.handle & 0x3FF
            classes.setdefault(c, (c in wheels, n.moving))

    return classes


def lamp_roles(data: bytes, offset: int) -> dict[int, set[int]]:
    """{container: lamp ids of the lamp nodes drawing it}."""
    roles: dict[int, set[int]] = {}

    for n in v.hierarchy_nodes(data, offset):
        if n.handle != v.NO_HANDLE and n.node_type == LAMP_NODE:
            roles.setdefault(n.handle & 0x3FF, set()).add(n.part)

    return roles


def hinge_nodes(data: bytes | bytearray, offset: int) -> dict[int, tuple[int, int]]:
    """{container: (part id, offset of its transform)} of the hinged parts (type-1 nodes with a transform); the part ids
    name the same part in both games (right door 0x1F, left door 0x20, bonnet 4, boot 5, bumpers 0x23/0x24)."""
    table = v.transform_table(data, offset)
    return {n.handle & 0x3FF: (n.part, table + n.transform * v.UPVH_TRANSFORM_SIZE) for n in v.hierarchy_nodes(data, offset)
            if n.node_type == HINGE_NODE and n.handle != v.NO_HANDLE and n.transform != v.NO_HANDLE}


def movers(data: bytes | bytearray, offset: int) -> dict[int, tuple[int, int, bool]]:
    """{container: (node, part id, drawn by that node itself)} of the nearest hinged node (type 1 with a transform) at or
    above the container's node in the hierarchy (depth-first, node word 6 = subtree size, the same in both games). DPL
    moves the whole subtree with it: a bike's handlebars (0x3C) carry the fork lamp and fairing parts under them, so a
    number plate put in one of those turned with the steering."""
    nodes = v.hierarchy_nodes(data, offset)
    open_nodes: list[tuple[int, int]] = []      # (last node of the subtree, node)
    found: dict[int, tuple[int, int, bool]] = {}

    for i, n in enumerate(nodes):
        while open_nodes and open_nodes[-1][0] < i:
            open_nodes.pop()

        open_nodes.append((i + max(n.subtree, 1) - 1, i))
        hinge = next((k for _, k in reversed(open_nodes)
                      if nodes[k].node_type == HINGE_NODE and nodes[k].transform != v.NO_HANDLE), None)

        if n.handle != v.NO_HANDLE and hinge is not None:
            found.setdefault(n.handle & 0x3FF, (hinge, nodes[hinge].part, hinge == i))

    return found


def drawn_containers(data: bytes, upvh: d.Entry) -> set[int]:
    """The containers the hierarchy's nodes draw: a car's own part of a shared (.vvv) pack."""
    return {n.handle & 0x3FF for n in v.hierarchy_nodes(data, upvh.offset) if n.handle != v.NO_HANDLE}


def driv3r_page(data: bytes, pack: d.Entry) -> d3city.Page:
    """The car's pack; its vertices re-read at the vehicle stride (60 bytes, cities use 48). A mesh whose material lives
    in the city's CarGlobals table (s16 page after the material = 29) gets GLOBAL_MATERIAL added to its material."""
    page = d3city.Page(data, pack.offset, pack.size)
    h = struct.unpack_from('<20I', data, pack.offset)

    # LOD entries can share groups: the bound keeps a mesh from being shifted twice
    for lods in page.containers:
        for _, groups in lods:
            for group in groups:
                for mesh in group.meshes:
                    if mesh.material_page == GLOBALS_PAGE and mesh.material < GLOBAL_MATERIAL:
                        mesh.material += GLOBAL_MATERIAL

    start = pack.offset + ((h[14] + h[12] * 2 + 0xFFF) & ~0xFFF)
    end = pack.offset + h[11]
    page.vertices = np.frombuffer(data, dtype='<f4', count=(end - start) // 4 // D3_STRIDE_FLOATS * D3_STRIDE_FLOATS,
                                  offset=start).reshape(-1, D3_STRIDE_FLOATS)
    return page


def driv3r_globals(game: Path, city: str) -> d3city.Page | None:
    """The city's shared car textures (Vehicles/<City>/CarGlobals<City>.vgt: an MDPC holding only a PCMP). A car material
    id at or above the car's own material count is an index into this table (its UPST lists page 29, material 0..n)."""
    path = game / 'Vehicles' / city.capitalize() / f'CarGlobals{city.capitalize()}.vgt'

    if not path.is_file():
        print(f'warning: {path} not found, shared car textures (glass, lights) keep the donor paint')
        return None

    data = path.read_bytes()
    pack = v.find_all(data, 'MDPC')[0]
    return d3city.Page(data, pack.offset, pack.size)


def d3_material(page: d3city.Page, shared: d3city.Page | None, index: int, texture: int = 0) -> d3city.Material | None:
    """A Driv3r car material: from the city's shared table at GLOBAL_MATERIAL and up (see driv3r_page), else the car's own.
    Telling them apart by the car's own material count picked the wrong table for the taxi's tyres (red lamp texture)
    and for every car of a .vvv, whose pack holds more materials."""
    source, index = (shared, index - GLOBAL_MATERIAL) if index >= GLOBAL_MATERIAL else (page, index)
    return source.material(index, texture) if source else None


def is_light(material: d3city.Material | None) -> bool:
    return material is not None and material.flags[0] & SHADER_TYPE_MASK == LIGHT_SHADER


def is_overlay(material: d3city.Material | None) -> bool:
    """A lamp glow overlay laid on a lens (light shader without the glow bit of flares and beams)."""
    return is_light(material) and not material.flags[0] & D3_GLOW_BIT


def mesh_words(raw: d.RawPack, container: int) -> set[int]:
    """The material words of every mesh of a container (all LODs)."""
    words = set()

    for g in raw.lod_groups[container]:
        if g is not None and raw.group_meshes[g] is not None:
            first, count = raw.group_meshes[g], struct.unpack_from('<H', raw.groups[g], 0x44)[0]
            words.update(struct.unpack_from('<I', raw.meshes[m], 48)[0] for m in range(first, first + count))

    return words


def lamp_containers(raw: d.RawPack, ssic: d.RawPack) -> dict[int, tuple[int, int]]:
    """{donor container: its substance words}: the containers drawn only with a light-shader material of the shared SSIC
    pack. They are the lamps (head, indicator, brake, reverse, plate); DPL switches each with its hierarchy node."""
    page = struct.unpack_from('<H', ssic.header, 4)[0]
    p = ssic.pcmp
    found = {}

    for c in range(len(raw.containers)):
        words = mesh_words(raw, c)

        if not words or any(w >> 16 != page or (w & 0xFFFF) >= len(p.materials) for w in words):
            continue

        firsts = [p.material_pointers[w & 0xFFFF] for w in words]
        flags = [struct.unpack_from('<2I', p.substances[p.pointers[f]]) for f in firsts if f is not None]

        if len(flags) == len(words) and all(f[1] & SHADER_TYPE_MASK == LIGHT_SHADER for f in flags):
            found[c] = flags[0]

    return found


def glow_colour(dds: bytes, uv: np.ndarray) -> np.ndarray:
    """Chromaticity (r, g, b summing to 1) of the brighter half of the texture's texels under the UVs."""
    img = np.asarray(Image.open(io.BytesIO(dds)).convert('RGB'), dtype=np.float64)
    h, w = img.shape[:2]
    texels = img[np.clip((uv[:, 1] % 1 * h).astype(int), 0, h - 1), np.clip((uv[:, 0] % 1 * w).astype(int), 0, w - 1)]
    rgb = texels[texels.sum(1) >= np.median(texels.sum(1))].mean(0)
    return rgb / max(rgb.sum(), 1.0)


def donor_lamp_looks(mdxn: bytes, ssic_mdxn: bytes, lamps: dict[int, tuple[int, int]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{donor lamp container: (vertex centre in metres, glow colour)}: its own vertices and the SSIC lamp atlas texels
    under them. Car-sized lamp containers (the head light beam) are left out."""
    pack, ssic = d.load_pack(mdxn, 0), d.load_pack(ssic_mdxn, 0)
    looks = {}

    for k in sorted(lamps):
        cont = pack.containers[k]
        idx = np.unique(d.container_indices(mdxn, pack, cont, range(1)))
        count, offset = pack.streams[cont.stream]
        vertices = np.frombuffer(mdxn, dtype='<f4', count=count * d.VERTEX_STRIDE // 4, offset=offset).reshape(count, -1)
        lamp = vertices[idx].astype(np.float64)
        place = lamp[:, 0:3] * np.array(cont.scale)
        atlas = d.material_texture(ssic_mdxn, ssic, cont.lods[0][1][0].meshes[0].material) if cont.lods[0][1] else None

        if len(idx) and atlas and np.ptp(place, 0).max() < BEAM_SIZE:
            looks[k] = (place.mean(0), glow_colour(atlas, lamp[:, 6:8]))

    return looks


def turned_indicator(page: d3city.Page, shared: d3city.Page | None, twin: list[d3city.Mesh],
                     head: tuple[list[d3city.Mesh], np.ndarray, np.ndarray]) -> list[tuple[int, np.ndarray]]:
    """The glow quads of a rear indicator turned round onto the head lamp glow `head` of the other side: new vertices
    appended to page.vertices. The tail (deformation) turns with them."""
    glows = [m for m in twin if d3_material(page, shared, m.material).flags[0] & D3_GLOW_BIT] or twin
    used = np.unique(np.concatenate([page.mesh_triangles(m).reshape(-1) for m in glows]))
    rows = page.vertices[used].astype(np.float32)

    for at in (0, 3, 8, 11):        # position, normal, tail offset, tail vector
        rows[:, at:at + 3] *= TURN_ROUND

    head_rows = page.vertices[np.unique(np.concatenate([page.mesh_triangles(m).reshape(-1) for m in head[0]]))]
    normal = head_rows[:, 3:6].mean(0)
    rows[:, 0:3] += head[1] + normal / max(float(np.linalg.norm(normal)), 1e-6) * INDICATOR_LIFT - rows[:, 0:3].mean(0)
    new_index = dict(zip(used.tolist(), range(len(page.vertices), len(page.vertices) + len(used))))
    page.vertices = np.vstack([page.vertices, rows.astype(page.vertices.dtype)])
    return [(m.material, np.vectorize(new_index.get)(page.mesh_triangles(m))) for m in glows]


def lamp_triangles(page: d3city.Page, shared: d3city.Page | None, lights: set[int],
                   donor_looks: dict[int, tuple[np.ndarray, np.ndarray]], car_roles: dict[int, set[int]],
                   donor_roles: dict[int, set[int]]) -> dict[int, list[tuple[int, np.ndarray]]]:
    """{donor lamp container: [(Driv3r material, triangles)]}: each donor lamp gets the Driv3r light containers (glow quads
    and lit lenses, top LOD; Driv3r draws them only with the lamps on, as DPL its lamp containers) with its lamp id at
    its end of the car and on its side: both games number lamps alike (12/13 indicators, 50/51 head lamps, 46/47 and
    48/49 rear lamps). A front indicator the Driv3r car lacks gets the other rear indicator turned round onto its head
    lamp (turned_indicator); a lamp without an id match takes the light container at its end and side whose glow colour
    is nearest to the donor lamp's own (then the nearest in place). Container boxes of small lamp parts are off in both
    games, so vertices place them. Beam meshes (longer than BEAM_SIZE) stay out."""
    looks = {}

    for s in lights:
        meshes = [m for g in page.containers[s][0][1] for m in g.meshes
                  if len(page.mesh_triangles(m)) and np.ptp(page.vertices[page.mesh_triangles(m).reshape(-1), 0:3], 0).max() < BEAM_SIZE]

        if meshes:
            glows = [m for m in meshes if d3_material(page, shared, m.material).flags[0] & D3_GLOW_BIT] or meshes
            glow = max(glows, key=lambda m: len(page.mesh_triangles(m)))
            place = page.vertices[np.unique(np.concatenate([page.mesh_triangles(m).reshape(-1) for m in meshes])), 0:3]
            uv = page.vertices[np.unique(page.mesh_triangles(glow)), 6:8]
            looks[s] = (meshes, place.mean(0), glow_colour(d3_material(page, shared, glow.material).dds, uv))

    def same_place(a: np.ndarray, b: np.ndarray) -> bool:
        return a[2] * b[2] > 0 and (min(abs(a[0]), abs(b[0])) <= SIDE_MIN or a[0] * b[0] > 0)

    def with_id(lamp: int, centre: np.ndarray | None) -> list[int]:
        return [s for s in looks if lamp in car_roles.get(s, set()) and (centre is None or same_place(looks[s][1], centre))]

    def with_id_of(ids: set[int], centre: np.ndarray) -> list[int]:
        return [s for lamp in sorted(ids) for s in with_id(lamp, centre)]

    # each light container goes to the nearest donor lamp of its id at its end and side: drawn by two donor lamps of one
    # id it lights twice on top of itself, and a third brake lamp took the rear brake lamp's glow from it
    owner: dict[int, int] = {}

    for s in looks:
        rivals = [k for k, (centre, _) in donor_looks.items() if s in with_id_of(donor_roles.get(k, set()), centre)]

        if rivals:
            owner[s] = min(rivals, key=lambda k: float(np.linalg.norm(donor_looks[k][0] - looks[s][1])))

    found: dict[int, list[tuple[int, np.ndarray]]] = {}

    for k, (centre, colour) in sorted(donor_looks.items()):
        ids = donor_roles.get(k, set())
        matches = with_id_of(ids, centre)
        same = [s for s in matches if owner[s] == k]

        if matches and not same:
            continue

        near = [s for s, (_, place, _) in looks.items() if same_place(place, centre)]
        indicator = next((lamp for lamp in sorted(ids) if lamp in INDICATOR_HEAD), None)
        heads = with_id(INDICATOR_HEAD[indicator], centre) if indicator else []
        twins = [s for s in with_id(INDICATOR_TWIN[indicator], None) if looks[s][1][2] * centre[2] < 0] if indicator else []

        if same:
            found[k] = [(m.material, page.mesh_triangles(m)) for s in same for m in looks[s][0]]
        elif heads and twins:
            found[k] = turned_indicator(page, shared, looks[twins[0]][0], looks[heads[0]])
        elif near:
            s = min(near, key=lambda s: float(np.linalg.norm(looks[s][2] - colour)
                                               + PLACE_WEIGHT * np.linalg.norm(looks[s][1] - centre)))
            found[k] = [(m.material, page.mesh_triangles(m)) for m in looks[s][0]]

    return found


def light_containers(page: d3city.Page, shared: d3city.Page | None) -> set[int]:
    """The Driv3r containers drawn only with the light shader (flares, beams): nothing of them is kept."""
    found = set()

    for c, lods in enumerate(page.containers):
        materials = [d3_material(page, shared, m.material) for _, groups in lods for g in groups for m in g.meshes]

        if materials and all(is_light(m) for m in materials):
            found.add(c)

    return found


def is_template(box: np.ndarray) -> bool:
    """A small part stored at the origin (the wheel template): the hierarchy draws it elsewhere, never as a body part."""
    lo, hi = box.min(0), box.max(0)
    return bool(np.abs((lo + hi) / 2).max() < TEMPLATE_CENTRE and (hi - lo).max() < TEMPLATE_SIZE)


def normalised(boxes: list[np.ndarray]) -> list[np.ndarray]:
    """Each box as (centre, size) in units of the whole car's extent, so cars of different sizes compare."""
    corners = np.concatenate(boxes)
    lo, hi = corners.min(0), corners.max(0)
    mid, extent = (lo + hi) / 2, hi - lo
    return [np.concatenate((((b.min(0) + b.max(0)) / 2 - mid) / extent, (b.max(0) - b.min(0)) / extent)) for b in boxes]


def match_containers(car_boxes: list[np.ndarray], donor_boxes: list[np.ndarray], car_lods: list[int], donor_lods: list[int],
                     skip: set[int], lamps: set[int], car_classes: dict[int, tuple[bool, bool]],
                     donor_classes: dict[int, tuple[bool, bool]], car_movers: dict[int, tuple[int, int, bool]],
                     donor_movers: dict[int, tuple[int, int, bool]]) -> dict[int, list[int]]:
    """{donor container: Driv3r containers}: each Driv3r container goes to the donor container nearest in place and size.
    Templates, Driv3r `skip` and donor `lamps` stay out; donor wheels take nothing (wheels go by hub, wheel_mapping).
    A container drawn at several LODs only goes to a donor container that has several too: DPL leaves single-LOD
    containers (lamps, plates, a wide hidden part) out of the normal view.
    Moving parts go by their hinged node (see movers; part ids are the same in both games): a hinged part only into the
    donor's part of that id (by place both doors of a car went into the donor's one moving door and swung together),
    one Driv3r part each (a fire engine's two right doors share an id: the second is static); a part under a hinged
    node (handlebars, fork, door) follows its node's part, or the donor's parts under that id; a hinged part the donor
    lacks (bumpers) and every part without one only into donor parts that nothing moves (a bike's number plate in a
    container under the handlebars turned with the steering). A moving part without a hinge (a truck's wing mirrors)
    goes to a donor part of its class (see part_classes), else to a static one."""
    cars = [c for c in range(len(car_boxes)) if c not in skip and not is_template(car_boxes[c])]
    donors = [k for k in range(len(donor_boxes)) if k not in lamps and not is_template(donor_boxes[k])]
    donor_shapes = normalised([donor_boxes[k] for k in donors])
    shapes = dict(zip(cars, normalised([car_boxes[c] for c in cars])))
    index = {k: i for i, k in enumerate(donors)}
    usable = [i for i in range(len(donors)) if not donor_classes.get(donors[i], STATIC_PART)[0]]
    hinged = {k: part for k, (_, part, own) in donor_movers.items() if own}
    moved_by = {k: part for k, (_, part, _) in donor_movers.items()}
    went: dict[int, int | None] = {}        # Driv3r hinged node -> donor container its own part went to (None: static)
    result: dict[int, list[int]] = {}

    def cost(c: int, i: int) -> float:
        return float(np.abs(shapes[c] - donor_shapes[i]).sum())

    def fits(c: int, i: int) -> bool:
        return car_lods[c] < 2 or donor_lods[donors[i]] >= 2

    def still(c: int, part_class: tuple[bool, bool] = STATIC_PART) -> list[int]:
        return [i for i in usable if donors[i] not in moved_by and fits(c, i)
                and donor_classes.get(donors[i], STATIC_PART) == part_class]

    def own_hinge(c: int) -> bool:
        return c in car_movers and car_movers[c][2]

    # without the LOD rule: a hinged part is drawn whatever its LODs (the DPL school bus's right door has one)
    def same_part(c: int) -> list[int]:
        return [i for i in usable if hinged.get(donors[i]) == car_movers[c][1]] if own_hinge(c) else []

    def pools(c: int) -> list[list[int]]:
        if c not in car_movers:
            return [still(c, car_classes.get(c, STATIC_PART)), still(c)]

        node, part, own = car_movers[c]

        if own:
            return [[i for i in same_part(c) if donors[i] not in result], still(c)]

        if node in went:
            return [[] if went[node] is None else [index[went[node]]], still(c)]

        return [[i for i in usable if moved_by.get(donors[i]) == part and fits(c, i)], still(c)]

    # hinged parts first, the nearest claiming its donor part; then the parts under them follow
    for c in sorted(cars, key=lambda c: (not own_hinge(c), min((cost(c, i) for i in same_part(c)), default=0.0))):
        candidates = pools(c)
        allowed = next((pool for pool in candidates if pool), None)
        best = None if allowed is None else min(allowed, key=lambda i: cost(c, i))

        if own_hinge(c):
            went[car_movers[c][0]] = donors[best] if best is not None and allowed is candidates[0] else None

        if best is not None:     # a part without a donor part of its kind is left out
            result.setdefault(donors[best], []).append(c)

    return {k: sorted(cs) for k, cs in result.items()}


def node_positions(data: bytes | bytearray, offset: int) -> dict[tuple[int, int], int]:
    """{(node type, part id): offset of the node's position (3 floats, the start of its transform)}."""
    table = v.transform_table(data, offset)
    found: dict[tuple[int, int], int] = {}

    for n in v.hierarchy_nodes(data, offset):
        if n.transform != v.NO_HANDLE:
            found.setdefault((n.node_type, n.part), table + n.transform * v.UPVH_TRANSFORM_SIZE)

    return found


def locator_positions(data: bytes | bytearray, offset: int, size: int) -> dict[int, list[int]]:
    """{part id: offsets of the positions (3 floats) of its locator nodes, in node order}; `size` = locator entry size."""
    table = v.locator_table(data, offset)
    found: dict[int, list[int]] = {}

    for n in v.hierarchy_nodes(data, offset):
        if n.node_type == LOCATOR_NODE and n.locator != v.NO_HANDLE:
            found.setdefault(n.part, []).append(table + n.locator * size)

    return found


def move_locators(donor_upvh: bytes, car_data: bytes, car_upvh: int) -> tuple[bytes, int]:
    """The donor's hierarchy with each locator point (seats, exhausts, the rest of the type-3 points) that the Driv3r car
    also has at the car's position; the donor's orientation stays (Driv3r stores none). A part the donor has fewer
    times than the car (a second exhaust stack) keeps the donor's count. A Driv3r seat on the centre line where the
    donor's sits at a door (some Driv3r cars) stays the donor's: DPL gets in from the side door to that seat.
    Returns (new UPVH, points moved)."""
    out = bytearray(donor_upvh)
    car = locator_positions(car_data, car_upvh, v.D3_LOCATOR_SIZE)
    moved = 0

    for part, places in locator_positions(out, 0, v.DPL_LOCATOR_SIZE).items():
        for at, source in zip(places, car.get(part, [])):
            centred = abs(struct.unpack_from('<f', car_data, source)[0]) < SIDE_MIN <= abs(struct.unpack_from('<f', out, at)[0])

            if part in SEAT_DOORS and centred:
                continue

            out[at:at + 12] = car_data[source:source + 12]
            moved += 1

    return bytes(out), moved


def grip_height(points: np.ndarray, seat: np.ndarray) -> float | None:
    """Height above the seat point of a bike's handlebar grips: the widest points 0.2-1.4 m ahead of the seat, between
    0.2 m below and 0.25 m above it (higher up a scooter's screen or mirrors are wider)."""
    rel = points - seat
    rel = rel[(rel[:, 2] > 0.2) & (rel[:, 2] < 1.4) & (rel[:, 1] > -0.2) & (rel[:, 1] < 0.25)]

    if not len(rel):
        return None

    width = np.abs(rel[:, 0]).max()
    return float(rel[np.abs(rel[:, 0]) > width - GRIP_BAND, 1].mean())


def lift_bike_seat(upvh: bytes, donor_upvh: bytes, car_points: np.ndarray, donor_points: np.ndarray) -> tuple[bytes, float]:
    """The bike's hierarchy with its seats (10/11) raised by part of how much higher the Driv3r bike's grips sit above
    the seat than the donor's: DPL places its own crouched rider pose at the seat, so on Driv3r's upright bikes (bars
    and tank 10-15 cm higher, same seat height) he sank into the bike. Returns (new UPVH, lift in metres)."""
    out = bytearray(upvh)
    seats = {p: at for p, at in locator_positions(out, 0, v.DPL_LOCATOR_SIZE).items() if p in SEAT_DOORS}
    donor_seat = locator_positions(donor_upvh, 0, v.DPL_LOCATOR_SIZE).get(10)

    if 10 not in seats or donor_seat is None:
        return bytes(out), 0.0

    car_grip = grip_height(car_points, np.frombuffer(out, '<f4', 3, seats[10][0]))
    donor_grip = grip_height(donor_points, np.frombuffer(donor_upvh, '<f4', 3, donor_seat[0]))

    if car_grip is None or donor_grip is None:
        return bytes(out), 0.0

    lift = float(np.clip(GRIP_SHARE * (car_grip - donor_grip), 0.0, SEAT_LIFT_MAX))

    for at in (at for places in seats.values() for at in places):
        struct.pack_into('<f', out, at + 4, struct.unpack_from('<f', out, at + 4)[0] + lift)

    return bytes(out), lift


def move_hinges(donor_upvh: bytes, car_data: bytes, car_upvh: int, mapping: dict[int, list[int]]) -> tuple[bytes, int]:
    """The donor's hierarchy with each hinged part's transform (pivot, hinge axis) taken from the Driv3r part of the same
    id that fills it: DPL swings the part about its node's transform, so a Driv3r door on the Hotrod's hinge opened in
    its middle. Returns (new UPVH, transforms moved)."""
    out = bytearray(donor_upvh)
    car = hinge_nodes(car_data, car_upvh)
    moved = 0

    for container, (part, at) in hinge_nodes(out, 0).items():
        source = next((car[c][1] for c in mapping.get(container, []) if c in car and car[c][0] == part), None)

        if source is not None:
            out[at:at + v.UPVH_TRANSFORM_SIZE] = car_data[source:source + v.UPVH_TRANSFORM_SIZE]
            moved += 1

    return bytes(out), moved


def move_hubs(donor_upvh: bytes, car_data: bytes, car_upvh: int) -> tuple[bytes, int, int]:
    """The donor's hierarchy with its wheel hubs at the Driv3r car's (both games number the hub nodes alike); (new UPVH,
    hubs moved, instance matrices moved). Both games store the wheel model at the front right hub, so it stays put."""
    out = bytearray(donor_upvh)
    car = node_positions(car_data, car_upvh)
    old: dict[tuple[int, int], np.ndarray] = {}
    new: dict[tuple[int, int], np.ndarray] = {}

    for key, at in node_positions(out, 0).items():
        if key[0] == WHEEL_NODE and key in car:
            old[key] = np.frombuffer(out, dtype='<f4', count=3, offset=at).copy()
            out[at:at + 12] = car_data[car[key]:car[key] + 12]
            new[key] = np.frombuffer(out, dtype='<f4', count=3, offset=at).copy()

    instances = move_wheel_instances(out, old, new)
    return bytes(out), len(new), instances


def move_wheel_instances(upvh: bytearray, old: dict[tuple[int, int], np.ndarray],
                         new: dict[tuple[int, int], np.ndarray]) -> int:
    """Rewrites, in place, the matrices DPL draws each wheel with: the wheel model sits at one hub (the template) and
    hub i gets the row-vector matrix R_i with translation hub_i - template @ R_i (UPVH holds two per wheel after the
    collision parts). Found by that translation from the donor's hubs, rewritten from the new ones; returns the count."""
    moved = 0

    for at in range(0, len(upvh) - 63, 4):
        m = np.frombuffer(upvh, dtype='<f4', count=16, offset=at).reshape(4, 4)
        rotation = m[:3, :3]

        with np.errstate(invalid='ignore', over='ignore'):
            if m[3, 3] != 1 or m[:3, 3].any() or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
                continue

        # a truck has two templates (front wheel, rear twin wheel); a symmetric donor can match one matrix twice
        pair = next(((wheel, template) for wheel in old for template in old if wheel != template
                     and np.abs(m[3, :3] - (old[wheel] - old[template] @ rotation)).max() < INSTANCE_MATCH), None)

        if pair:
            upvh[at + 48:at + 60] = (new[pair[0]] - new[pair[1]] @ rotation).astype('<f4').tobytes()
            moved += 1

    return moved


def collision_records(data: bytes | bytearray, offset: int, locator_size: int) -> tuple[int, list[tuple[float, bytes]]]:
    """(offset of the UPVH's PDL block, per collision record (bounding radius, its parts))."""
    at = v.pdl_block(data, offset, locator_size)
    count, _, table, first = struct.unpack_from('<4I', data, at)
    records = []
    part = at + first

    for r in range(count):
        parts, _, radius, _ = struct.unpack_from('<IIfI', data, at + table + 16 * r)
        records.append((radius, bytes(data[part:part + parts * v.PDL_PART_SIZE])))
        part += parts * v.PDL_PART_SIZE

    return at, records


def body_half_width(upvh: bytes | bytearray) -> float:
    """Half width of a DPL UPVH's body box (PDL record 0, part 0), in the vehicle's axes."""
    part = collision_records(upvh, 0, v.DPL_LOCATOR_SIZE)[1][0][1]
    half = np.array(struct.unpack_from('<3f', part, 0))
    rows = np.array([struct.unpack_from('<3f', part, 16 + 16 * i) for i in range(3)])
    return float((np.abs(rows).T @ half)[0])


def move_doors(upvh: bytes, donor_upvh: bytes) -> tuple[bytes, int]:
    """The door entry points (0x3C/0x3D; Driv3r has none, so they are still the donor's) placed on the Driv3r body as the
    donor's sit on the donor's: the same gap outside the body side and moved along the car with their seat. DPL's entry
    planner walks round the body box (PDL record 0, the Driv3r car's after move_collision) to the door point and then
    to the seat; with the donor's points a narrower body left them up to 0.4 m out in the road and a wider one inside
    the body, so the player walked round or slid in late. Each point goes to its seat's side: a Driv3r seat 0 on the
    donor's seat 1 side (most boats) left the door across the hull and the walk-in never ended.
    Call after move_locators and move_collision. Returns (new UPVH, points moved)."""
    out = bytearray(upvh)
    grow = body_half_width(out) - body_half_width(donor_upvh)
    places = locator_positions(out, 0, v.DPL_LOCATOR_SIZE)
    donor_places = locator_positions(donor_upvh, 0, v.DPL_LOCATOR_SIZE)
    moved = 0

    for seat, door in SEAT_DOORS.items():
        if seat not in places or door not in places or seat not in donor_places:
            continue

        forward = struct.unpack_from('<f', out, places[seat][0] + 8)[0] - struct.unpack_from('<f', donor_upvh, donor_places[seat][0] + 8)[0]
        seat_x = struct.unpack_from('<f', out, places[seat][0])[0]

        for at in places[door]:
            x, y, z = struct.unpack_from('<3f', out, at)
            side = seat_x if abs(seat_x) >= SIDE_MIN else x
            struct.pack_into('<3f', out, at, float(np.copysign(abs(x) + grow, side)), y, z + forward)
            moved += 1

    return bytes(out), moved


def dpl_shapes(parts: bytes) -> bytes:
    """Driv3r PDL parts with DPL's shape codes (and DPL's zero floats after them)."""
    out = bytearray(parts)

    for at in range(0, len(out), v.PDL_PART_SIZE):
        code = struct.unpack_from('<I', out, at + SHAPE_CODE)[0]

        if code not in D3_TO_DPL_SHAPE:
            raise SystemExit(f'Driv3r collision shape {code:#x} has no DPL code (D3_TO_DPL_SHAPE)')

        struct.pack_into('<I3f', out, at + SHAPE_CODE, D3_TO_DPL_SHAPE[code], 0, 0, 0)

    return bytes(out)


def bounding_radius(parts: bytes) -> float:
    """DPL's radius of a node's collision record: the largest |centre| + |half size| of its parts. Driv3r's is larger
    for a quarter of its node records; the body record (part 0 bounds the rest) follows the same rule in both games."""
    return max(float(np.linalg.norm(struct.unpack_from('<3f', parts, at + PART_CENTRE)) + np.linalg.norm(struct.unpack_from('<3f', parts, at)))
               for at in range(0, len(parts), v.PDL_PART_SIZE))


def move_collision(donor_upvh: bytes, car_data: bytes, car_upvh: int) -> tuple[bytes, int]:
    """The donor's hierarchy with the Driv3r car's collision: the body record and each record of a node the car also
    has (same node type and part id: wheel hubs, doors, bonnet ...). DPL sizes the wheels (radius, width) and the body
    box from these, so the donor's left the car floating on the donor's wheels. Records keep the donor's order, so its
    nodes' record indices stay valid; the PDL block may change size (header word 6). Returns (new UPVH, records moved)."""
    at, donor = collision_records(donor_upvh, 0, v.DPL_LOCATOR_SIZE)
    _, car = collision_records(car_data, car_upvh, v.D3_LOCATOR_SIZE)
    car_record = {(n.node_type, n.part): n.record for n in v.hierarchy_nodes(car_data, car_upvh) if n.record != v.NO_HANDLE}
    source = {0: 0}

    for n in v.hierarchy_nodes(donor_upvh, 0):
        if n.record != v.NO_HANDLE and (n.node_type, n.part) in car_record:
            source.setdefault(n.record, car_record[(n.node_type, n.part)])

    records = [(car[source[r]][0] if r == 0 else bounding_radius(car[source[r]][1]), dpl_shapes(car[source[r]][1]))
               if r in source else record for r, record in enumerate(donor)]
    old_size = struct.unpack_from('<I', donor_upvh, v.UPVH_PDL_SIZE)[0]
    _, _, table, first = struct.unpack_from('<4I', donor_upvh, at)
    head = bytearray(donor_upvh[at:at + first])
    struct.pack_into('<I', head, 4, sum(len(p) // v.PDL_PART_SIZE for _, p in records))
    offset = first - table

    for r, (radius, parts) in enumerate(records):
        struct.pack_into('<IIfI', head, table + 16 * r, len(parts) // v.PDL_PART_SIZE, offset, radius, 0)
        offset += len(parts) - v.PDL_OFFSET_SHORT

    block = bytes(head) + b''.join(p for _, p in records)
    out = bytearray(donor_upvh[:at] + block + donor_upvh[at + old_size:])
    struct.pack_into('<I', out, v.UPVH_PDL_SIZE, len(block))
    return bytes(out), len(source)


def dpl_vertices(car: np.ndarray, flat_tail: bool, still: list[int]) -> bytearray:
    """Driv3r vertices (15 floats) as DPL's: position / SCALE, normal, uv, tail (offset / SCALE, 3 floats, shade x 255).
    The `still` vertices get an empty tail, as on DPL's wheels: DPL deforms by it and a wheel then turns off its hub."""
    out = car.astype('<f4').copy()
    out[:, 0:3] /= SCALE

    if flat_tail:
        out[:, 8:14] = 0
    else:
        out[:, 8:11] /= SCALE

    out[still, 8:14] = 0
    out[:, 14] *= SHADE_SCALE
    return bytearray(out.tobytes())


def opaque_dds(dds: bytes) -> bytes:
    """The DDS with every pixel's alpha at full. DPL's car substances blend by alpha (bit 0x4000) and DPL's own car
    textures keep alpha full; Driv3r's body alpha is a paint mask, which would draw the body see-through."""
    out = bytearray(dds)
    fourcc = bytes(dds[84:88])

    if fourcc in (b'DXT3', b'DXT5'):
        blocks = np.frombuffer(out, dtype=np.uint8, offset=DDS_HEADER, count=(len(out) - DDS_HEADER) // 16 * 16).reshape(-1, 16)

        if fourcc == b'DXT3':
            blocks[:, 0:8] = 0xFF        # 4-bit explicit alpha per pixel
        else:
            blocks[:, 0:2] = 0xFF        # both alpha end points full, every index on the first
            blocks[:, 2:8] = 0

    return bytes(out)


def shaded_dds(dds: bytes, shade: tuple[float, float]) -> bytes:
    """The DXT texture with each 4x4 block's colour end points darkened by a random factor in `shade`: DPL's second
    (damaged) paint texture, which Driv3r lacks (it damages paint only by deformation), and Driv3r's dark parts."""
    out = bytearray(dds)
    fourcc = bytes(dds[84:88])
    size = 8 if fourcc == b'DXT1' else 16
    colour = 0 if fourcc == b'DXT1' else 8
    blocks = np.frombuffer(out, dtype=np.uint8, offset=DDS_HEADER, count=(len(out) - DDS_HEADER) // size * size).reshape(-1, size)
    ends = blocks[:, colour:colour + 4].copy().view('<u2').astype(np.int64)
    shade = np.random.default_rng(0).uniform(*shade, len(blocks))[:, None]
    r, g, b = (ends >> 11) & 31, (ends >> 5) & 63, ends & 31
    dark = (r * shade).astype(np.int64) << 11 | (g * shade).astype(np.int64) << 5 | (b * shade).astype(np.int64)

    if fourcc == b'DXT1':       # DXT1 reads its block as 4 colours only while end point 0 > end point 1
        keep = (ends[:, 0] > ends[:, 1]) != (dark[:, 0] > dark[:, 1])
        dark[keep] = ends[keep]

    blocks[:, colour:colour + 4] = dark.astype('<u2').view(np.uint8).reshape(-1, 4)
    return bytes(out)


def with_alpha(dds: bytes, source: bytes) -> bytes:
    """The DXT3/DXT5 texture with the alpha half of every block taken from `source` (same format, size and mips)."""
    if dds[84:88] not in (b'DXT3', b'DXT5') or dds[:DDS_HEADER] != source[:DDS_HEADER] or len(dds) != len(source):
        return dds      # limit: a mask of another layout keeps the clean alpha, upgrade: decode and re-encode it
    out = np.frombuffer(dds, dtype=np.uint8).copy()
    alpha = np.frombuffer(source, dtype=np.uint8)
    body = slice(DDS_HEADER, DDS_HEADER + (len(dds) - DDS_HEADER) // 16 * 16)
    out[body].reshape(-1, 16)[:, :8] = alpha[body].reshape(-1, 16)[:, :8]
    return out.tobytes()


def container_boxes(data: bytes | memoryview, first: int, count: int, record_size: int, box_offset: int) -> list[np.ndarray]:
    """The 8 box corners (8 x 3) of each container record (Driv3r: 0x188 bytes, box at +0x28; DPL: 0x14C, +0x24)."""
    return [np.array([struct.unpack_from('<3f', data, first + i * record_size + box_offset + k * 16) for k in range(8)]) for i in range(count)]


def rebuild(raw: d.RawPack, page: d3city.Page, shared: d3city.Page | None, car_boxes: list[np.ndarray],
            mapping: dict[int, list[int]], wheels: set[int],
            lamp_parts: dict[int, list[tuple[int, np.ndarray]]], flat_tail: bool,
            placements: dict[int, tuple[np.ndarray, np.ndarray]] | None = None, tinted: bool = False) -> tuple[int, int]:
    """Replaces the pack's geometry; returns (meshes written, Driv3r materials that got a DPL material). `wheels`: the
    donor's wheel containers; its lamp containers are filled from `lamp_parts` with DPL's lamp substance words.
    `tinted`: the model has a colour palette, so its paint keeps the alpha mask and takes DPL's tinted paint words.
    `placements`: {donor wheel container: (hub its Driv3r wheel is stored at, hub to move it to)}; a wheel moved to the
    other side is turned 180 degrees about the up axis, and the group keeps the donor's frame with the new pivot."""
    template_group, template_mesh = bytes(raw.groups[0]), bytes(raw.meshes[0])
    lamp_groups = {c: bytes(raw.groups[raw.lod_groups[c][0]]) for c in lamp_parts}
    # a Driv3r wheel group (identity frame at its own hub) in a wheel slot stored on the left swung about the far hub
    # when steering, like a door; a moved rear wheel kept the front hub's pivot
    frames = {(c, slot): wheel_frame(raw.groups[raw.lod_groups[c][slot]], target)
              for c, (_, target) in (placements or {}).items() for slot in range(7) if raw.lod_groups[c][slot] is not None}
    # a mesh's material word is page << 16 | material in that page: 0x2000 is the shared SSIC pack, so the materials
    # added here need the pack's own page (a material past a page's count crashes the loader at 0x5EE51E)
    own_page = struct.unpack_from('<H', raw.header, 4)[0]
    raw.groups, raw.group_meshes, raw.meshes, raw.mesh_indices = [], [], [], []
    raw.indices = bytearray()
    raw.descriptors = [raw.descriptors[0]]
    still: set[int] = set()
    # DPL stores each container slot's vertices as one run of its own, sized by the LOD record; Driv3r's shared pool
    # (slots and parts interleaved) drew parts flickering and wheels wrong, so every slot gets its own copy
    order: list[int] = []
    block: dict[int, int] = {}
    lamp_runs: dict[int, set[int]] = {}
    textures: dict[bytes, int] = {}
    materials: dict[tuple[int, bool, bool], int] = {}
    runs: dict[int, list[tuple[int, int]]] = {}

    def material_of(index: int, wheel: bool = False, lamp: bool = False) -> int:
        """A DPL material for a Driv3r car material, with the substance words of DPL's own car parts of that kind. `lamp`:
        a part of a lamp container (keeps its alpha); Driv3r's dark/lens shader parts elsewhere get darkened textures."""
        if (index, wheel, lamp) not in materials:
            found = d3_material(page, shared, index)
            added = None

            if found:
                shader = found.flags[0] & SHADER_TYPE_MASK
                paint = shader == D3_BODY_SHADER and bool(found.flags[0] & D3_PAINT_BIT)
                substance = (DPL_LAMP_SUBSTANCE if lamp else (DPL_TINT_SUBSTANCE if tinted else DPL_PAINT_SUBSTANCE) if paint
                             else DPL_WHEEL_SUBSTANCE if wheel and shader == D3_BODY_SHADER
                             else DPL_CAR_SUBSTANCES.get(shader, DPL_CAR_SUBSTANCE))

                if not lamp and shader == LIGHT_SHADER:
                    found = d3city.Material(found.flags, shaded_dds(found.dds, DARK_SHADE))

                clean_dds = found.dds

                # the clean tinted paint's alpha is gloss, with big clear areas on the Driv3r bike: the frame drew see-through
                if not lamp and shader != D3_GLASS_SHADER:
                    found = d3city.Material(found.flags, opaque_dds(found.dds))

                # a tinted car's damaged texture keeps the original alpha, which the paint mask then replaces
                damaged = paint and shaded_dds(clean_dds if tinted else found.dds, DAMAGE_SHADE)

                added = add_material(raw.pcmp, found, textures)
                # DPL tints by the damaged texture's alpha; the clean alpha (gloss) there left the body grey
                mask = d3_material(page, shared, index, D3_PAINT_MASK) if paint and tinted else None

                if mask:
                    damaged = with_alpha(damaged, mask.dds)

                if added is not None and paint and add_material(raw.pcmp, d3city.Material(found.flags, damaged), textures) is not None:
                    # only the damaged texture's slot stays, right after the clean one's: the paint substance draws both
                    for table in (raw.pcmp.materials, raw.pcmp.material_pointers, raw.pcmp.pointers, raw.pcmp.substances, raw.pcmp.substance_slots):
                        table.pop()

                    struct.pack_into('<I', raw.pcmp.substances[-1], 12, 2)

                if added is not None:       # add_material appended its substance last; DPL's own words replace its flags
                    struct.pack_into('<2I', raw.pcmp.substances[-1], 0, *substance)

            materials[index, wheel, lamp] = 0 if added is None else added      # limit: unknown textures draw the donor's paint

        return materials[index, wheel, lamp]

    def add_mesh(triangles: np.ndarray, word: int) -> np.ndarray:
        """Writes the mesh with its vertices in the current slot's run; returns the run indices it uses."""
        for i in np.unique(triangles).tolist():
            if i not in block:
                block[i] = len(order)
                order.append(i)

        triangles = np.array([block[i] for i in triangles.reshape(-1).tolist()]).reshape(triangles.shape)
        if REORDER_STRIPS:
            strip, ordered = strip_of_greedy(triangles)
        else:
            strip, ordered = strip_of(triangles), triangles

        check_strip(strip, ordered)
        m = bytearray(template_mesh)
        struct.pack_into('<I', m, 12, len(strip))
        struct.pack_into('<I', m, 48, word)
        raw.meshes.append(m)
        raw.mesh_indices.append(len(raw.indices) // 2)
        raw.indices += np.array(strip, dtype='<u2').tobytes()
        return triangles

    def add_group(group_template: bytes, matrix: list[float] | None, first_mesh: int) -> None:
        g = bytearray(group_template)

        if matrix is not None:
            struct.pack_into('<16f', g, 4, *matrix)

        struct.pack_into('<H', g, 0x44, len(raw.meshes) - first_mesh)
        raw.groups.append(g)
        raw.group_meshes.append(first_mesh if len(raw.meshes) > first_mesh else None)

    for c, record in enumerate(raw.containers):
        sources = mapping.get(c, [])
        struct.pack_into('<3f', record, 8, SCALE, SCALE, SCALE)
        struct.pack_into('<H', record, 0x18, 0)

        for slot in range(7):
            first_group = len(raw.groups)
            first_index, first_vertex = len(raw.indices) // 2, len(order)
            block.clear()

            # DPL keeps a part's body, glass and interior in one group per slot (its blended meshes sort per group):
            # several Driv3r containers in one donor container share the first one's group, or their glass flickers
            merge = len(sources) > 1
            slot_groups = [group for source in sources for group in page.containers[source][slot][1]]
            first_mesh = len(raw.meshes)
            pending: dict[int, list[np.ndarray]] = {}

            for source in sources:
                kind, groups = page.containers[source][slot]

                for group in groups:
                    if not merge:
                        first_mesh = len(raw.meshes)

                    for mesh in group.meshes:
                        triangles = page.mesh_triangles(mesh)
                        shadow = kind == SHADOW_KIND

                        found = d3_material(page, shared, mesh.material)

                        if not len(triangles) or not shadow and is_light(found) and not is_overlay(found):    # glow quads, beams
                            continue

                        word = SHADOW_WORD if shadow else own_page << 16 | material_of(mesh.material, c in wheels)

                        if merge:
                            pending.setdefault(word, []).append(triangles)
                            continue

                        written = add_mesh(triangles, word)

                        if c in wheels and not shadow:      # a shadow hull's tail is its own data, not a deformation
                            still.update(np.unique(written).tolist())

                    if not merge:
                        add_group(template_group, frames.get((c, slot), group.matrix), first_mesh)

            # one mesh per material, as DPL's own groups (at most 6 meshes in any DPL vehicle): with 8 the meshes of
            # the second Driv3r part (rear window, rear seat, inner panels) were not drawn
            for word, parts in pending.items():
                add_mesh(np.concatenate(parts), word)

            if merge and slot_groups:
                add_group(template_group, frames.get((c, slot), slot_groups[0].matrix), first_mesh)

            if slot == 0 and c in lamp_parts:
                first_mesh = len(raw.meshes)

                lamp_runs[c] = {int(i) for index, triangles in lamp_parts[c]
                                for i in add_mesh(triangles, own_page << 16 | material_of(index, lamp=True)).reshape(-1)}
                add_group(lamp_groups[c], None, first_mesh)

            runs.setdefault(c, []).append((first_vertex, len(order)))
            count = len(raw.groups) - first_group
            raw.lod_groups[c][slot] = first_group if count else None
            kinds = [page.containers[s][slot][0] for s in sources if page.containers[s][slot][1]]
            struct.pack_into('<I', record, 0xA4 + slot * LOD_RECORD + 4, count)
            # (strip indices << 16 | vertices - 2, drawn flag) as on DPL's own cars; their shadow hull slots keep zeros
            drawn = bool(count) and SHADOW_KIND not in kinds
            sizes = (len(raw.indices) // 2 - first_index) << 16 | (len(order) - first_vertex - 2) if drawn else 0
            struct.pack_into('<2I', record, 0xA4 + slot * LOD_RECORD + 8, sizes, int(drawn))

            if kinds:
                struct.pack_into('<I', record, 0xA4 + slot * LOD_RECORD + 16, KIND_TO_DPL.get(kinds[0], kinds[0]))

        if sources or c in lamp_parts:
            boxes = [car_boxes[s] for s in sources]
            boxes += [page.vertices[t.reshape(-1), 0:3] for _, t in lamp_parts.get(c, [])]
            corners = np.concatenate(boxes)
            lo, hi = corners.min(0), corners.max(0)

            for k, (x, y, z) in enumerate((x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])):
                struct.pack_into('<4f', record, 0x24 + k * 16, x, y, z, 1.0)

    if len(order) > 0xFFFF:
        raise SystemExit(f'{len(order)} vertices: more than 16-bit mesh indices reach')

    car = page.vertices[order].copy()

    for c, (source, target) in (placements or {}).items():
        turn = TURN_ROUND if source[0] * target[0] < 0 else np.ones(3)

        for start, stop in runs.get(c, []):
            part = car[start:stop]
            part[:, 0:3] = (part[:, 0:3] - source) * turn + target
            part[:, 3:6] *= turn        # normal
            part[:, 8:11] *= turn       # damage offset and damaged normal (zero on the wheel, set on its shadow hull)
            part[:, 11:14] *= turn

    for run in lamp_runs.values():      # lit lenses lie on the dark lens of the body: lift them off it
        at = sorted(run)
        car[at, 0:3] += car[at, 3:6] * LAMP_LIFT

    raw.streams = [dpl_vertices(car, flat_tail, sorted(still))]
    return len(raw.meshes), sum(1 for i in materials.values() if i)


def donor_parts(tree: d.ChunkTree, vehicle_id: int) -> dict[bytes, d.Chunk]:
    """The chunks (by tag) of the VEHC whose VO3D holds this vehicle id."""
    for top in tree.chunks:
        if top.tag == b'VEHC' and top.tree:
            parts = {c.tag: c for c in top.tree.chunks}

            if struct.unpack_from('<I', parts[b'VO3D'].data, 0)[0] == vehicle_id:
                return parts

    raise SystemExit(f'no DPL vehicle with id {vehicle_id:#x} (see dpl_vehicles.py list)')


def palette(bo3: dict[int, list[v.HandlingRecord]], data: bytes, upvh: d.Entry) -> dict[int, int]:
    """{handling key: colour} of the Driv3r car's model (its BO3 block; the model id is UPVH +0x14, also in a .vvv)."""
    model = struct.unpack_from('<I', data, upvh.offset + UPVH_MODEL)[0]
    return {r.key: r.raw for r in bo3.get(model, []) if r.key in PALETTE_KEYS}


def with_palette(vo3d: bytes, colours: dict[int, int], keys: range | set[int] = PALETTE_KEYS) -> bytes:
    """The VO3D with the donor's records of `keys` (its colours) replaced by these raw values (records stay sorted by
    key; bytes after them stay)."""
    records = v.read_dvo3(vo3d, VO3D_DVO3)
    end = VO3D_DVO3 + 12 + 8 * len(records)
    kept = [(r.key, r.marker, r.raw) for r in records if r.key not in keys]
    marker = records[0].marker
    # only keys the donor already sets: DPL's bikes read other keys (151-157 among them) as numbers, and a palette colour
    # there as a lean or steering value made the bike's physics blow up seconds after the player mounted
    present = {r.key for r in records}
    merged = sorted(kept + [(key, marker, raw) for key, raw in colours.items() if key in present])
    head = bytearray(vo3d[:VO3D_DVO3 + 12])
    struct.pack_into('<I', head, VO3D_DVO3 + 8, len(merged))
    return bytes(head) + b''.join(struct.pack('<HHI', *r) for r in merged) + vo3d[end:]


def convert(data: bytes, car: tuple[int, d.Entry, d.Entry], shared: d3city.Page | None, source: dict[bytes, d.Chunk],
            ssic_pack: bytes, flat_tail: bool = False, tinted: bool = False) -> tuple[bytes, bytes, str]:
    """The donor's (MDXN, UPVH) rebuilt with the Driv3r car (`data`, (type, MDPC, UPVH) from driv3r_cars), and a report.
    `source` = the donor's chunks from the original file, `ssic_pack` = the original SSIC MDXN; `tinted` as in rebuild."""
    _, pack, upvh = car
    page = driv3r_page(data, pack)
    car_boxes = container_boxes(data, pack.offset + struct.unpack_from('<I', data, pack.offset + 12)[0], len(page.containers), 0x188, 0x28)
    lights = light_containers(page, shared)
    # only containers a node draws take part: a shared pack holds the other cars of the file too, and a truck keeps a
    # second rim template no node draws (it went into a door and swung out with it)
    foreign = set(range(len(page.containers))) - drawn_containers(data, upvh)
    lights -= foreign
    donor_upvh = bytes(source[b'UPVH'].data)
    raw = d.split_pack(source[b'MDXN'].data, 0, len(source[b'MDXN'].data))
    lamps = lamp_containers(raw, d.split_pack(ssic_pack, 0, len(ssic_pack)))
    boxes = [container_boxes(c, 0, 1, d.CONTAINER_SIZE, 0x24)[0] for c in raw.containers]
    car_lods = [sum(1 for _, groups in lods if groups) for lods in page.containers]
    donor_lods = [sum(1 for g in lods if g is not None) for lods in raw.lod_groups]
    wheels = wheel_containers(donor_upvh, 0)
    car_wheels = wheel_containers(data, upvh.offset)
    # a wheel part stored at the origin (a truck's rim) is a template for the hubs: kept, it drew up in the load bed
    templates = {c for c in car_wheels if np.abs(car_boxes[c].mean(0)).max() < TEMPLATE_CENTRE}
    # a part several other nodes draw is placed by their transforms (a truck's marker lamp drawn by two lamp nodes):
    # in a static donor part it drew once at the car's origin
    users = collections.Counter(n.handle & 0x3FF for n in v.hierarchy_nodes(data, upvh.offset) if n.handle != v.NO_HANDLE)
    instanced = {c for c, n in users.items() if n >= 2} - set(car_wheels)
    donor_mdxn = bytes(source[b'MDXN'].data)
    wheel_map, placements = wheel_mapping({c: h for c, h in car_wheels.items() if c not in templates | foreign},
                                          {c: container_centre(page, c) for c in car_wheels}, hub_places(data, upvh.offset),
                                          wheels, {k: pack_centre(donor_mdxn, k) for k in wheels if k < len(boxes)},
                                          hub_places(donor_upvh, 0))
    placed_wheels = {c for cs in wheel_map.values() for c in cs}
    mapping = match_containers(car_boxes, boxes, car_lods, donor_lods, lights | foreign | templates | instanced | placed_wheels,
                               set(lamps) | set(wheel_map), part_classes(data, upvh.offset), part_classes(donor_upvh, 0),
                               movers(data, upvh.offset), movers(donor_upvh, 0))
    mapping.update(wheel_map)
    hierarchy, hubs, instances = move_hubs(donor_upvh, data, upvh.offset)
    hierarchy, located = move_locators(hierarchy, data, upvh.offset)
    seat_lift = 0.0

    if len(hub_places(donor_upvh, 0)) == BIKE_HUBS:
        car_points = np.concatenate([container_points(page, c) for c in drawn_containers(data, upvh) - lights])
        donor_points = np.concatenate([pack_points(donor_mdxn, c) for c in range(len(raw.containers))])
        hierarchy, seat_lift = lift_bike_seat(hierarchy, donor_upvh, car_points, donor_points)

    hierarchy, hinges = move_hinges(hierarchy, data, upvh.offset, mapping)
    hierarchy, collisions = move_collision(hierarchy, data, upvh.offset)
    hierarchy, doors = move_doors(hierarchy, donor_upvh)
    looks = donor_lamp_looks(bytes(source[b'MDXN'].data), ssic_pack, lamps)
    lamp_parts = lamp_triangles(page, shared, lights, looks, lamp_roles(data, upvh.offset), lamp_roles(donor_upvh, 0))
    meshes, converted = rebuild(raw, page, shared, car_boxes, mapping, set(wheels), lamp_parts, flat_tail, placements, tinted)
    placed = {c for cs in mapping.values() for c in cs}
    unplaced = sorted(c for c in drawn_containers(data, upvh) - lights - templates - instanced - placed
                      if c < len(car_boxes) and not is_template(car_boxes[c]))
    report = (f'{len(mapping)} of {len(boxes)} donor containers filled, {len(lights)} light containers left out, {meshes} meshes, '
              f'{converted} Driv3r materials as DPL materials, Driv3r parts without a donor part: {unplaced or "none"}\n'
              'donor container <- Driv3r containers: ' + ', '.join(f'{k} <- {cs}' for k, cs in sorted(mapping.items())) + '\n'
              f'{hubs} wheel hubs, {instances} wheel instance matrices, {hinges} hinges, {collisions} collision records and '
              f'{located} locator points (seats, exhausts) and {doors} door entry points moved '
              f'to the Driv3r car\'s{f", bike seat {seat_lift * 100:.1f} cm up" if seat_lift else ""}; lamp glow into donor lamps: '
              + ', '.join(f'{k} ({sum(len(t) for _, t in pieces)} triangles)' for k, pieces in sorted(lamp_parts.items())))
    return d.join_pack(raw), hierarchy, report


def quoted(line: bytes, value: bytes) -> bytes:
    """The line with its last quoted string replaced by value."""
    close = line.rindex(b'"')
    return line[:line.rindex(b'"', 0, close) + 1] + value + line[close:]


def set_level(game: Path, sp_name: str, key: bytes = LEVEL_KEY, folder: str = 'vehicles', city: str = CITY_LEVEL) -> None:
    """Points a key (SpooledVehicles, CharacterModels) of the Driv3r city's LEVEL block of levels.txt at
    <folder>/<sp_name>; keeps every other byte."""
    edit_level(game, key, lambda line: quoted(line, f'{folder}\\{sp_name}'.encode()), city)
    print(f'{city} LEVEL loads {folder}\\{sp_name}')


def level_blocks(lines: list[bytes]) -> list[tuple[int, int]]:
    """(LEVEL line, END line) of each LEVEL block of levels.txt."""
    return [(i, next(j for j in range(i, len(lines)) if lines[j].strip().upper().startswith(b'END')))
            for i, line in enumerate(lines) if line.lstrip().upper().startswith(b'LEVEL')]


def force_start(line: bytes) -> bool:
    return [word.upper() for word in line.split()] == [b'ENABLE', b'FORCESTART']


def city_level(lines: list[bytes], city: str = CITY_LEVEL) -> tuple[int, int]:
    """The Driv3r city's LEVEL block (LEVEL "Miami"). Made on first use, at the end, from the first LEVEL, which goes
    back to DPL's NYC Now: the game takes the first LEVEL as NOW and the second as THEN (story mode). Its ForceStart
    is dropped, so the game shows the title, where DPLExtended's D3 DLC entry starts the city. Another city's block is
    a copy of Miami's (Driv3r cars, Tanner, HUD) with its own d4c and map."""
    blocks = level_blocks(lines)
    name = f'"{city}"'.encode()

    for start, end in blocks:
        if lines[start].split(b',')[0].split(None, 1)[-1].strip() == name:
            return start, end

    if city != CITY_LEVEL:
        miami, miami_end = city_level(lines)
        block = [line for line in lines[miami:miami_end + 1] if not force_start(line)]
        block[0] = b'LEVEL ' + name + b',\t\t"Cities\\' + city.lower().encode() + b'_pc.d4c"'
        block = [quoted(line, b'Overlays\\' + city.lower().encode() + b'.map') if line.split()[:1] == [b'Map'] else line
                 for line in block]
        last_end = level_blocks(lines)[-1][1]
        lines[:] = lines[:last_end + 1] + [b''] + block + lines[last_end + 1:]
        print(f'levels.txt: LEVEL {name.decode()} added at the end, a copy of {CITY_LEVEL}\'s')
        return city_level(lines, city)

    first, first_end = blocks[0]
    last_end = blocks[-1][1]
    block = [line for line in lines[first:first_end + 1] if not force_start(line)]
    block[0] = b'LEVEL ' + name + b',\t\t"' + CITY_FILE + b'"'

    if not any(line.lstrip().startswith(b'StartPosition') for line in block):
        block.insert(1, b'\tStartPosition\t\t0, 0, 0')

    nyc_now = [quoted(lines[first], NYC_NOW_FILE)]

    for line in lines[first + 1:first_end + 1]:
        key = line.split()[0] if line.split() else b''

        if force_start(line) or key == b'StartPosition':
            continue

        nyc_now.append(quoted(line, NYC_NOW_VALUES[key]) if key in NYC_NOW_VALUES else line)

    lines[:] = lines[:first] + nyc_now + lines[first_end + 1:last_end + 1] + [b''] + block + lines[last_end + 1:]
    print(f'levels.txt: LEVEL {name.decode()} added at the end, the first LEVEL is NYC Now again')
    return city_level(lines)


def edit_level(game: Path, key: bytes, edit: Callable[[bytes], bytes], city: str = CITY_LEVEL) -> None:
    """Replaces the key's line of the Driv3r city's LEVEL block of levels.txt by edit(line); every other byte stays.
    levels.txt is backed up to <game>/_backup/<date>/ before the first change of the day."""
    levels = game / 'Cities' / 'levels.txt'
    backup = game / '_backup' / datetime.date.today().isoformat() / 'levels.txt'

    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(levels, backup)
        print(f'backup {backup}')

    lines = levels.read_bytes().split(b'\n')
    start, end = city_level(lines, city)
    row = next((i for i in range(start, end) if lines[i].lstrip().startswith(key)), None)

    if row is None:
        raise SystemExit(f'levels.txt: the {city} LEVEL block has no {key.decode()} line')

    lines[row] = edit(lines[row])
    levels.write_bytes(b'\n'.join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('dpl', type=Path)
    ap.add_argument('driv3r', type=Path)
    ap.add_argument('--city', default='miami')
    ap.add_argument('--file', help='Driv3r Vehicles file instead of <city>.vvs, e.g. miami.vvv (special cars, Tanner\'s car)')
    ap.add_argument('--list', action='store_true', help='list the Driv3r cars of the city')
    ap.add_argument('--donor', type=lambda s: int(s, 0), help='DPL vehicle id to replace (DPLHook Vehicle tab order)')
    ap.add_argument('--car', type=int, help='Driv3r car index from --list')
    ap.add_argument('--fresh', action='store_true', help='copy nyc_vehicles.sp again first (undoes earlier swaps)')
    ap.add_argument('--level', action='store_true', help='let the first LEVEL load the copy')
    ap.add_argument('--restore-level', action='store_true', help='let the first LEVEL load nyc_vehicles.sp again')
    ap.add_argument('--flat-tail', action='store_true', help='zero the vertex tail offsets (no deformation data)')
    args = ap.parse_args()
    vehicles_dir = args.dpl / 'Vehicles'

    if not (vehicles_dir / ORIGINAL).is_file() or not (args.dpl / 'Cities' / 'levels.txt').is_file():
        raise SystemExit(f'{args.dpl}: no Vehicles/{ORIGINAL} or Cities/levels.txt, pass your Driver Parallel Lines folder')

    if args.restore_level:
        set_level(args.dpl, 'NYC_vehicles.sp')
        return

    data, cars = driv3r_cars(args.driv3r, args.city, args.file)

    if args.list:
        for i, (car_type, pack, upvh) in enumerate(cars):
            page = driv3r_page(data, pack)
            count = len(drawn_containers(data, upvh)) if car_type < 0 else len(page.containers)
            print(f'{i:2d} type {car_type:3d} containers {count:2d} vertices {len(page.vertices):5d} '
                  f'hierarchy nodes {len(v.hierarchy_nodes(data, upvh.offset))}')
        return

    if args.donor is None or args.car is None or not 0 <= args.car < len(cars):
        raise SystemExit(f'pass --donor <DPL id> and --car 0..{len(cars) - 1} (see --list)')

    copy = vehicles_dir / COPY

    if args.fresh or not copy.exists():
        print(f'copying {ORIGINAL} -> {COPY} ...')
        shutil.copyfile(vehicles_dir / ORIGINAL, copy)

    car_type = cars[args.car][0]
    shared = driv3r_globals(args.driv3r, args.city)
    new = copy.with_suffix('.sp.new')

    # the donor always comes from the original file, so a second run replaces an earlier swap instead of building on it
    with (vehicles_dir / ORIGINAL).open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as original:
        tree = d.read_tree(original, 0, 2)
        ssic = next(c for c in tree.chunks if c.tag == b'SSIC')
        ssic_pack = bytes(next(c for c in ssic.tree.chunks if c.tag == b'MDXN').data)
        pack, hierarchy, report = convert(data, cars[args.car], shared, donor_parts(tree, args.donor), ssic_pack, args.flat_tail)
        del tree, ssic

    with copy.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as sp:
        tree = d.read_tree(sp, 0, 2)
        parts = donor_parts(tree, args.donor)
        parts[b'MDXN'].data, parts[b'UPVH'].data = pack, hierarchy

        with new.open('wb') as out:
            for piece in d.tree_pieces(tree):
                out.write(piece)

        del tree, parts     # release the views into the old file before it is replaced

    print(f'Driv3r {args.city} car {args.car} (type {car_type}) -> DPL id {args.donor:#x}: {report}')

    try:
        os.replace(new, copy)
    except PermissionError as error:
        raise SystemExit(f'{copy} is in use: close the game, then run this again (the new file waits as {new.name})') from error

    if args.level:
        set_level(args.dpl, COPY)


if __name__ == '__main__':
    main()
