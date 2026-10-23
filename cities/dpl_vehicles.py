"""Reads Driver Parallel Lines' vehicle spool file (Vehicles/nyc_vehicles.sp). Never writes it.

usage: dpl_vehicles.py <sp> list
       dpl_vehicles.py <sp> export <entry> <out.glb> [--lod N] [--tex-size 256]
       dpl_vehicles.py <sp> nodes <entry>       hierarchy (UPVH) nodes and the container each one draws
       dpl_vehicles.py <sp> handling <entry>    handling keys (VO3D)

File: CHNK of SSIC (SSLP lookup, shared texture-only MDXN, VSCT type list), one VEHC per vehicle (VO3D handling,
UPVH hierarchy, MDXN model, SSBK sound bank) and BINF build text.
- SSLP: header 0x18 bytes, then 6-byte records (u16 vehicle id, u16 VEHC entry index, u16 0).
- VSCT: 'VSCT', u16 1, u16 count, count x 22 bytes (4 flag bytes, 16-char name, 2 bytes). Index = vehicle id low 12 bits.
- Model pack page = 0x2001 + 124 * type + variant.
Export: containers keep their stored positions (vehicle space); one wheel is stored, the others come from the hierarchy.
"""
import argparse
import mmap
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dplchunk as d
import vehchunk as v
from dpl_export import Textures, container_mesh
from glb import GlbBuilder

VSCT_RECORD = 22
HANDLE_CONTAINER_MASK = 0x3FF


def vehicle_names(data: mmap.mmap, ssic: d.Entry) -> list[str]:
    for entry in d.entries(data, ssic.offset):
        if entry.tag == 'VSCT':
            count = struct.unpack_from('<H', data, entry.offset + 6)[0]
            return [bytes(data[entry.offset + 8 + i * VSCT_RECORD + 4:entry.offset + 8 + i * VSCT_RECORD + 20]).split(b'\0')[0].decode('latin1')
                    for i in range(count)]

    return []


def vehicles(data: mmap.mmap) -> tuple[list[str], list[dict[str, d.Entry]]]:
    """(type names, per VEHC its sub-entries by tag) in file order."""
    top = d.entries(data, 0)
    names = vehicle_names(data, next(e for e in top if e.tag == 'SSIC'))
    return names, [{e.tag: e for e in d.entries(data, t.offset)} for t in top if t.tag == 'VEHC']


def vehicle_label(data: mmap.mmap, names: list[str], parts: dict[str, d.Entry]) -> str:
    vid = struct.unpack_from('<I', data, parts['VO3D'].offset)[0]
    base, variant = vid & 0xFFF, vid >> 12
    return f'{names[base] if base < len(names) else "?"}{f" v{variant}" if variant else ""} (id {vid:#x})'


def list_vehicles(data: mmap.mmap, names: list[str], found: list[dict[str, d.Entry]]) -> None:
    for i, parts in enumerate(found):
        pack = d.load_pack(data, parts['MDXN'].offset)
        nodes = len(v.hierarchy_nodes(data, parts['UPVH'].offset))
        print(f'{i:3d} {vehicle_label(data, names, parts):34s} page {pack.header[1]:5d} containers {len(pack.containers):2d} '
              f'hierarchy nodes {nodes:2d} model {parts["MDXN"].size / 1e6:.1f} MB')


def export_vehicle(data: mmap.mmap, parts: dict[str, d.Entry], lod: int | None, tex_size: int, out: Path) -> int:
    """Writes the glb; returns the number of containers placed."""
    pack = d.load_pack(data, parts['MDXN'].offset)
    glb = GlbBuilder()
    textures = Textures(glb, data, pack, tex_size)
    placed = 0

    for i, container in enumerate(pack.containers):
        used = [k for k, (_, groups) in enumerate(container.lods) if groups]
        chosen = lod if lod is not None else (used[0] if used else 0)
        kind, groups = container.lods[chosen]
        # container_mesh draws LOD entry 0: hand it a copy with the chosen entry first
        mesh = container_mesh(glb, data, pack, textures, d.Container(container.scale, container.stream, container.box, [(kind, groups)]))

        if mesh is None:
            continue

        glb.add_node(f'c{i}', mesh, None, {'lod': chosen, 'kind': kind, 'used_lods': [[k, container.lods[k][0]] for k in used]})
        placed += 1

    glb.save(out)
    return placed


def print_nodes(data: mmap.mmap, parts: dict[str, d.Entry]) -> None:
    page = struct.unpack_from('<I', data, parts['MDXN'].offset + 4)[0]
    print(f'pack page {page} (handle page = page & 63 = {page & 63})')

    for i, node in enumerate(v.hierarchy_nodes(data, parts['UPVH'].offset)):
        drawn = '-' if node.handle == v.NO_HANDLE else f'page {node.handle >> 10} container {node.handle & HANDLE_CONTAINER_MASK}'
        print(f'{i:3d} type {node.node_type} part {node.part:3d} {drawn}')


def print_handling(data: mmap.mmap, parts: dict[str, d.Entry]) -> None:
    for record in v.read_dvo3(data, parts['VO3D'].offset + 4):
        value = f'{record.value:.6g}' if record.marker == v.FLOAT_MARKER else f'{record.raw:#x} (marker {record.marker:#06x})'
        print(f'{record.key:3d} {value}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('sp', type=Path)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    exp = sub.add_parser('export')
    exp.add_argument('entry', type=int)
    exp.add_argument('out', type=Path)
    exp.add_argument('--lod', type=int, help='LOD entry 0-6 (default: first used entry per container)')
    exp.add_argument('--tex-size', type=int, default=256)

    for name in ('nodes', 'handling'):
        sub.add_parser(name).add_argument('entry', type=int)

    args = ap.parse_args()

    if not args.sp.is_file():
        raise SystemExit(f'{args.sp} not found: pass the path of Vehicles/nyc_vehicles.sp in your Driver Parallel Lines folder')

    with args.sp.open('rb') as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        names, found = vehicles(data)

        if args.command == 'list':
            list_vehicles(data, names, found)
            return

        if not 0 <= args.entry < len(found):
            raise SystemExit(f'entry {args.entry}: the file has entries 0..{len(found) - 1} (see "list")')

        parts = found[args.entry]
        print(vehicle_label(data, names, parts))

        if args.command == 'export':
            placed = export_vehicle(data, parts, args.lod, args.tex_size, args.out)
            print(f'containers {placed} -> {args.out}')
        elif args.command == 'nodes':
            print_nodes(data, parts)
        else:
            print_handling(data, parts)


if __name__ == '__main__':
    main()
