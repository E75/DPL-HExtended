"""Driv3r's climbable ladders as a list for the hook's ladders (bDriv3rMoves).

    d3_ladders.py <Driv3r folder> <city> <out ladders.txt>

Driv3r: the `PINS` chunk of a city's .d3c names the Global page's models in handle order (entry k = handle k, the
artist's source file, Z:\\...\\<NAME>.X). Ladders are the models whose name holds LADDER (Miami: jetty, marina, small
a jetty. Each placed instance is baked to world space (d3city.bake_instance): its lowest and highest vertex give the
bottom, the horizontal axis its vertices spread least along gives the ladder's normal. Where the handrails bend over
them; a flat ladder (Miami's jetty ladders) ends at the deck, its normal either way round (the hook climbs from the
swimmer's side).

Writes one ladder per line: x z bottom top nx nz model, metres in Driv3r's (= the converted city's) space.
"""
import argparse
import re
from pathlib import Path

import numpy as np

import d3city
import dplchunk as d

LADDER = b'LADDER'
QUATERNION_AS_STORED = set()
TOP_BAND = 0.3                          # metres under the highest vertex: the handrails, which bend over the deck
MIN_LEAN = 0.05                         # metres the handrails must lean over the deck; a flat ladder ends at the deck
OVER_DECK = 0.1                         # metres past the ladder plane from which a vertex is a handrail over the deck
RAIL_TO_DECK = 0.3
                                        # (rails from 1.19, he stood at 1.5); upgrade: the deck's own collision


def model_names(d3c: Path) -> list[str]:
    """The Global page's model names in handle order, from the PINS chunk."""
    def find(tree: d.ChunkTree) -> bytes | None:
        for c in tree.chunks:
            if c.tag == b'PINS':
                return bytes(c.data)

            found = find(c.tree) if c.tree else None

            if found:
                return found

        return None

    pins = find(d.read_tree(d3c.read_bytes(), 0, 3)) or b''
    return [m.decode('latin-1').rsplit('\\', 1)[-1][:-2] for m in re.findall(rb'[A-Za-z]:\\[^\x00]+?\.[Xx]', pins)]


def ladders(driv3r: Path, city: str) -> list[tuple[float, float, float, float, float, float, str]]:
    cities = driv3r / 'Cities'
    d3c = cities / f'{city}_DAY_PC.d3c'
    names = model_names(d3c)
    wanted = {k: n for k, n in enumerate(names) if LADDER in n.upper().encode()}
    page = d3city.read_pages(cities / f'{city}_DAY_PC' / 'SuperRegions.pcs')[d3city.GLOBAL_PAGE]
    result = []

    for instance in d3city.read_instances(d3c, city.lower() in QUATERNION_AS_STORED):
        if instance.handle >> 10 != d3city.GLOBAL_PAGE or instance.handle & 1023 not in wanted:
            continue

        baked = d3city.bake_instance(page, instance.handle & 1023, instance)

        if baked is None:
            continue

        v = baked.vertices[:, 0:3]
        flat = v[:, [0, 2]] - v[:, [0, 2]].mean(axis=0)
        _, axes = np.linalg.eigh(flat.T @ flat)
        normal = axes[:, 0]
        x, z = v[:, [0, 2]].mean(axis=0)
        lean = v[v[:, 1] > v[:, 1].max() - TOP_BAND][:, [0, 2]].mean(axis=0) - (x, z)
        over = (v[:, [0, 2]] - (x, z)) @ normal * np.sign(lean @ normal) > OVER_DECK
        deck = float(v[over, 1].min()) + RAIL_TO_DECK if abs(lean @ normal) > MIN_LEAN and over.any() else float(v[:, 1].max())

        if lean @ normal > 0.0:
            normal = -normal
        result.append((float(x), float(z), float(v[:, 1].min()), deck, float(normal[0]),
                       float(normal[1]), wanted[instance.handle & 1023]))

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('driv3r', type=Path)
    ap.add_argument('city')
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    found = ladders(args.driv3r, args.city)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(f'{x:.2f} {z:.2f} {b:.2f} {t:.2f} {nx:.3f} {nz:.3f} {n.replace(" ", "_")}\n'
                                for x, z, b, t, nx, nz, n in found), encoding='utf-8')
    print(f'{args.city}: {len(found)} ladders -> {args.out}')


if __name__ == '__main__':
    main()
