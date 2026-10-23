"""Exports the containers of one Driver Parallel Lines model pack (MDXN in Cities/*.d4c) to a binary glTF.

usage: dpl_export.py <d4c> list
       dpl_export.py <d4c> <pack number> <out.glb> [--tex-size 256]

Instances (GEBI v10) are not decoded yet, so the containers are laid out in a grid by their box size (LOD entry 0 only).
Node names are c<container index>; extras hold the stream and the box for checks.
"""
import argparse
import collections
import io
import mmap
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import dplchunk as d
from glb import GlbBuilder

FLOATS_PER_VERTEX = d.VERTEX_STRIDE // 4
GRID_ROW_WIDTH = 400.0      # metres of containers per gallery row
GRID_GAP = 10.0


def strip_triangles(raw: np.ndarray) -> np.ndarray:
    """Triangles (Mx3) of a triangle strip; degenerate stitching triangles dropped, odd ones flipped to keep the winding."""
    a, b, c = raw[:-2], raw[1:-1], raw[2:]
    odd = (np.arange(len(a)) % 2) == 1
    tri = np.stack([np.where(odd, b, a), np.where(odd, a, b), c], axis=1)
    return tri[(a != b) & (b != c) & (a != c)]


def merged_primitives(per_material: dict[int, list[tuple[np.ndarray, np.ndarray]]], material_index: Callable[[int], int]) -> tuple[list[dict], int]:
    """glTF primitives, one per material, from {material: [(vertices Nx8, triangles Mx3 into them)]}. Returns (primitives, triangle count)."""
    prims = []
    triangles = 0

    for material, parts in per_material.items():
        offsets = np.cumsum([0] + [len(v) for v, _ in parts])[:-1]
        v = np.concatenate([v for v, _ in parts])
        tri = np.concatenate([t + o for (_, t), o in zip(parts, offsets)])
        prims.append({'positions': np.ascontiguousarray(v[:, 0:3], dtype='<f4'), 'normals': np.ascontiguousarray(v[:, 3:6], dtype='<f4'),
                      'uvs': np.ascontiguousarray(v[:, 6:8], dtype='<f4'), 'indices': tri.reshape(-1).astype('<u4'),
                      'material': material_index(material)})
        triangles += len(tri)

    return prims, triangles


def packs(data: mmap.mmap) -> list[tuple[str, int, int]]:
    """(chunk tag, absolute MDXN offset, size) for every model pack, in file order."""
    found = []

    for top in d.entries(data, 0):
        for entry in d.entries(data, top.offset):
            if entry.tag == 'MDXN':
                found.append((top.tag, entry.offset, entry.size))

    return found


class Textures:
    def __init__(self, glb: GlbBuilder, data: mmap.mmap, pack: d.Pack, size: int) -> None:
        self.glb, self.data, self.pack, self.size = glb, data, pack, size
        self.of: dict[int, int] = {}

    def material(self, material: int) -> int:
        if material in self.of:
            return self.of[material]

        dds = d.material_texture(self.data, self.pack, material)
        texture, alpha = None, False

        if dds is not None:
            try:
                image = Image.open(io.BytesIO(dds)).convert('RGBA')
                alpha = image.getchannel('A').getextrema()[0] < 250

                if self.size and max(image.size) > self.size:
                    scale = self.size / max(image.size)
                    image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.BILINEAR)

                out = io.BytesIO()
                image.save(out, 'PNG')
                texture = self.glb.add_image(out.getvalue())
            except (OSError, ValueError, NotImplementedError) as error:     # an unknown format stays grey
                print(f'material {material} not decoded: {error}')

        index = self.glb.add_material(f'm{material}', texture)

        if alpha:
            self.glb.materials[index]['alphaMode'] = 'MASK'
            self.glb.materials[index]['alphaCutoff'] = 0.5

        self.of[material] = index
        return index


def container_mesh(glb: GlbBuilder, data: mmap.mmap, pack: d.Pack, textures: Textures, container: d.Container) -> int | None:
    """One glTF mesh with the LOD 0 meshes of a container merged per material, or None."""
    count, offset = pack.streams[container.stream]
    vertices = np.frombuffer(data, dtype='<f4', count=count * FLOATS_PER_VERTEX, offset=offset).reshape(-1, FLOATS_PER_VERTEX)
    scale = np.array(container.scale, dtype=np.float64)
    per_material = collections.defaultdict(list)

    for group in container.lods[0][1]:
        matrix = np.array(group.matrix, dtype=np.float64).reshape(4, 4)
        identity = np.allclose(matrix, np.eye(4), atol=1e-4)

        for mesh in group.meshes:
            raw = np.frombuffer(data, dtype='<u2', count=mesh.index_count, offset=pack.indices_offset + mesh.start_index * 2).astype(np.int64)
            tri = strip_triangles(raw) if mesh.prim_type == d.STRIP else raw[:len(raw) // 3 * 3].reshape(-1, 3)
            tri = tri[(tri < count).all(axis=1)]

            if not len(tri):
                continue

            used, inverse = np.unique(tri.reshape(-1), return_inverse=True)
            v = vertices[used, :8].astype(np.float64)
            v[:, 0:3] *= scale

            if not identity:
                v[:, 0:3] = v[:, 0:3] @ matrix[:3, :3].T + matrix[:3, 3]
                v[:, 3:6] = v[:, 3:6] @ matrix[:3, :3].T

            per_material[mesh.material].append((v, inverse.reshape(-1, 3)))

    prims, _ = merged_primitives(per_material, textures.material)
    return glb.add_mesh('c', prims) if prims else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('d4c', type=Path)
    ap.add_argument('pack')
    ap.add_argument('out', nargs='?', type=Path)
    ap.add_argument('--tex-size', type=int, default=256)
    args = ap.parse_args()

    with args.d4c.open('rb') as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        found = packs(data)

        if args.pack == 'list':
            for i, (tag, offset, size) in enumerate(found):
                print(f'{i} {tag} {offset:#x} {size / 1e6:.1f} MB')
            return

        pack = d.load_pack(data, found[int(args.pack)][1])
        glb = GlbBuilder()
        textures = Textures(glb, data, pack, args.tex_size)
        x = z = row_depth = 0.0
        placed = 0

        for i, container in enumerate(pack.containers):
            mesh = container_mesh(glb, data, pack, textures, container)

            if mesh is None:
                continue

            lo = np.min(container.box, axis=0)
            hi = np.max(container.box, axis=0)

            if x > GRID_ROW_WIDTH:
                x, z, row_depth = 0.0, z + row_depth + GRID_GAP, 0.0

            matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x - lo[0], 0, z - lo[2], 1]
            glb.add_node(f'c{i}', mesh, matrix, {'stream': container.stream, 'box': [lo.tolist(), hi.tolist()]})
            x += hi[0] - lo[0] + GRID_GAP
            row_depth = max(row_depth, hi[2] - lo[2])
            placed += 1

        glb.save(args.out)
        print(f'pack {args.pack} containers {placed} of {len(pack.containers)} materials {len(textures.of)} -> {args.out}')


if __name__ == '__main__':
    main()
