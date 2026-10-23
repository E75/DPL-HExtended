"""Minimal binary glTF 2.0 writer (no dependencies): meshes with POSITION / NORMAL / TEXCOORD_0 / indices."""
import json
import struct
from pathlib import Path


class GlbBuilder:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []
        self.images = []
        self.textures = []
        self.root_children = []

    def _view(self, payload: bytes, target: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)

        view = {'buffer': 0, 'byteOffset': len(self.blob), 'byteLength': len(payload)}

        if target:
            view['target'] = target

        self.blob += payload
        self.views.append(view)
        return len(self.views) - 1

    def _accessor(self, view: int, component: int, count: int, kind: str, lo: list[float] | None = None, hi: list[float] | None = None) -> int:
        accessor = {'bufferView': view, 'componentType': component, 'count': count, 'type': kind}

        if lo is not None:
            accessor['min'], accessor['max'] = lo, hi

        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_image(self, png_bytes: bytes) -> int:
        self.images.append({'bufferView': self._view(png_bytes), 'mimeType': 'image/png'})
        self.textures.append({'source': len(self.images) - 1})
        return len(self.textures) - 1

    def add_material(self, name: str, texture: int | None = None) -> int:
        material = {'name': name, 'doubleSided': True, 'pbrMetallicRoughness': {'metallicFactor': 0.0, 'roughnessFactor': 0.9}}

        if texture is not None:
            material['pbrMetallicRoughness']['baseColorTexture'] = {'index': texture}
        else:
            material['pbrMetallicRoughness']['baseColorFactor'] = [0.6, 0.6, 0.6, 1.0]

        self.materials.append(material)
        return len(self.materials) - 1

    def add_mesh(self, name: str, primitives: list[dict]) -> int:
        """primitives: list of dict(positions, normals, uvs, indices, material)"""
        prims = []

        for p in primitives:
            if hasattr(p['positions'], 'tobytes'):      # numpy arrays (float32 Nx3 / Nx2, uint32 indices): the city exporter
                pos, nor, uvs, idx = p['positions'].tobytes(), p['normals'].tobytes(), p['uvs'].tobytes(), p['indices'].tobytes()
                n, lo, hi = len(p['positions']), p['positions'].min(axis=0).tolist(), p['positions'].max(axis=0).tolist()
                idx_count = len(p['indices'])
            else:
                pos = b''.join(struct.pack('<3f', *v) for v in p['positions'])
                nor = b''.join(struct.pack('<3f', *v) for v in p['normals'])
                uvs = b''.join(struct.pack('<2f', *v) for v in p['uvs'])
                idx = struct.pack('<%dI' % len(p['indices']), *p['indices'])
                n, idx_count = len(p['positions']), len(p['indices'])
                xs, ys, zs = zip(*p['positions'])
                lo, hi = [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]

            a_pos = self._accessor(self._view(pos, 34962), 5126, n, 'VEC3', lo, hi)
            a_nor = self._accessor(self._view(nor, 34962), 5126, n, 'VEC3')
            a_uv = self._accessor(self._view(uvs, 34962), 5126, n, 'VEC2')
            a_idx = self._accessor(self._view(idx, 34963), 5125, idx_count, 'SCALAR')
            prim = {'attributes': {'POSITION': a_pos, 'NORMAL': a_nor, 'TEXCOORD_0': a_uv}, 'indices': a_idx, 'mode': 4}

            if p.get('material') is not None:
                prim['material'] = p['material']

            prims.append(prim)

        self.meshes.append({'name': name, 'primitives': prims})
        return len(self.meshes) - 1

    def add_node(self, name: str, mesh: int | None = None, matrix: list[float] | None = None, extras: dict | None = None) -> int:
        node = {'name': name}

        if mesh is not None:
            node['mesh'] = mesh

        if matrix is not None:
            node['matrix'] = list(matrix)

        if extras:
            node['extras'] = extras

        self.nodes.append(node)
        self.root_children.append(len(self.nodes) - 1)
        return len(self.nodes) - 1

    def add_instanced_node(self, name: str, mesh: int, translations: list[tuple[float, float, float]], rotations: list[tuple[float, float, float, float]], extras: dict | None = None) -> int:
        """One node drawn many times (EXT_mesh_gpu_instancing). translations: (x, y, z) list, rotations: quaternions (x, y, z, w)."""
        t = self._accessor(self._view(b''.join(struct.pack('<3f', *v) for v in translations)), 5126, len(translations), 'VEC3')
        r = self._accessor(self._view(b''.join(struct.pack('<4f', *q) for q in rotations)), 5126, len(rotations), 'VEC4')
        node = {'name': name, 'mesh': mesh, 'extensions': {'EXT_mesh_gpu_instancing': {'attributes': {'TRANSLATION': t, 'ROTATION': r}}}}

        if extras:
            node['extras'] = extras

        self.nodes.append(node)
        self.root_children.append(len(self.nodes) - 1)
        self.instancing = True
        return len(self.nodes) - 1

    def save(self, path: str | Path) -> None:
        nodes = self.nodes + [{'name': 'root', 'children': self.root_children}]
        doc = {'asset': {'version': '2.0', 'generator': 'driv3r-threejs'}, 'scene': 0, 'scenes': [{'nodes': [len(nodes) - 1]}],
               'nodes': nodes, 'meshes': self.meshes, 'buffers': [{'byteLength': len(self.blob)}], 'bufferViews': self.views, 'accessors': self.accessors}

        if getattr(self, 'instancing', False):
            doc['extensionsUsed'] = ['EXT_mesh_gpu_instancing']

        for key, value in (('materials', self.materials), ('images', self.images), ('textures', self.textures)):
            if value:
                doc[key] = value

        text = json.dumps(doc, separators=(',', ':')).encode()
        text += b' ' * (-len(text) % 4)
        blob = bytes(self.blob) + b'\0' * (-len(self.blob) % 4)
        total = 12 + 8 + len(text) + 8 + len(blob)

        with open(path, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, total))
            f.write(struct.pack('<II', len(text), 0x4E4F534A) + text)
            f.write(struct.pack('<II', len(blob), 0x004E4942) + blob)
