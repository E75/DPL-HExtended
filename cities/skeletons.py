"""Skeletons (UPCB chunks) of either game and how their bones line up. Read only.

usage: skeletons.py <file> [<file> ...]
  Driv3r: Missions/missionNN.dam (character models + skeletons); DPL: anims/nyc.an4 (skeletons + animations).

UPCB: u32 skeleton count, ...; each skeleton starts with its 32-char name ('male_skeleton'), bone records are 176 bytes
starting with a 48-char name ('bone_pelvis'), then two 4x4 f32 matrices (row vectors, translation in the last row): the
bone relative to its parent and that matrix inverted. The records hold no parent: the hierarchy follows the names
(PARENTS). Both games share the bind pose except the arms (Driv3r level, DPL hanging) and the extra bones. Characters: Driv3r .dam = MDPC + UPXD + UPCB; DPL
Characters/nyc.sp = SSIC (SSLP, shared MDXN, CSCT id list) + one CHAC (UPXD, MDXN) per character. Skinned vertices
are 64 bytes in both.
"""
import argparse
import mmap
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vehchunk as v

SKELETON_NAME = re.compile(rb'[a-z]+_skeleton')
BONE_NAME = re.compile(rb'bone_[a-z_]+')
LOCAL_MATRIX = 48           # offset of the bone-to-parent matrix in a bone record
CHEST = 'upper chest'       # PARENTS: bone_upperchest where the skeleton has one (Driv3r), else bone_back (DPL)
PARENTS = {'left_upperleg': 'pelvis', 'left_lowerleg': 'left_upperleg', 'left_foot': 'left_lowerleg', 'left_toe': 'left_foot',
           'right_upperleg': 'pelvis', 'right_lowerleg': 'right_upperleg', 'right_foot': 'right_lowerleg', 'right_toe': 'right_foot',
           'back': 'pelvis', 'upperchest': 'back', 'left_clavicle': CHEST, 'right_clavicle': CHEST, 'head': CHEST, 'hair': 'head',
           'left_upperarm': 'left_clavicle', 'left_lowerarm': 'left_upperarm', 'left_wrist': 'left_lowerarm', 'left_hand': 'left_wrist',
           'left_fingers': 'left_hand', 'right_upperarm': 'right_clavicle', 'right_lowerarm': 'right_upperarm',
           'right_wrist': 'right_lowerarm', 'right_hand': 'right_wrist', 'gun': 'right_hand'}


def skeleton_blocks(path: Path) -> dict[str, bytes]:
    """{skeleton name: its bytes up to the next skeleton} of the first UPCB chunk of a file."""
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        chunks = v.find_all(data, 'UPCB')

        if not chunks:
            return {}

        block = bytes(data[chunks[0].offset:chunks[0].offset + chunks[0].size])

    starts = [(m.start(), m.group().decode()) for m in SKELETON_NAME.finditer(block)]
    return {name: block[start:starts[n + 1][0] if n + 1 < len(starts) else len(block)] for n, (start, name) in enumerate(starts)}


def skeletons(path: Path) -> dict[str, list[str]]:
    """{skeleton name: bones in order} of the first UPCB chunk of a file."""
    return {name: [m.group().decode() for m in BONE_NAME.finditer(block)] for name, block in skeleton_blocks(path).items()}


def bind_pose(path: Path, skeleton: str = 'male_skeleton') -> dict[str, np.ndarray]:
    """{bone name: 4x4 bone -> model space matrix of the bind pose (row vectors)} of one skeleton of the file."""
    block = skeleton_blocks(path)[skeleton]
    local = {m.group().decode()[5:]: np.array(struct.unpack_from('<16f', block, m.start() + LOCAL_MATRIX), dtype=np.float64).reshape(4, 4)
             for m in BONE_NAME.finditer(block)}
    chest = 'upperchest' if 'upperchest' in local else 'back'
    model: dict[str, np.ndarray] = {}

    def place(bone: str) -> np.ndarray:
        if bone not in model:
            parent = PARENTS.get(bone)
            parent = chest if parent == CHEST else parent
            model[bone] = local[bone] if parent is None else local[bone] @ place(parent)

        return model[bone]

    return {f'bone_{bone}': place(bone) for bone in local}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('files', type=Path, nargs='+')
    args = ap.parse_args()
    found = {}

    for path in args.files:
        found_here = skeletons(path)

        if not found_here:
            print(f'{path.name}: no UPCB skeleton (Driv3r: Missions/*.dam, DPL: anims/*.an4)')

        for name, bones in found_here.items():
            label = f'{path.name}:{name}'
            found[label] = bones
            print(f'{label}: {len(bones)} bones: {" ".join(b[5:] for b in bones)}')

    labels = list(found)

    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            x, y = found[labels[a]], found[labels[b]]
            print(f'{labels[a]} vs {labels[b]}: shared {len(set(x) & set(y))}, only first {sorted(set(x) - set(y))}, '
                  f'only second {sorted(set(y) - set(x))}')


if __name__ == '__main__':
    main()
