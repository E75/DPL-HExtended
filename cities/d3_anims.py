"""Driv3r character clips (Anims/animation_<city>.ab3) added to a copy of DPL's animation bank (anims/nyc.an4).

usage: d3_anims.py check <animation_<city>.ab3> <nyc.an4> [--clip male_swim]
       d3_anims.py roundtrip <nyc.an4>          re-encodes every DPL rotation key: the encoder must give the same bits
       d3_anims.py build <animation_<city>.ab3> <missionNN.dam> <nyc.an4> <out.an4> [--clips name ...]

Both games use the same Reflections AnimLib bank: a 16-byte header (char[2] context, u16 entry id, u32 clip count, u32
version, u32 string block) and the clips back to back. The versions differ in the clip and joint records:
- Driv3r v8 ("HPC1", names in the string block): clip = u32 flags, f32 duration, f32 frame rate, u32, u32 joints,
  f32 blend in, f32 blend out, f32 dyn, f32 max, u32 name offset; joint = u16 R, u16 P, u16 bone name offset, u16 0,
  u16 rotation times[R] (1/256 s), s16 x,y,z rotation keys[R] (Driv3r 0x58BAC0: the low bits of x and y give the
  largest component, 16383, the others are value >> 1, then normalised), u16 position times[P], s16 x,y,z /128 m [P].
- DPL v13 ("DB", ACLP chunk, names only as hashes in AIDX): clip = AnimLib::CAnimClip (28 bytes: f32 duration, u16 flags,
  u16 joints, f32 blend in, f32 blend out, s16 slot, u16 events, u32 id, u32 bone mask), 8 bytes per event, then
  joint = u16 R, u16 P, u8 times[pad4(R)] (1/30 s), u32 rotation keys[R] (CompressedRotationKey, PS2 beta 0x35F9C8),
  u8 times[pad4(P)], s16 x,y,z /1024 m [pad4(P)]. AIDX: u16 0, u16 count, then sorted (u32 id, 0, 0) for every clip.
Joints follow the skeleton's bone order (skeletons.py), then the root (Driv3r's "WorldTransformMatrix").

A key's quaternion q matches the bone-to-parent matrix M of skeletons.py as rotation(q) = M[:3, :3] transposed. The
retarget copies each shared bone's model-space rotation from Driv3r: DPL's back takes Driv3r's upperchest (DPL has no
upperchest), the bones Driv3r lacks (left_fingers, gun, hair) keep DPL's bind pose, every bone keeps DPL's length.
"""
import argparse
import math
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dplchunk as d
import skeletons as sk

V8_CLIP = struct.Struct('<IffIIffffI')
V13_CLIP = struct.Struct('<fHHffhHII')
JOINT_V8 = struct.Struct('<4H')
FPS = 30.0
V8_TIME = 256.0             # v8 key times are in 1/256 s
V8_LARGEST = 16383
V8_POSITION = 128.0
V13_POSITION = 1024.0
V13_VERSION = 13
MAX_FRAMES = 255            # v13 key times are u8 frames
ROT_A_BITS = 0x7FF          # v13 rotation key: bits 0-2 signs, 3-13 a (w = 1 - a^2), 14-31 triangular direction index
ROT_INDEX_MAX = 511
ROOT = 'WorldTransformMatrix'
NO_SLOT = -1
DPL_FROM_D3 = {'bone_back': 'bone_upperchest'}
# DPL keeps a bank's resident clips in a fixed table of 720 (0x561EBB appends without a bound check; more
# overwrites the count and crashes the id search at load). nyc_now.an4 holds 703; 16 are added (719, as tested in game):
# the moves plus Tanner's walk, jog and run, which the hook plays for the Tanner player model only.
MAX_RESIDENT_CLIPS = 720
DEFAULT_CLIPS = ['male_tread_water', 'male_swim',
                 'male_ladder_on_below', 'male_ladder_on_above', 'male_ladder_cycle', 'male_ladder_off_start',
                 'male_ladder_off_end',
                 'male_jump_on_spot_start', 'male_jump_air', 'male_jump_forward_end', 'male_jump_run_start',
                 'male_run_roll_start', 'male_roll_run',
                 'male_walk', 'male_jog', 'male_run']
CLIP_CYCLE = 0x2            # the flag of DPL's walk, run and breathing clips: the clip loops
CLIMB_CLIPS = {'male_ladder_cycle', 'male_ladder_off_start'}    # the hook lifts the climber itself: root height fixed
CYCLE_CLIPS = {'male_tread_water', 'male_swim', 'male_ladder_cycle', 'male_jump_air', 'male_walk', 'male_jog', 'male_run'}


@dataclass
class Joint:
    rot_times: np.ndarray       # seconds
    rotations: np.ndarray       # (n, 4) x, y, z, w
    pos_times: np.ndarray
    positions: np.ndarray       # (n, 3) metres
    bone: str = ''              # Driv3r: the bone the joint drives; DPL: by its place in the skeleton


@dataclass
class Clip:
    name: str
    clip_id: int
    duration: float
    flags: int
    blend_in: float
    blend_out: float
    joints: list[Joint]


def pad4(n: int) -> int:
    return (n + 3) & ~3


def decode_rotation_v8(key: tuple[int, int, int]) -> np.ndarray:
    largest = (key[0] & 1) | (key[1] & 1) << 1
    q = np.insert(np.array(key) >> 1, largest, V8_LARGEST).astype(np.float64)
    return q / np.linalg.norm(q)


def read_v8(path: Path) -> list[Clip]:
    """Every clip of a Driv3r .ab3 bank, with its name and the bone of each joint."""
    data = path.read_bytes()
    count, version, strings = struct.unpack_from('<3I', data, 4)

    if data[:4] != b'HPC1' or version != 8:
        raise SystemExit(f'{path.name}: not a version 8 HPC1 bank')

    def text(offset: int) -> str:
        return data[strings + offset:data.index(b'\0', strings + offset)].decode('latin1')

    clips = []
    at = 0x10

    for _ in range(count):
        flags, duration, _, _, n_joints, blend_in, blend_out, _, _, name_at = V8_CLIP.unpack_from(data, at)
        at += V8_CLIP.size
        joints = []

        for _ in range(n_joints):
            r, p, bone_at, _ = JOINT_V8.unpack_from(data, at)
            at += JOINT_V8.size
            rot_times = np.frombuffer(data, '<u2', r, at) / V8_TIME
            at += 2 * r
            rotations = np.array([decode_rotation_v8(k) for k in struct.iter_unpack('<3h', data[at:at + 6 * r])]).reshape(r, 4)
            at += 6 * r
            pos_times = np.frombuffer(data, '<u2', p, at) / V8_TIME
            at += 2 * p
            positions = np.frombuffer(data, '<i2', 3 * p, at).reshape(p, 3) / V8_POSITION
            at += 6 * p
            joints.append(Joint(rot_times, rotations, pos_times, positions, text(bone_at)))

        clips.append(Clip(text(name_at), 0, duration, flags, blend_in, blend_out, joints))

    return clips


def decode_rotation(cq: int) -> np.ndarray:
    """AnimLib::CompressedRotationKey -> x, y, z, w (PS2 beta 0x35F9C8)."""
    signs = [1.0 if cq & bit else -1.0 for bit in (1, 2, 4)]
    a = ((cq >> 3) & ROT_A_BITS) / ROT_A_BITS
    w = 1.0 - a * a
    s = math.sqrt(max(0.0, 1.0 - w * w))
    n = cq >> 14
    i = math.isqrt(n)
    phi = i * (math.pi / 2) / ROT_INDEX_MAX
    theta = 0.0 if i == 0 else (n - i * i) * (math.pi / 2) / (2 * i)
    sa, sb = math.sin(phi), math.sin(theta)
    x = sa * sb
    z = sa * math.sqrt(max(0.0, 1.0 - sb * sb))
    y = math.sqrt(max(0.0, 1.0 - x * x - z * z))
    return np.array([signs[0] * s * x, signs[1] * s * y, signs[2] * s * z, w])


def encode_rotation(q: np.ndarray) -> int:
    """The inverse of decode_rotation: w = 1 - a^2, the vector part's direction as (phi, theta) on a triangular grid."""
    q = q / np.linalg.norm(q)
    q = q if q[3] >= 0.0 else -q
    a = min(ROT_A_BITS, round(math.sqrt(max(0.0, 1.0 - q[3])) * ROT_A_BITS))
    signs = (q[0] >= 0.0) | (q[1] >= 0.0) << 1 | (q[2] >= 0.0) << 2
    length = float(np.linalg.norm(q[:3]))
    ux, uy, uz = (abs(q[:3]) / length) if length > 1e-9 else (0.0, 1.0, 0.0)
    i = min(ROT_INDEX_MAX, round(math.acos(min(1.0, uy)) * ROT_INDEX_MAX / (math.pi / 2)))
    j = 0 if i == 0 else min(2 * i, round(math.atan2(ux, uz) * 2 * i / (math.pi / 2)))
    return int(signs) | a << 3 | (i * i + j) << 14


def joint_size(r: int, p: int) -> int:
    return 4 + pad4(r) + 4 * r + pad4(p) + 6 * pad4(p)


def read_v13(data: bytes | memoryview, at: int, count: int) -> tuple[list[Clip], int]:
    """count DPL clips starting at data[at]; also where the last one ends."""
    clips = []

    for _ in range(count):
        duration, flags, n_joints, blend_in, blend_out, _, events, clip_id, _ = V13_CLIP.unpack_from(data, at)
        at += V13_CLIP.size + 8 * events
        joints = []

        for _ in range(n_joints):
            r, p = struct.unpack_from('<2H', data, at)
            rot_times = np.frombuffer(data, 'u1', r, at + 4) / FPS
            keys = struct.unpack_from(f'<{r}I', data, at + 4 + pad4(r))
            rotations = np.array([decode_rotation(k) for k in keys]).reshape(r, 4)
            pos_at = at + 4 + pad4(r) + 4 * r
            pos_times = np.frombuffer(data, 'u1', p, pos_at) / FPS
            positions = np.frombuffer(data, '<i2', 3 * p, pos_at + pad4(p)).reshape(p, 3) / V13_POSITION
            joints.append(Joint(rot_times, rotations, pos_times, positions))
            at += joint_size(r, p)

        clips.append(Clip(f'{clip_id:08x}', clip_id, duration, flags, blend_in, blend_out, joints))

    return clips, at


def write_v13(clip: Clip) -> bytes:
    """One DPL clip record; every joint's keys are on whole frames."""
    out = bytearray(V13_CLIP.pack(clip.duration, clip.flags, len(clip.joints), clip.blend_in, clip.blend_out, NO_SLOT, 0,
                                  clip.clip_id, (1 << len(clip.joints)) - 1))

    for joint in clip.joints:
        r, p = len(joint.rotations), len(joint.positions)
        out += struct.pack('<2H', r, p)
        out += bytes(round(t * FPS) for t in joint.rot_times).ljust(pad4(r), b'\0')
        out += struct.pack(f'<{r}I', *(encode_rotation(q) for q in joint.rotations))
        out += bytes(round(t * FPS) for t in joint.pos_times).ljust(pad4(p), b'\0')
        keys = np.zeros((pad4(p), 3))
        keys[:p] = joint.positions
        out += np.clip(np.round(keys * V13_POSITION), -32768, 32767).astype('<i2').tobytes()

    return bytes(out)


def sample(times: np.ndarray, keys: np.ndarray, t: float, rotation: bool) -> np.ndarray:
    """The key value at t: held before the first and after the last key, nlerp (rotations) or lerp between."""
    if len(keys) == 1 or t <= times[0]:
        return keys[0]

    if t >= times[-1]:
        return keys[-1]

    n = int(np.searchsorted(times, t, side='right'))
    f = (t - times[n - 1]) / (times[n] - times[n - 1])
    a, b = keys[n - 1], keys[n]

    if not rotation:
        return a + (b - a) * f

    b = b if a @ b >= 0.0 else -b
    q = a + (b - a) * f
    return q / np.linalg.norm(q)


def rotation_rows(q: np.ndarray) -> np.ndarray:
    """The 3x3 row-vector rotation of a key (the transpose of the usual column-vector matrix)."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]).T


def rows_rotation(m: np.ndarray) -> np.ndarray:
    """The key of a 3x3 row-vector rotation."""
    r = m.T
    t = np.trace(r)

    if t > 0:
        s = math.sqrt(t + 1) * 2
        return np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4])

    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1 + r[i, i] - r[j, j] - r[k, k]) * 2
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    q[3] = (r[k, j] - r[j, k]) / s
    return q


def parent_of(bone: str, bones: list[str]) -> str | None:
    parent = sk.PARENTS.get(bone[5:])

    if parent == sk.CHEST:
        parent = 'upperchest' if 'bone_upperchest' in bones else 'back'

    return None if parent is None else f'bone_{parent}'


def bind_locals(path: Path) -> dict[str, np.ndarray]:
    """{bone: 4x4 bone-to-parent bind matrix} of the file's male skeleton."""
    block = sk.skeleton_blocks(path)['male_skeleton']
    return {m.group().decode(): np.array(struct.unpack_from('<16f', block, m.start() + sk.LOCAL_MATRIX)).reshape(4, 4)
            for m in sk.BONE_NAME.finditer(block)}


def retarget(clip: Clip, d3_bones: list[str], dpl_bind: dict[str, np.ndarray], clip_id: int) -> Clip:
    """The Driv3r clip on DPL's skeleton, sampled on every frame (see the module notes for the rules)."""
    frames = max(1, min(MAX_FRAMES, round(clip.duration * FPS)))
    times = np.arange(frames + 1) / FPS
    by_bone = {j.bone: j for j in clip.joints}
    dpl_bones = list(dpl_bind)
    rotations = {b: [] for b in dpl_bones}
    pelvis = by_bone['bone_pelvis']

    for t in times * (clip.duration / (frames / FPS)):
        d3_model: dict[str, np.ndarray] = {}

        for bone in d3_bones:
            parent = parent_of(bone, d3_bones)
            local = rotation_rows(sample(by_bone[bone].rot_times, by_bone[bone].rotations, t, True))
            d3_model[bone] = local if parent is None else local @ d3_model[parent]

        dpl_model: dict[str, np.ndarray] = {}

        for bone in dpl_bones:
            parent = parent_of(bone, dpl_bones)
            source = DPL_FROM_D3.get(bone, bone)
            dpl_model[bone] = (d3_model[source] if source in d3_model
                               else dpl_bind[bone][:3, :3] @ (np.eye(3) if parent is None else dpl_model[parent]))
            local = dpl_model[bone] if parent is None else dpl_model[bone] @ dpl_model[parent].T
            rotations[bone].append(rows_rotation(local))

    # Driv3r's root carries the hip height (walk 1.06 m) and its pelvis sits near 0; DPL's root stays on the ground
    # (y 0, only its travel is used) and its pelvis carries the height. So the root's height moves to the pelvis (a
    # roll's dive down with it); a climb keeps only its first height, the hook lifts him itself.
    root = by_bone[ROOT]
    root_positions = np.array([sample(root.pos_times, root.positions, t, False) for t in times])
    heights = root_positions[:, 1].copy()

    if clip.name in CLIMB_CLIPS:
        heights[:] = heights[0]

    root_positions[:, 1] = 0.0
    joints = []

    for bone in dpl_bones:
        keys = np.array(rotations[bone])

        for n in range(1, len(keys)):
            keys[n] = keys[n] if keys[n] @ keys[n - 1] >= 0.0 else -keys[n]

        if bone == 'bone_pelvis':
            positions = np.array([sample(pelvis.pos_times, pelvis.positions, t, False) for t in times * (clip.duration / (frames / FPS))])
            positions[:, 1] += heights
            joints.append(Joint(times, keys, times, positions))
        else:
            joints.append(Joint(times, keys, np.zeros(1), dpl_bind[bone][3:, :3].copy()))

    joints.append(Joint(root.rot_times[-1:] * 0, root.rotations[:1], np.zeros(1), root_positions[:1]))

    if len(root.rotations) > 1 or len(root.positions) > 1:
        joints[-1] = Joint(times, np.array([sample(root.rot_times, root.rotations, t, True) for t in times]),
                           times, root_positions)

    flags = CLIP_CYCLE if clip.name in CYCLE_CLIPS else 0
    return Clip(clip.name, clip_id, frames / FPS, flags, clip.blend_in, clip.blend_out, joints)


def bank_chunks(tree: d.ChunkTree) -> tuple[d.Chunk, d.Chunk]:
    ssic = next(c for c in tree.chunks if c.tag == b'SSIC')
    found = {c.tag: c for c in ssic.tree.chunks}
    return found[b'AIDX'], found[b'ACLP']


def step_angles(clips: list[Clip]) -> np.ndarray:
    """Angle in degrees between neighbouring rotation keys of every joint: small when the keys decode right."""
    steps = []

    for clip in clips:
        for joint in clip.joints:
            q = joint.rotations

            if len(q) > 1:
                steps.extend(np.degrees(2 * np.arccos(np.abs((q[1:] * q[:-1]).sum(1)).clip(0.0, 1.0))))

    return np.array(steps)


def dpl_clips(path: Path) -> tuple[list[Clip], bytes, int]:
    """The main ACLP bank of a DPL .an4: clips, the chunk's bytes and where its clips end."""
    _, aclp = bank_chunks(d.read_tree(path.read_bytes(), 0, 2))
    data = bytes(aclp.data)
    clips, end = read_v13(data, 0x10, struct.unpack_from('<I', data, 4)[0])
    return clips, data, end


def check(ab3: Path, an4: Path, clip_name: str) -> None:
    d3 = read_v8(ab3)
    dpl, _, _ = dpl_clips(an4)

    for label, clips in (('Driv3r', d3), ('DPL', dpl)):
        norms = np.concatenate([np.linalg.norm(j.rotations, axis=1) for c in clips for j in c.joints if len(j.rotations)])
        steps = step_angles(clips)
        print(f'{label}: {len(clips)} clips, {sum(len(c.joints) for c in clips)} joints, quat length {norms.min():.4f}..'
              f'{norms.max():.4f}, key step median {np.median(steps):.2f} deg, 99% {np.percentile(steps, 99):.1f} deg')

    clip = next((c for c in d3 if c.name == clip_name), None)

    if clip:
        print(f'{clip.name}: {clip.duration:.3f} s, flags {clip.flags:#x}, blend {clip.blend_in:.2f}/{clip.blend_out:.2f}')

        for joint in clip.joints:
            print(f'  {joint.bone:22s} {len(joint.rotations):3d} rotations, {len(joint.positions):3d} positions')


def roundtrip(an4: Path) -> None:
    _, data, end = dpl_clips(an4)
    at = 0x10
    same = total = 0

    while at < end:
        events, n_joints = struct.unpack_from('<H', data, at + 18)[0], struct.unpack_from('<H', data, at + 6)[0]
        at += V13_CLIP.size + 8 * events

        for _ in range(n_joints):
            r, p = struct.unpack_from('<2H', data, at)

            for k in struct.unpack_from(f'<{r}I', data, at + 4 + pad4(r)):
                total += 1
                same += encode_rotation(decode_rotation(k)) == k

            at += joint_size(r, p)

    print(f'{same} of {total} rotation keys re-encode to the same bits; clips end at {end:#x} of {len(data):#x}')


def build(ab3: Path, dam: Path, an4: Path, out: Path, names: list[str]) -> None:
    if out.resolve() == an4.resolve():
        raise SystemExit('write to a copy, not over the source bank')

    d3 = {c.name: c for c in read_v8(ab3)}
    missing = [n for n in names if n not in d3]

    if missing:
        raise SystemExit(f'not in {ab3.name}: {" ".join(missing)}')

    d3_bones = list(bind_locals(dam))
    dpl_bind = bind_locals(an4)
    tree = d.read_tree(an4.read_bytes(), 0, 2)
    aidx, aclp = bank_chunks(tree)
    count = struct.unpack_from('<H', aidx.data, 2)[0]
    ids = [struct.unpack_from('<I', aidx.data, 4 + 12 * i)[0] for i in range(count)]
    old = bytes(aclp.data)
    clip_count = struct.unpack_from('<I', old, 4)[0]

    if clip_count + len(names) > MAX_RESIDENT_CLIPS:
        raise SystemExit(f'{an4.name}: {clip_count} + {len(names)} clips exceed DPL\'s table of {MAX_RESIDENT_CLIPS}')

    _, end = read_v13(old, 0x10, clip_count)
    added = []
    listing = []

    for name in names:
        clip_id = zlib.crc32(f'd3:{name}'.encode())

        if clip_id in ids:
            raise SystemExit(f'{name}: id {clip_id:08x} is taken')

        ids.append(clip_id)
        clip = retarget(d3[name], d3_bones, dpl_bind, clip_id)
        added.append(write_v13(clip))
        listing.append(f'{clip_id:08x} {name} {clip.duration:.3f}')

    aclp.data = old[:4] + struct.pack('<I', clip_count + len(added)) + old[8:end] + b''.join(added) + old[end:]
    ids.sort()
    aidx.data = (bytes(aidx.data[:2]) + struct.pack('<H', len(ids)) + b''.join(struct.pack('<3I', i, 0, 0) for i in ids)
                 + bytes(aidx.data[4 + 12 * count:]))

    with out.open('wb') as f:
        for piece in d.tree_pieces(tree):
            f.write(piece)

    out.with_suffix('.clips.txt').write_text('\n'.join(listing) + '\n', encoding='utf-8')
    print(f'{out.name}: {len(added)} clips added ({sum(map(len, added))} bytes), ids in {out.with_suffix(".clips.txt").name}')


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('ab3', type=Path)
    c.add_argument('an4', type=Path)
    c.add_argument('--clip', default='male_swim')
    r = sub.add_parser('roundtrip')
    r.add_argument('an4', type=Path)
    b = sub.add_parser('build')
    b.add_argument('ab3', type=Path)
    b.add_argument('dam', type=Path)
    b.add_argument('an4', type=Path)
    b.add_argument('out', type=Path)
    b.add_argument('--clips', nargs='+', default=DEFAULT_CLIPS)
    args = ap.parse_args()

    if args.command == 'check':
        check(args.ab3, args.an4, args.clip)
    elif args.command == 'roundtrip':
        roundtrip(args.an4)
    else:
        build(args.ab3, args.dam, args.an4, args.out, args.clips)


if __name__ == '__main__':
    main()
