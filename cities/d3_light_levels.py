"""Points the Miami LEVEL of Cities/levels.txt at the converted vehicle spool and animation bank.

usage: d3_light_levels.py <DPL dir> [--vehicles nyc_d3_vehicles.sp] [--animations nyc_now_d3.an4] [--map miami.map]

levels.txt is backed up to <DPL dir>/_backup/<date>/ before the first change of the day; every other byte stays.
"""
import argparse
from pathlib import Path

import d3_vehicle_swap as s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('dpl', type=Path)
    ap.add_argument('--vehicles', default='nyc_d3_vehicles.sp')
    ap.add_argument('--animations', default='nyc_now_d3.an4')
    ap.add_argument('--map', default='miami.map')
    args = ap.parse_args()
    s.set_level(args.dpl, args.vehicles, b'SpooledVehicles', 'vehicles')
    s.set_level(args.dpl, args.animations, b'Animations', 'Anims')
    s.set_level(args.dpl, args.map, b'Map', 'Overlays')


if __name__ == '__main__':
    main()
