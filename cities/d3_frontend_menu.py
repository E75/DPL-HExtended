"""Makes the frontend menu with the D3 DLC entry: a copy of Driver Parallel Lines' title menu, GUI/D3FRONT.MEC and .TXT.

usage: d3_frontend_menu.py <DPL dir>

DPLExtended loads GUI\\D3FRONT instead of GUI\\FRONTEND when the copy exists. The title page (TitleSc) keeps its six
buttons; only fixed-size fields change in place (docs/D3-TO-DPL-CONVERSION.md section 9):
- the four buttons the hook relabels (Load Profile, Options, Credits and the hidden Dev Menu) get a text name, so the
- Options, Credits, New Game and Quit get a status name, so the hook can grey or hide them in the city choice;
- the Dev Menu button moves up one row (below Quit) and its label in the TXT copy reads "D3 DLC".
The originals GUI/Frontend.MEC and .TXT are only read.
"""
import argparse
import struct
from pathlib import Path

NAME_FIELD = 32
BUTTON = b'\x03'                     # a button record starts 03 xx 00 80 3e, its y position follows at +5
BUTTON_Y = 5
BUTTON_ACTION = 0x227                # action name field from the button record's start; the status name follows
BUTTON_STATUS = BUTTON_ACTION + NAME_FIELD
TEXT_STYLE = b'\x00\x00\x02\x66\x66\x66\x3f\x66\x66\x66\x3f'   # text id, then this, in each state of a title button
TEXT_NAME = 52                       # text name field from the text id
ROW = 0.0666666                      # distance between title buttons
# (text id of the button's label, action name, text name to add, status name to add or None to keep the file's)
BUTTONS = [
    (42, b'', None, b'd3stat_newgame'),
    (13, b'bu_load_profile', b'd3txt_load', None),
    (44, b'LaunchOptionsMenu', b'd3txt_options', b'd3stat_options'),
    (45, b'', None, b'd3stat_quit'),
    (46, b'bu_credits', b'd3txt_credits', b'd3stat_credits'),
    (43, b'launch_dev_menu', b'd3txt_dev', None),
]
DEV_TEXT_ID = 43
DEV_LABEL = 'D3 DLC'


def empty(data: bytes, at: int) -> bool:
    return data[at] == 0 and data[at + 1:at + NAME_FIELD] == b'\xcc' * (NAME_FIELD - 1)


def put_name(data: bytearray, at: int, name: bytes) -> None:
    if not empty(data, at):
        raise SystemExit(f'Frontend.MEC: the name field at {at:#x} is not empty: {bytes(data[at:at + NAME_FIELD])!r}')

    data[at:at + len(name) + 1] = name + b'\0'


def title_buttons(data: bytes) -> dict[int, int]:
    """Button record start per label text id on the title page."""
    start, end = data.find(b'TitleSc\0'), data.find(b'NewGame\0')
    buttons = {}
    at = start

    while (at := data.find(BUTTON, at + 1, end)) >= 0:
        if data[at + 2:at + 5] != b'\x00\x80\x3e' or data[at + 1] > 1:
            continue

        text = data.find(TEXT_STYLE, at, at + BUTTON_ACTION) - 1
        buttons[data[text]] = at

    if sorted(buttons) != sorted(b[0] for b in BUTTONS):
        raise SystemExit(f'Frontend.MEC: title buttons {sorted(buttons)}, expected {sorted(b[0] for b in BUTTONS)}')

    return buttons


def build_mec(data: bytearray) -> None:
    buttons = title_buttons(data)

    for text_id, action, text_name, status in BUTTONS:
        at = buttons[text_id]
        found = bytes(data[at + BUTTON_ACTION:at + BUTTON_ACTION + NAME_FIELD]).split(b'\0')[0]

        if found != action:
            raise SystemExit(f'Frontend.MEC: button {text_id} has action {found!r}, expected {action!r}')

        if text_name:
            texts = [i for i in range(at, at + BUTTON_ACTION) if data[i] == text_id and data[i + 1:i + 12] == TEXT_STYLE]

            if len(texts) != 2:
                raise SystemExit(f'Frontend.MEC: button {text_id} has {len(texts)} label texts, expected 2')

            for text in texts:
                put_name(data, text + TEXT_NAME, text_name)

        if status:
            put_name(data, at + BUTTON_STATUS, status)

    quit_y = struct.unpack_from('<f', data, buttons[45] + BUTTON_Y)[0]
    struct.pack_into('<f', data, buttons[DEV_TEXT_ID] + BUTTON_Y, quit_y + ROW)


def build_txt(text: str) -> str:
    old = f'<ID>{DEV_TEXT_ID:05d}</ID><TEXT>Dev Menu</TEXT>'

    if text.count(old) != 1:
        raise SystemExit(f'Frontend.TXT: "{old}" is there {text.count(old)} times, expected once')

    return text.replace(old, f'<ID>{DEV_TEXT_ID:05d}</ID><TEXT>{DEV_LABEL}</TEXT>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('dpl', type=Path)
    args = ap.parse_args()
    gui = args.dpl / 'GUI'

    mec = bytearray((gui / 'Frontend.MEC').read_bytes())
    build_mec(mec)
    (gui / 'D3FRONT.MEC').write_bytes(mec)

    # UTF-16 with BOM and CRLF, as DPL's own
    with open(gui / 'Frontend.TXT', encoding='utf-16', newline='') as source:
        text = build_txt(source.read())

    with open(gui / 'D3FRONT.TXT', 'w', encoding='utf-16', newline='') as target:
        target.write(text)

    print(f'wrote {gui / "D3FRONT.MEC"} and {gui / "D3FRONT.TXT"}')


if __name__ == '__main__':
    main()
