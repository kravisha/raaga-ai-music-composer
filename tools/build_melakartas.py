"""Turn the Stage 1 knowledge pack's melakarta maps into library data.

``docs/spec/stage1_knowledge_pack/`` holds the creator's pack verbatim and is
frozen: a section quoted in a commit or a test must resolve to the creator's
own wording, so nothing at runtime reads those files.  This script is the
bridge.  It parses the three map documents, checks every record against the
pack's own validation rules (document 06), and writes
``raagacomposer/raaga/data/melakartas.json``, which the library loads.

Run::

    .venv\\Scripts\\python.exe tools\\build_melakartas.py [--check]

``--check`` regenerates in memory and compares with the file on disk without
writing, which is what ``tests/unit/test_stage1_pack.py`` uses to prove the
committed data still says what the pack says.

What is carried across, and how it is classed (pack document 00, and the
Agent Factory's ``KnowledgeClass``):

* ``[GRAMMAR]`` - id, name, chakra, arohanam, avarohanam and the three blocks
  the scale decomposes into.  Hard knowledge.
* ``[HEURISTIC]`` - the block characters, the starter tags and the "good for"
  uses.  Defeasible starter priors, kept apart from the curated ``moods`` in
  ``raagas.json`` so that a learned weight can move one without touching the
  other (pack document 05 section 6).

Nothing here invents jeeva swaras, nyasa, prayogas, gamaka or a tempo range:
the pack does not have them and specification section 37 says unknown fields
remain unknown.  A melakarta with no curated entry is therefore a scale and
its character, and the application says as much.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "docs" / "spec" / "stage1_knowledge_pack"
OUT = ROOT / "raagacomposer" / "raaga" / "data" / "melakartas.json"
MAPS = ("02_MELAKARTA_MAP_01_24.txt", "03_MELAKARTA_MAP_25_48.txt",
        "04_MELAKARTA_MAP_49_72.txt")

ROW = re.compile(r"^(\d{2})\|")

#: Pack document 00, "MELAKARTA HARD RULES".
VALID_RG = (("R1", "G1"), ("R1", "G2"), ("R1", "G3"),
            ("R2", "G2"), ("R2", "G3"), ("R3", "G3"))
VALID_DN = (("D1", "N1"), ("D1", "N2"), ("D1", "N3"),
            ("D2", "N2"), ("D2", "N3"), ("D3", "N3"))

#: Pack document 01 section A.  The library's own table must agree with this;
#: ``tests/unit/test_stage1_pack.py`` test B is what checks that it does.
PITCH_CLASS = {"S": 0, "R1": 1, "R2": 2, "R3": 3, "G1": 2, "G2": 3, "G3": 4,
               "M1": 5, "M2": 6, "P": 7, "D1": 8, "D2": 9, "D3": 10,
               "N1": 9, "N2": 10, "N3": 11}


class PackError(ValueError):
    """A record that breaks one of the pack's own validation rules."""


def _split_character(field: str) -> tuple:
    """``"R2G2:tender, introspective"`` -> ``("R2G2", "tender, introspective")``."""
    block, _, character = field.partition(":")
    return block.strip(), character.strip()


def parse(pack: Path = PACK) -> List[dict]:
    """Every melakarta record in the pack, in id order, unvalidated."""
    records: Dict[int, dict] = {}
    for name in MAPS:
        text = (pack / name).read_text(encoding="utf-8")
        for line in text.splitlines():
            if not ROW.match(line):
                continue
            parts = line.split("|")
            if len(parts) != 10:
                raise PackError(f"{name}: expected 10 fields, got {len(parts)}: "
                                f"{line[:60]!r}")
            mela = int(parts[0])
            if mela in records:
                raise PackError(f"melakarta {mela} appears twice")
            aro = parts[3].split()
            avaro = parts[4].split()
            rg, rg_character = _split_character(parts[5])
            madhyama, m_character = _split_character(parts[6])
            dn, dn_character = _split_character(parts[7])
            records[mela] = {
                "id": mela,
                "name": parts[1].strip(),
                "chakra": parts[2].strip(),
                # The library writes the upper tonic as "S+" (raaga/library.py,
                # parse_swara); the pack writes both tonics as plain "S".
                "arohanam": aro[:-1] + ["S+"],
                "avarohanam": ["S+"] + avaro[1:],
                "rg": rg,
                "madhyama": madhyama,
                "dn": dn,
                "block_character": {rg: rg_character, madhyama: m_character,
                                    dn: dn_character},
                "tags": [t.strip() for t in parts[8].split(",") if t.strip()],
                "good_for": [g.strip() for g in parts[9].split(";") if g.strip()],
            }
    return [records[k] for k in sorted(records)]


def validate(records: List[dict]) -> None:
    """The pack's own VALIDATION RULES, document 06.  Raises on the first break."""
    ids = [r["id"] for r in records]
    if ids != list(range(1, 73)):
        raise PackError(f"expected melakartas 1..72, got {len(ids)} records")

    scales = set()
    for record in records:
        mela, name = record["id"], record["name"]
        aro = [s.rstrip("+-") for s in record["arohanam"]]
        avaro = [s.rstrip("+-") for s in record["avarohanam"]]
        where = f"{mela} {name}"

        if len(aro) != 8 or aro[0] != "S" or aro[-1] != "S":
            raise PackError(f"{where}: arohanam is not S..S in eight steps: {aro}")
        if aro[4] != "P":
            raise PackError(f"{where}: Panchama is not fixed: {aro}")
        if avaro != list(reversed(aro)):
            raise PackError(f"{where}: avarohanam is not the arohanam reversed")

        r, g, m, d, n = aro[1], aro[2], aro[3], aro[5], aro[6]
        if (r, g) not in VALID_RG:
            raise PackError(f"{where}: {r}{g} is not one of the six R-G blocks")
        if (d, n) not in VALID_DN:
            raise PackError(f"{where}: {d}{n} is not one of the six D-N blocks")
        expected_m = "M1" if mela <= 36 else "M2"
        if m != expected_m:
            raise PackError(f"{where}: melakarta {mela} must use {expected_m}, has {m}")
        if PITCH_CLASS[r] > PITCH_CLASS[g]:
            raise PackError(f"{where}: {r} is not below {g}")
        if PITCH_CLASS[d] > PITCH_CLASS[n]:
            raise PackError(f"{where}: {d} is not below {n}")

        if record["rg"] != r + g or record["dn"] != d + n or record["madhyama"] != m:
            raise PackError(f"{where}: the stated blocks do not match the scale")
        for block, character in record["block_character"].items():
            if not character:
                raise PackError(f"{where}: block {block} has no character")
        if not record["tags"] or not record["good_for"]:
            raise PackError(f"{where}: heuristic tags or uses are missing")

        scale = " ".join(aro)
        if scale in scales:
            raise PackError(f"{where}: another melakarta already has this scale")
        scales.add(scale)


def build(pack: Path = PACK) -> dict:
    records = parse(pack)
    validate(records)
    return {
        "_comment": ("Generated by tools/build_melakartas.py from "
                     "docs/spec/stage1_knowledge_pack/.  Do not edit by hand; "
                     "edit the pack and regenerate."),
        "melakartas": records,
    }


def rendered(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare with the committed file instead of writing")
    args = parser.parse_args(argv)

    text = rendered(build())
    if args.check:
        if not OUT.exists():
            print(f"{OUT} does not exist; run without --check", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") == text:
            print(f"{OUT.name} is up to date with the pack")
            return 0
        print(f"{OUT.name} differs from the pack; regenerate it", file=sys.stderr)
        return 1

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(build()['melakartas'])} melakartas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
