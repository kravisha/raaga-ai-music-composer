"""The Stage 1 knowledge pack (docs/spec/stage1_knowledge_pack/) as a checked
asset rather than a text: its own validation rules and mandatory unit tests
(document 06), and the check that the library's melakarta entries agree
with its grammar.  Grammar rows are hard knowledge; tags are heuristics and
are not tested for truth here."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Dict, List

import pytest

from raagacomposer.raaga.library import SWARA_SEMITONES, library, parse_swara

pytestmark = pytest.mark.unit

PACK = Path(__file__).resolve().parents[2] / "docs" / "spec" / "stage1_knowledge_pack"
ROW = re.compile(r"^(\d{2})\|")
VALID_RG = {("R1", "G1"), ("R1", "G2"), ("R1", "G3"), ("R2", "G2"),
            ("R2", "G3"), ("R3", "G3")}
VALID_DN = {("D1", "N1"), ("D1", "N2"), ("D1", "N3"), ("D2", "N2"),
            ("D2", "N3"), ("D3", "N3")}
PITCH_CLASS = {"S": 0, "R1": 1, "R2": 2, "R3": 3, "G1": 2, "G2": 3, "G3": 4,
               "M1": 5, "M2": 6, "P": 7, "D1": 8, "D2": 9, "D3": 10,
               "N1": 9, "N2": 10, "N3": 11}


def _records() -> Dict[int, dict]:
    records: Dict[int, dict] = {}
    for name in ("02_MELAKARTA_MAP_01_24.txt", "03_MELAKARTA_MAP_25_48.txt",
                 "04_MELAKARTA_MAP_49_72.txt"):
        for line in (PACK / name).read_text(encoding="utf-8").splitlines():
            if not ROW.match(line):
                continue
            parts = line.split("|")
            records[int(parts[0])] = {
                "name": parts[1], "chakra": parts[2],
                "aro": parts[3].split(), "avaro": parts[4].split(),
                "tags": parts[8].split(","), "good_for": parts[9].split(";"),
            }
    return records


@pytest.fixture(scope="module")
def records() -> Dict[int, dict]:
    return _records()


# -- validation rules (document 06) ----------------------------------------
def test_exactly_72_records_with_ids_1_to_72(records):
    assert sorted(records) == list(range(1, 73))


def test_every_arohana_is_krama_sampurna_with_valid_blocks(records):
    for mela, record in records.items():
        aro = record["aro"]
        assert len(aro) == 8 and aro[0] == "S" and aro[-1] == "S", (mela, aro)
        assert aro[4] == "P", (mela, aro)
        r, g, m, d, n = aro[1], aro[2], aro[3], aro[5], aro[6]
        assert (r, g) in VALID_RG, (mela, r, g)
        assert m in ("M1", "M2"), (mela, m)
        assert (d, n) in VALID_DN, (mela, d, n)
        assert record["avaro"] == list(reversed(aro)), (mela, record["avaro"])
        # R below G and D below N in principal pitch position.
        assert PITCH_CLASS[r] <= PITCH_CLASS[g] and PITCH_CLASS[d] <= PITCH_CLASS[n]


def test_madhyama_halves(records):
    for mela, record in records.items():
        expected = "M1" if mela <= 36 else "M2"
        assert record["aro"][3] == expected, (mela, record["aro"])


def test_no_two_melakartas_share_a_scale(records):
    scales = [" ".join(r["aro"]) for r in records.values()]
    assert len(set(scales)) == 72


# -- mandatory unit tests A, B, C ------------------------------------------
def test_a_keeravani_is_number_21(records):
    assert records[21]["name"] == "Keeravani"
    assert records[21]["aro"] == "S R2 G2 M1 P D1 N3 S".split()
    assert records[21]["avaro"] == "S N3 D1 P M1 G2 R2 S".split()


def test_b_swara_overlaps_share_a_pitch_but_keep_their_labels():
    for a, b in (("R2", "G1"), ("R3", "G2"), ("D2", "N1"), ("D3", "N2")):
        assert PITCH_CLASS[a] == PITCH_CLASS[b]
        assert a != b
    # The application's own swara table agrees with the pack's pitch classes
    # for every swara it knows.
    for swara, semitone in SWARA_SEMITONES.items():
        base = parse_swara(swara)[0]
        if base in PITCH_CLASS:
            assert semitone % 12 == PITCH_CLASS[base], swara


def test_c_melakarta_endpoints(records):
    assert records[1]["name"] == "Kanakangi"
    assert records[1]["aro"] == "S R1 G1 M1 P D1 N1 S".split()
    assert records[36]["name"] == "Chalanata"
    assert records[36]["aro"] == "S R3 G3 M1 P D3 N3 S".split()
    assert records[37]["name"] == "Salagam"
    assert records[37]["aro"] == "S R1 G1 M2 P D1 N1 S".split()
    assert records[72]["name"] == "Rasikapriya"
    assert records[72]["aro"] == "S R3 G3 M2 P D3 N3 S".split()


# -- the library against the pack's grammar ----------------------------------
def _bases(tokens: List[str]) -> List[str]:
    return [parse_swara(t)[0] for t in tokens]


def test_the_library_melakarta_entries_agree_with_the_pack(records):
    checked = 0
    for raaga in library().all():
        if not raaga.melakarta:
            continue
        record = records[int(raaga.melakarta)]
        # The pack uses the katapayadi names (Mechakalyani, Dheerasankarabharanam);
        # the library uses the names musicians say, so either may contain the
        # other, or an alias may match.
        def plain(text: str) -> str:
            return re.sub(r"[^a-z]", "", text.lower()).replace("sh", "s")

        names = {plain(raaga.name)} | {plain(a) for a in raaga.aliases}
        pack_name = plain(record["name"])
        assert any(n == pack_name or n in pack_name or pack_name in n
                   for n in names), (raaga.name, record["name"])
        assert _bases(raaga.arohanam) == record["aro"], raaga.name
        assert _bases(raaga.avarohanam) == record["avaro"], raaga.name
        checked += 1
    assert checked == 72, "the library carries every melakarta the pack defines"


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_melakartas",
        PACK.parents[2] / "tools" / "build_melakartas.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rescale(record: dict, arohanam: str) -> None:
    """Give a record a different scale, consistently in both directions.

    Changing only the arohanam trips the reversal rule before anything else,
    which is not what most of these cases are trying to provoke.
    """
    aro = arohanam.split()
    record["arohanam"] = aro
    record["avarohanam"] = list(reversed(aro))


@pytest.mark.parametrize("break_it, complaint", [
    (lambda rs: rs.pop(), "1..72"),
    (lambda rs: _rescale(rs[20], "S R2 G2 M1 P D1 S+"), "eight steps"),
    # Melakarta 21 is in the first half, so its madhyama must be M1.
    (lambda rs: _rescale(rs[20], "S R2 G2 M2 P D1 N3 S+"), "must use M1"),
    (lambda rs: _rescale(rs[0], "S R2 G1 M1 P D1 N1 S+"),
     "not one of the six R-G blocks"),
    (lambda rs: _rescale(rs[0], "S R1 G1 M1 P D2 N1 S+"),
     "not one of the six D-N blocks"),
    (lambda rs: rs[35].__setitem__("avarohanam", "S+ N3 D3 P M1 G3 R2 S".split()),
     "not the arohanam reversed"),
    (lambda rs: rs[35].__setitem__("arohanam", "S R3 G3 M1 D3 P N3 S+".split()),
     "Panchama is not fixed"),
    (lambda rs: rs[20].__setitem__("dn", "D2N3"),
     "stated blocks do not match the scale"),
    (lambda rs: rs[7].__setitem__("tags", []), "tags or uses are missing"),
    (lambda rs: rs[7].__setitem__("good_for", []), "tags or uses are missing"),
    (lambda rs: rs[7]["block_character"].__setitem__(rs[7]["rg"], ""),
     "has no character"),
])
def test_the_validator_rejects_a_record_that_breaks_the_packs_rules(
        break_it, complaint):
    """The rules in pack document 06 are enforced, not merely quoted."""
    builder = _builder()
    records = builder.parse()
    break_it(records)
    with pytest.raises(builder.PackError, match=complaint):
        builder.validate(records)


def test_the_validator_accepts_the_pack_as_it_stands():
    builder = _builder()
    builder.validate(builder.parse())


def test_the_generated_library_data_still_says_what_the_pack_says():
    """``raagacomposer/raaga/data/melakartas.json`` is generated, not written.

    The pack under ``docs/spec/`` is frozen and nothing at runtime reads it,
    so the committed data is what the library actually loads.  This is the
    check that the two have not drifted apart; ``--check`` is the same
    comparison from the command line.
    """
    builder = _builder()
    committed = builder.OUT.read_text(encoding="utf-8")
    assert committed == builder.rendered(builder.build()), (
        "melakartas.json is out of date; run tools/build_melakartas.py")


def test_heuristic_tags_are_present_but_are_not_grammar(records):
    """Every record carries starter tags and uses; the pack itself says
    they are defeasible, so nothing here asserts an emotion."""
    for record in records.values():
        assert record["tags"] and record["good_for"]
