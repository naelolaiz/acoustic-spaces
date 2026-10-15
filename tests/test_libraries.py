"""Material libraries of CATT-Acoustic and ODEON (materials.libraries)."""

from pathlib import Path

import pytest

from acoustic_spaces import Room
from acoustic_spaces.cli import main
from acoustic_spaces.examples import example_path
from acoustic_spaces.materials import find_material, read_library
from acoustic_spaces.scene import Scene
from acoustic_spaces.scene.model import Material

EXAMPLE = example_path("material_libraries").parent
CATT = EXAMPLE / "catt_materials.geo"
ODEON = EXAMPLE / "odeon_materials.li8"


def test_catt_materials_in_percent_with_scattering() -> None:
    carpet, board, walls = read_library(CATT)
    assert carpet.name == "carpet" and carpet.number is None
    assert carpet.frequencies == (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
    assert carpet.absorption == pytest.approx((0.08, 0.24, 0.57, 0.69, 0.71, 0.73))
    assert carpet.scattering == pytest.approx((0.1, 0.1, 0.1, 0.15, 0.2, 0.25))
    # Another name for a material takes its values.
    assert walls.name == "walls" and walls.absorption == board.absorption
    assert find_material(CATT, "WALLS").scattering == board.scattering


def test_catt_definitions_in_a_geometry_file(tmp_path: Path) -> None:
    geo = tmp_path / "hall.geo"
    geo.write_text(
        "; a comment: ABS ignored = <1 1 1 1 1 1>\n"
        "abs seats = <40 50 60 70 80 80 80 75> L <30 40 50 60 70 80> {0 202 101}\n"
        "ABS woodblocks = <12 10 8 7 6 5> L < estimate(0.2) >\n"
        "ABS stage=<10 10 10 10 10 10 10>\n"
        "CORNERS\n1 0 0 0\nPLANES\n[1 floor / 1 2 3 / seats]\n"
    )
    seats, blocks, stage = read_library(geo)
    # 8 kHz and 16 kHz values; scattering held at 4 kHz's to match.
    assert seats.frequencies[-1] == 16000.0 and len(seats.scattering) == 8
    assert seats.scattering[-2:] == pytest.approx((0.8, 0.8))
    assert blocks.scattering is None  # estimated by CATT, not given
    assert stage.frequencies[-1] == 8000.0
    material = Material(file=geo, entry="seats")
    assert material.absorption == pytest.approx([0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8])
    assert material.scattering == pytest.approx([0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8])
    assert material.source == "'seats' of hall.geo (63 Hz copied from the nearest band)"


def test_odeon_library_by_number_or_description() -> None:
    tiles, door = read_library(ODEON)
    assert tiles.number == 1001 and tiles.name.startswith("Acoustic ceiling tiles")
    assert tiles.frequencies[0] == 63.0 and len(tiles.absorption) == 8
    assert tiles.scattering is None
    assert find_material(ODEON, 1002) == door
    assert find_material(ODEON, "wooden door, solid") == door
    material = Material(file=ODEON, entry=1001)
    assert material.absorption == pytest.approx(list(tiles.absorption))
    assert material.source == f"'1001 {tiles.name}' of odeon_materials.li8"


def test_odeon_library_as_odeon_writes_it(tmp_path: Path) -> None:
    # ODEON's own Material.Li8: a byte order mark, CRLF, tabs, product name and category
    # fields, and Material calculator entries with their settings after the description
    # and the sample area and third octaves after the octave values.
    third_octaves = "\t".join(["0.50000"] * 24)
    lib = tmp_path / "Material.Li8"
    lib.write_bytes(
        (
            "0\t{Product name*=Transparent}, {Category=Special}\r\n"
            "\t0.00000\t0.00000\t0.00000\t0.00000\t0.00000\t0.00000\t0.00000\t0.00000\r\n\r\n"
            "15001\t{Product name*=Slotted plate, 50 mm cavity}, {Category=Acoustic material}"
            "\t{Material Calculator. Do not edit manually!};(IsInfinite,0)(PanelLength,3.9)\r\n"
            "\t0.07000\t0.65000\t0.52000\t0.08000\t0.02000\t0.01000\t0.01000\t0.01000"
            f"\t/Area\t10.14000\t/A\t{third_octaves}\r\n\r\n"
        ).encode("utf-8-sig")
    )
    transparent, plate = read_library(lib)
    assert (transparent.number, transparent.name, transparent.category) == (
        0,
        "Transparent",
        "Special",
    )
    assert plate.name == "Slotted plate, 50 mm cavity" and plate.category == "Acoustic material"
    assert plate.absorption == pytest.approx((0.07, 0.65, 0.52, 0.08, 0.02, 0.01, 0.01, 0.01))
    assert find_material(lib, "slotted plate, 50 mm cavity") == plate
    assert Material(file=lib, entry=15001).source.startswith("'15001 Slotted plate")


def test_odeon_library_of_six_bands_in_a_windows_encoding(tmp_path: Path) -> None:
    lib = tmp_path / "old.lib"
    lib.write_bytes("  12  Træ på strøer\n 0.2 0.15 0.1 0.08 0.07 0.06\n".encode("cp1252"))
    (wood,) = read_library(lib)
    assert wood.name == "Træ på strøer" and wood.number == 12
    assert wood.frequencies == (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)


def test_library_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.geo"
    bad.write_text("ABS a = <1 2 3>\n")
    with pytest.raises(ValueError, match="expected 6 to 8 absorption values"):
        read_library(bad)
    bad.write_text("ABS a = absoflex\n")
    with pytest.raises(ValueError, match="'absoflex', which the file does not define"):
        read_library(bad)
    odd = tmp_path / "odd.li8"
    odd.write_text("1 Odd\n0.1 0.2 0.3\n")
    with pytest.raises(ValueError, match="expected 8 absorption values"):
        read_library(odd)
    empty = tmp_path / "notes.txt"
    empty.write_text("nothing here\n")
    with pytest.raises(ValueError, match="no materials found"):
        read_library(empty)
    with pytest.raises(ValueError, match="no material 'marble'; it has carpet, plasterboard"):
        find_material(CATT, "marble")
    with pytest.raises(ValueError, match="a library of 3 materials: name one with entry"):
        Material(file=CATT)
    with pytest.raises(ValueError, match="give the file too"):
        Material(preset="plasterboard", entry="carpet")


def test_a_library_of_one_material_needs_no_entry(tmp_path: Path) -> None:
    one = tmp_path / "one.geo"
    one.write_text("ABS felt = <5 10 30 60 80 90>\n")
    assert Material(file=one).absorption[4] == pytest.approx(0.6)


def test_the_example_takes_its_materials_from_both() -> None:
    room = Room.from_yaml(example_path("material_libraries"))
    materials = room.scene.materials
    assert materials["carpet"].source.startswith("'carpet' of catt_materials.geo")
    assert materials["plasterboard"].scattering[5] == pytest.approx(0.1)  # 2 kHz
    assert materials["ceiling_tiles"].absorption[0] == pytest.approx(0.3)  # ODEON has 63 Hz
    assert room.problems() == []
    # The scene keeps naming the entries when written out.
    data = Scene.from_yaml(example_path("material_libraries")).model_dump(mode="json")
    assert data["materials"]["ceiling_tiles"]["entry"] == 1001


def test_materials_command_lists_a_library(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["materials", "--file", str(ODEON), "door"]) == 0
    out = capsys.readouterr().out
    assert "1002 Wooden door, solid" in out and "1001" not in out
    assert "\nDoors\n" in out  # ODEON's category
    assert main(["materials", "--file", str(ODEON), "ceilings"]) == 0  # found by category
    assert "1001 Acoustic ceiling tiles" in capsys.readouterr().out
    assert main(["materials", "--file", str(CATT)]) == 0
    out = capsys.readouterr().out
    assert "  scattering" in out and "walls" in out
    assert main(["materials", "--file", str(CATT), "marble"]) == 1
    assert main(["materials", "--file", str(EXAMPLE / "missing.li8")]) == 1
