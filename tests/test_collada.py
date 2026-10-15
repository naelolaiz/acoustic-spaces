"""Collada files (geometry.io's own reader)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room
from acoustic_spaces.examples import example_path
from acoustic_spaces.geometry import load_mesh

ROOM = example_path("rehearsal_room").parent / "room.dae"


def _make_model():
    spec = importlib.util.spec_from_file_location("make_model", ROOM.with_name("make_model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_rehearsal_room_in_inches(tmp_path: Path) -> None:
    mesh = load_mesh(ROOM)
    np.testing.assert_allclose(mesh.vertices.max(axis=0), (7.9248, 6.096, 4.572))
    assert mesh.labels[0] == ("Wood_Floor_Light", "floor", "ID0")
    assert len(mesh.faces) == 2 + 2 + 2 + 2 * 3 + 2 * 2  # the gables are five-cornered
    room = Room.from_yaml(example_path("rehearsal_room"))
    assert room.report.closed and room.report.normals_inward
    assert room.report.volume_m3 == pytest.approx(7.9248 * 6.096 * (3.048 + 1.524 / 2))
    # Assigned in lower case: names match whatever their case.
    assert room.problems() == [] and room.warnings() == []
    assert set(room.area_by_material()) == {"wood_floor", "brick", "boards"}
    # make_model.py writes the file as it is.
    model = _make_model()
    assert model.collada(model._surfaces()) == ROOM.read_text(encoding="utf-8")


@pytest.mark.parametrize(("unit", "up"), [(("metre", 1.0), "Z_UP"), (("cm", 0.01), "Y_UP")])
def test_unit_and_up_axis_from_the_file(tmp_path: Path, unit, up: str) -> None:
    s = 1.0 / unit[1]
    if up == "Z_UP":
        floor = [(0, 0, 0), (2 * s, 0, 0), (2 * s, 3 * s, 0), (0, 3 * s, 0)]
    else:  # y up: the floor in x and -z, so that it faces up
        floor = [(0, 0, 0), (2 * s, 0, 0), (2 * s, 0, -3 * s), (0, 0, -3 * s)]
    path = tmp_path / "floor.dae"
    path.write_text(_make_model().collada([("floor", "Oak", [floor])], unit=unit, up=up))
    mesh = load_mesh(path)
    np.testing.assert_allclose(mesh.vertices.max(axis=0), (2.0, 3.0, 0.0), atol=1e-9)
    assert mesh.face_areas().sum() == pytest.approx(6.0)
    normal = np.cross(*(mesh.vertices[mesh.faces[0, 1:]] - mesh.vertices[mesh.faces[0, 0]]))
    assert normal[2] > 0
    # The scene's units and up axis win: read as written, in metres with z up.
    raw = load_mesh(path, units="m", up_axis="z")
    assert np.ptp(raw.vertices, axis=0).max() == pytest.approx(3.0 * s)


NODES = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2008/03/COLLADASchema" version="1.5.0">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_materials><material id="felt-id" name="Felt"/></library_materials>
  <library_geometries>
    <geometry id="tile" name="tile">
      <mesh>
        <source id="tile-pos">
          <float_array id="tile-arr" count="12">0 0 0 1 0 0 1 1 0 0 1 0</float_array>
          <technique_common><accessor source="#tile-arr" count="4" stride="3"/></technique_common>
        </source>
        <source id="tile-normals"><float_array id="n" count="3">0 0 1</float_array></source>
        <vertices id="tile-v"><input semantic="POSITION" source="#tile-pos"/></vertices>
        <triangles count="2" material="m">
          <input semantic="VERTEX" source="#tile-v" offset="0"/>
          <input semantic="NORMAL" source="#tile-normals" offset="1"/>
          <p>0 0 1 0 2 0 0 0 2 0 3 0</p>
        </triangles>
        <polygons count="1">
          <input semantic="VERTEX" source="#tile-v" offset="0"/>
          <ph><p>0 1 2 3</p><h>0 1 2</h></ph>
        </polygons>
        <lines count="1"><input semantic="VERTEX" source="#tile-v" offset="0"/><p>0 1</p></lines>
      </mesh>
    </geometry>
  </library_geometries>
  <library_nodes>
    <node id="component">
      <instance_geometry url="#tile">
        <bind_material><technique_common>
          <instance_material symbol="m" target="#felt-id"/>
        </technique_common></bind_material>
      </instance_geometry>
    </node>
  </library_nodes>
  <library_visual_scenes>
    <visual_scene id="scene">
      <node><translate>10 0 0</translate><instance_node url="#component"/></node>
      <node>
        <translate>0 5 0</translate><rotate>0 0 1 90</rotate><scale>2 2 2</scale>
        <instance_node url="#component"/>
      </node>
      <node>
        <matrix>-1 0 0 0 0 1 0 0 0 0 1 3 0 0 0 1</matrix><instance_node url="#component"/>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
"""


def test_nodes_place_instanced_components(tmp_path: Path) -> None:
    path = tmp_path / "nodes.dae"
    path.write_text(NODES)
    mesh = load_mesh(path)
    # Triangles with another input after the vertex, and a polygon with a hole, which
    # is read without it; the line is left out. No unit: metres.
    assert mesh.labels == [("Felt", "tile"), ("tile",)]
    assert len(mesh.faces) == 3 * (2 + 2)
    corners = mesh.vertices[mesh.faces]
    middles = corners.mean(axis=1)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    # Moved 10 m along x; turned a quarter about z and doubled, 5 m along y; mirrored in
    # x and raised 3 m, its faces still facing up.
    first, second, third = np.split(np.arange(len(mesh.faces)), [4, 8])
    assert np.all((middles[first, 0] > 10) & (middles[first, 0] < 11))
    np.testing.assert_allclose(mesh.face_areas()[second], 2.0)  # halves of 4 m2 squares
    assert np.all((middles[second, 0] < 0) & (middles[second, 1] > 5))
    assert np.all((middles[third, 0] < 0) & (middles[third, 2] == 3.0))
    assert np.all(normals[:, 2] > 0)


def test_collada_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.dae"
    bad.write_text("<COLLADA")
    with pytest.raises(ValueError, match="not a readable Collada file"):
        load_mesh(bad)
    bad.write_text("<svg/>")
    with pytest.raises(ValueError, match=r"not a Collada file \(it holds <svg>\)"):
        load_mesh(bad)
    bad.write_text('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"/>')
    with pytest.raises(ValueError, match="no faces found"):
        load_mesh(bad)
