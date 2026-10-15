"""Writes room.dae, the model of this example.

A rehearsal room of 26 x 20 ft under a pitched roof (walls 10 ft high, the ridge at
15 ft), written as SketchUp exports Collada: in inches with +z up, each surface a
geometry with its material, the gable walls as five-cornered polygons. Run it from this
folder: python make_model.py
"""

from __future__ import annotations

from pathlib import Path

#: The room in inches: length (x), width (y), wall height and ridge height.
LENGTH, WIDTH, WALL, RIDGE = 312.0, 240.0, 120.0, 180.0


def _surfaces():
    """Each surface's name, material and polygons (corners counter-clockwise seen from
    inside the room)."""
    x, y, h, r = LENGTH, WIDTH, WALL, RIDGE
    m = y / 2
    return [
        ("floor", "Wood_Floor_Light", [[(0, 0, 0), (x, 0, 0), (x, y, 0), (0, y, 0)]]),
        ("south wall", "Brick_Rough", [[(0, 0, 0), (0, 0, h), (x, 0, h), (x, 0, 0)]]),
        ("north wall", "Brick_Rough", [[(x, y, 0), (x, y, h), (0, y, h), (0, y, 0)]]),
        (
            "gable walls",
            "Brick_Rough",
            [
                [(0, y, h), (0, m, r), (0, 0, h), (0, 0, 0), (0, y, 0)],
                [(x, 0, h), (x, m, r), (x, y, h), (x, y, 0), (x, 0, 0)],
            ],
        ),
        (
            "roof",
            "Roof_Boards",
            [
                [(0, m, r), (x, m, r), (x, 0, h), (0, 0, h)],
                [(x, m, r), (0, m, r), (0, y, h), (x, y, h)],
            ],
        ),
    ]


def collada(surfaces, *, unit=("inch", 0.0254), up="Z_UP") -> str:
    """A Collada document with one geometry, and one node, per surface."""
    materials = sorted({material for _, material, _ in surfaces})
    geometries, nodes = [], []
    for k, (name, material, polygons) in enumerate(surfaces):
        corners = [c for polygon in polygons for c in polygon]
        numbers = " ".join(f"{v:g}" for corner in corners for v in corner)
        counts = " ".join(str(len(p)) for p in polygons)
        indices = " ".join(str(i) for i in range(len(corners)))
        geometries.append(
            f"""    <geometry id="ID{k}" name="{name}">
      <mesh>
        <source id="ID{k}-positions">
          <float_array id="ID{k}-array" count="{3 * len(corners)}">{numbers}</float_array>
          <technique_common>
            <accessor source="#ID{k}-array" count="{len(corners)}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="ID{k}-vertices">
          <input semantic="POSITION" source="#ID{k}-positions"/>
        </vertices>
        <polylist count="{len(polygons)}" material="surface">
          <input semantic="VERTEX" source="#ID{k}-vertices" offset="0"/>
          <vcount>{counts}</vcount>
          <p>{indices}</p>
        </polylist>
      </mesh>
    </geometry>"""
        )
        nodes.append(
            f"""      <node id="node{k}" name="{name}">
        <instance_geometry url="#ID{k}">
          <bind_material>
            <technique_common>
              <instance_material symbol="surface" target="#{material}ID"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>"""
        )
    effects = "\n".join(
        f'    <effect id="{m}-effect"><profile_COMMON><technique sid="common"><lambert/>'
        "</technique></profile_COMMON></effect>"
        for m in materials
    )
    library = "\n".join(
        f'    <material id="{m}ID" name="{m}"><instance_effect url="#{m}-effect"/></material>'
        for m in materials
    )
    newline = "\n"
    return f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <unit name="{unit[0]}" meter="{unit[1]}"/>
    <up_axis>{up}</up_axis>
  </asset>
  <library_effects>
{effects}
  </library_effects>
  <library_materials>
{library}
  </library_materials>
  <library_geometries>
{newline.join(geometries)}
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="SketchUp">
{newline.join(nodes)}
    </visual_scene>
  </library_visual_scenes>
  <scene>
    <instance_visual_scene url="#SketchUp"/>
  </scene>
</COLLADA>
"""


if __name__ == "__main__":
    path = Path(__file__).with_name("room.dae")
    path.write_text(collada(_surfaces()), encoding="utf-8")
    print(f"wrote {path}")
