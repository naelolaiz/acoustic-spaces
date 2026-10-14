# Measured rooms from BRAS

Two rooms of the *Benchmark for Room Acoustical Simulation* (BRAS), as scenes this
package simulates, with the parameters of the responses measured in them:

| Folder | Room | Volume | Surfaces |
|---|---|---:|---:|
| `CR2` | seminar room, TU Berlin | 146 m³ | 5 materials |
| `CR3` | chamber music hall, Konzerthaus Berlin | 3331 m³ | 7 materials |

`validation/bras_validation.py` simulates them and compares the results with the
measurements; the report is in `docs/validation.md`.

## Source and license

Lukas Aspöck, Fabian Brinkmann, David Ackermann, Stefan Weinzierl and Michael Vorländer:
*BRAS – Benchmark for Room Acoustical Simulation*, TU Berlin and RWTH Aachen, 2020,
version 3, [DOI 10.14279/depositonce-6726.3](https://doi.org/10.14279/depositonce-6726.3).

The BRAS data is published under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and so are the files in
`CR2/` and `CR3/` made from it: the OBJ models, the material CSVs, `scene.yaml` and
`measured.json`. Share them under the same license and keep the credit above. They are
not covered by the GPL of the rest of the repository; the two scripts here are.

## What the files are

Each room folder holds:

- `CRx.obj`: the room model, converted from the BRAS SketchUp file by `convert_skp.py`
  (metres, +z up, in the coordinates of the BRAS source and receiver positions), one
  `usemtl` per BRAS material.
- `materials/fitted/*.csv` and `materials/initial/*.csv`: the BRAS surface data as
  published, in third octaves (absorption, then scattering). *Fitted* is the absorption
  BRAS adjusted until a simulation matched the measured reverberation time; *initial*
  is its estimate before that, from tables and measurements of the materials.
- `scene.yaml`: the room with the fitted materials, the temperature and humidity during
  the measurements, the two positions of the dodecahedron loudspeaker (`LS1`, `LS2`) and
  the five microphone positions (`MP1` to `MP5`).
- `measured.json`: for each of the ten loudspeaker and microphone pairs, the octave-band
  ISO 3382-1 parameters of the measured response, analysed by
  `acoustic_spaces.parameters.from_impulse_response`; and the spatial averages BRAS
  publishes (third octaves, D50 as a fraction), to check that analysis against. G is
  missing: the responses have no free-field reference.

The measured responses themselves (SOFA files of about 600 MB per room) stay out of the
repository.

## Remaking the files

Download and unpack `1_scene_descriptions-CR2.zip`, `1_scene_descriptions-CR3.zip`,
`3_surface_descriptions.zip` and `4_additional_data.zip` from the
[BRAS page](https://depositonce.tu-berlin.de/handle/11303/7506.3) into one folder, then:

```sh
pip install openskp                     # the sketchup extra: reads SketchUp files
python validation/bras/convert_skp.py "BRAS/1 Scene descriptions/CR2 small room (seminar room)/Geometry/CR2.skp" validation/bras/CR2/CR2.obj
python validation/bras/convert_skp.py "BRAS/1 Scene descriptions/CR3 medium room (chamber music hall)/Geometry/CR3.skp" validation/bras/CR3/CR3.obj
python validation/bras/prepare.py BRAS   # needs the sofa extra (h5py)
```

`convert_skp.py` explains what it repairs in the models on the way.

## Left out

- **CR4** (the auditorium): openskp cannot read its SketchUp file, whose text labels hang
  on a component, and no other free reader could either. Exported to OBJ from SketchUp
  itself, it would fit in here the same way.
- **CR1** (a box-shaped room with a single reflector) and the scenes of single
  reflections and diffraction (RS1 to RS11): the first comparison is with whole rooms.
- The responses measured with the dummy head and the binaural ones: the comparison is of
  ISO 3382-1 parameters, which need the omnidirectional microphone.
