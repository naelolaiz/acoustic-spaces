# Measured rooms from BRAS

Four rooms of the *Benchmark for Room Acoustical Simulation* (BRAS), as scenes this
package simulates, with the parameters of the responses measured in them:

| Folder | Room | Volume | Surfaces |
|---|---|---:|---:|
| `CR1_DoorAngle1` | coupled rooms, a laboratory and a reverberation chamber, RWTH Aachen, the door between them open by 4.1° | 225 m³ | 4 materials |
| `CR1_DoorAngle3` | the same, the door open by 30.4° | not closed | 4 materials |
| `CR2` | seminar room, TU Berlin | 146 m³ | 5 materials |
| `CR3` | chamber music hall, Konzerthaus Berlin | 3331 m³ | 7 materials |
| `CR4` | auditorium, TU Berlin | 8657 m³ | 8 materials |

In `CR1_DoorAngle3` the door leaf cuts some 2 cm into the wall at its hinge, so the
model has no closed inside and no volume; it is simulated as an open model or not at
all, and its triangles close the laboratory's wall in `CR1_DoorAngle1` (see below).

`validation/bras_validation.py` simulates them and compares the results with the
measurements, and `validation/bras_binaural.py` compares the IACC of binaural responses
in CR2, CR3 and CR4; the report is in `docs/validation.md`.

## Source and license

Lukas Aspöck, Fabian Brinkmann, David Ackermann, Stefan Weinzierl and Michael Vorländer:
*BRAS – Benchmark for Room Acoustical Simulation*, TU Berlin and RWTH Aachen, 2020,
version 3, [DOI 10.14279/depositonce-6726.3](https://doi.org/10.14279/depositonce-6726.3).

The BRAS data is published under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and so are the files in
the room folders made from it: the OBJ models, the material CSVs, `scene.yaml`,
`measured.json` and `binaural.json`, and `QSC_K8.csv` here. Share them under the same license and keep the credit above. They are
not covered by the GPL of the rest of the repository; the two scripts here are.

## What the files are

Each room folder holds:

- `CRx.obj`: the room model, converted from the BRAS SketchUp file by `convert_skp.py`
  (metres, +z up, in the coordinates of the BRAS source and receiver positions), one
  `usemtl` per BRAS material.
- `materials/fitted/*.csv` and `materials/initial/*.csv`: the BRAS surface data as
  published, in third octaves (absorption, then scattering). *Fitted* is the absorption
  BRAS scaled, by one factor per third octave for all the materials of a room (in CR1
  for each of its two rooms), until Eyring's formula gave the measured reverberation
  time; *initial* is its estimate before that, from tables and measurements of the
  materials.
- `scene.yaml`: the room with the fitted materials, the temperature and humidity during
  the measurements, the two positions of the dodecahedron loudspeaker (`LS1`, `LS2`) and
  the five microphone positions (`MP1` to `MP5`; `MP3` and `MP4` in CR1, both in the
  laboratory, with `LS2`, while `LS1` stands in the reverberation chamber).
- `measured.json`: for each of the ten loudspeaker and microphone pairs (four in CR1),
  the octave-band ISO 3382-1 parameters of the measured response, analysed by
  `acoustic_spaces.parameters.from_impulse_response`; and the spatial averages BRAS
  publishes (third octaves, D50 as a fraction; none for CR1), to check that analysis
  against. G is missing, as the responses have no free-field reference of their own;
  `level_db` is each band's level instead, whose differences between the pairs are
  those of G.

- `binaural.json` (CR2, CR3 and CR4): the setup of the binaural measurements, the head
  and torso simulator FABIAN facing loudspeaker `LS7` and the five QSC K8 loudspeakers
  `LS3` to `LS7`, one at a time (positions from the SOFA file of the binaural
  responses, the loudspeakers' tweeters; yaw and pitch from the labels of
  `CRx_BRIR.skp`), and IACC_E and IACC_L of each measured response with FABIAN's head
  straight on the torso, analysed by
  `acoustic_spaces.parameters.interaural_cross_correlation`.

And in this folder, `QSC_K8.csv`: the QSC K8's directivity as this package reads a
directivity CSV, the magnitudes of BRAS's third-octave spectra every 5 degrees (the
on-axis response included, as published).

The measured responses themselves stay out of the repository: those of the
dodecahedron are SOFA files of 6 to 11 MB per room, in scene archives of 389 to 785
MB. So do FABIAN's head-related impulse responses (two CSV files of 173 MB for the
head straight on the torso, in an archive of 4.7 GB), which
`validation/bras_binaural.py` reads from the BRAS folder.

## Remaking the files

Download and unpack `1_scene_descriptions-CR1.zip` to `1_scene_descriptions-CR4.zip`,
`3_surface_descriptions.zip`, `4_additional_data.zip` and
`2_source_and_receiver_descriptions-QSC_K8.zip` from the
[BRAS page](https://depositonce.tu-berlin.de/handle/11303/7506.3) into one folder, then:

```sh
pip install openskp                     # the sketchup extra: reads SketchUp files
CR1="BRAS/1 Scene descriptions/CR1 coupled rooms (laboratory and reverberation chamber)/Geometry/rooms"
python validation/bras/convert_skp.py "$CR1/CR1_DoorAngle3.skp" validation/bras/CR1_DoorAngle3/CR1_DoorAngle3.obj
python validation/bras/convert_skp.py "$CR1/CR1_DoorAngle1.skp" validation/bras/CR1_DoorAngle1/CR1_DoorAngle1.obj \
    --fill-from validation/bras/CR1_DoorAngle3/CR1_DoorAngle3.obj
python validation/bras/convert_skp.py "BRAS/1 Scene descriptions/CR2 small room (seminar room)/Geometry/CR2.skp" validation/bras/CR2/CR2.obj
python validation/bras/convert_skp.py "BRAS/1 Scene descriptions/CR3 medium room (chamber music hall)/Geometry/CR3.skp" validation/bras/CR3/CR3.obj
python validation/bras/convert_skp.py "BRAS/1 Scene descriptions/CR4 large room (auditorium)/Geometry/CR4.skp" validation/bras/CR4/CR4.obj
python validation/bras/prepare.py BRAS   # every scene; needs the sofa extra (h5py)
```

`validation/bras_binaural.py` also needs
`2_source_and_receiver_descriptions-FABIAN_HRIRs.zip`, unpacked into the same folder
with the `Fabian HRIRs.zip` inside it unpacked where it lies, so that
`2 Source and receiver descriptions/FABIAN HRIRs/CSV-files` holds the
`HATO_0_1x1_64442_HRIR_L_top_pole.csv` and `..._R_top_pole.csv` it reads.

`convert_skp.py` explains what it repairs in the models on the way.

## Left out

- The scenes of single reflections and diffraction (RS1 to RS11): the comparison is
  with whole rooms.
- The binaural responses with the head turned (views other than 0): the binaural
  comparison is of IACC with the head straight.
- Which way up the loudspeakers stood: the notes in `CRx_BRIR_loudspeaker_setup.skp`
  say they "were placed upside down in three of four cases" without saying which, so
  all are simulated as their directivity was measured.
