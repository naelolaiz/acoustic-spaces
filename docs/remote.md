# Running on another computer

Long runs (many rays, third-octave bands, the wave solver up to 250 or 500 Hz, big
coverage maps, outdoor scenes over kilometres) can run on a computer with more cores or
a GPU: a server you reach over SSH, a rented GPU machine, or a job on a cloud service
such as Hugging Face Jobs. The scene goes there as a *pack*, one zip file; a container
image runs the command line on the GPU there; the results come back as the same files
`simulate` and `map` write here.

Nothing in acoustic_spaces uploads anything: you choose where a pack goes, and it holds
everything needed to rebuild the scene (its model, materials and positions). Send it
only where you would send the drawings themselves.

## Packs

```sh
acoustic-spaces pack hall/scene.yaml hall.zip
```

writes the scene and every file it names: its model, terrain, buildings map, material
libraries and coefficient files, directivity files, the WAV files sources play and HRTFs,
with the files those name in turn (a `.gltf` model's buffers and images, a loudspeaker
array's element data). It lists what went in and their sizes. Files under the scene's folder keep their place
relative to it; files elsewhere go under `external/<their folder>/`. The scene file in
the pack names them all relatively, so the pack runs wherever it is unpacked. The
editor does the same from **File > Pack for another computer**, also for a scene not yet
saved.

A scene with problems (a listener outside the room, surfaces without a material) is
not packed, as `simulate` would not run it either; warnings are shown. A file that
names one outside its own folder (`../shared/element.csv` in an array's YAML) cannot
keep its place in a pack: move that file next to it.

Every command that reads a scene takes a pack in its place: `simulate`, `map`,
`validate`, `inspect`, `calibrate`, `walkthrough`, `export`, `view`, `animate` and
`tune`, with the same options.

```sh
acoustic-spaces simulate hall.zip --out results --bands third_octave   # on the other computer
```

The pack is unpacked into a temporary folder for the command and removed after it.
`pack.json` in it records the version that wrote it and a SHA-256 of every file, so a
damaged pack is refused, and a pack written by another version runs with a warning:
results can differ between versions, so for numbers to compare, run the same version on
both computers. Runs of a pack give the same results, bit for bit, as runs of the scene
file on the same computer.

## The container image

The repository's `Dockerfile` builds an image with the command line, the `gpu` and
`sofa` extras, the Vulkan loader and Mesa's Vulkan drivers, on Ubuntu 24.04. It runs
headless: the wave solver and the late rays reach the GPU through Vulkan (wgpu), with no
display. Build it from a checkout:

```sh
docker build -t acoustic-spaces .
docker build -t acoustic-spaces --build-arg EXTRAS=gpu,sofa,ifc .   # packs with IFC models
```

Its entry point is `acoustic-spaces`, so the container takes a command and its options;
without one it runs `info`. First check that it finds the GPU:

```sh
docker run --rm --gpus all acoustic-spaces info
```

`GPU:` should name the card, `(Vulkan, DiscreteGPU)`. With `llvmpipe ... (Vulkan, CPU)`,
Mesa's software driver, or `none`, the GPU's Vulkan driver does not reach the container
and every run takes the CPU:

- NVIDIA: the host needs the NVIDIA driver and the NVIDIA Container Toolkit, and the
  container the driver's *graphics* capability, which holds its Vulkan part. The image
  asks for it (`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`); a service that
  sets `compute` alone, as many CUDA images do, leaves Vulkan out. Older toolkits mount
  the library without telling the Vulkan loader where it is; the image's
  `acoustic-spaces` command finds it and writes that file itself.
- AMD and Intel: Mesa's drivers are in the image; pass the GPU's device nodes with
  `--device /dev/dri` and, where they belong to the `render` group,
  `--group-add "$(getent group render | cut -d: -f3)"`.

On the CPU, runs take every core (`simulation.performance.threads: 0`), the same as
here. Device `auto` times the CPU against the GPU on the machine it runs on (see
[Simulation](simulation.md)); a container keeps no timings between runs, so each run
times anew, unless the scene or the options name `cpu` or `gpu`, or the cache folder
(`/tmp/cache`) is a volume that `acoustic-spaces tune` filled once.

Run a pack with the current folder mounted as `/work`, the container's working folder,
writing as your own user:

```sh
docker run --rm --gpus all -v "$PWD:/work" --user "$(id -u):$(id -g)" \
    acoustic-spaces simulate hall.zip --out results
```

## A server over SSH

On any machine you reach over SSH (your own, or one rented by the hour with a GPU),
build the image there from a checkout, or send it from here:

```sh
docker save acoustic-spaces | ssh server docker load
scp hall.zip server:jobs/
ssh server 'cd jobs && docker run --rm --gpus all -v "$PWD:/work" \
    --user "$(id -u):$(id -g)" acoustic-spaces simulate hall.zip --out hall-results'
scp -r server:jobs/hall-results .
```

Without Docker, install acoustic_spaces on the server as here (with the `gpu` extra,
and a Vulkan driver for its GPU) and run the pack the same way.

## Hugging Face Jobs

[Jobs](https://huggingface.co/docs/hub/jobs-quickstart) run a container on Hugging
Face's hardware, CPUs or NVIDIA GPUs from a T4 to an H200, paid by the minute from the
account's credit (`hf jobs hardware` lists them and their prices). They take a public
image or the image a Docker Space builds; a private Space keeps the image private. With
the `hf` command (`pip install -U huggingface_hub`, then `hf auth login`), once:

```sh
hf repos create <user>/acoustic-spaces --type space --space-sdk docker --private
mkdir space && git archive HEAD | tar -x -C space
printf -- '---\ntitle: acoustic-spaces\nsdk: docker\n---\n\n' | cat - README.md > space/README.md
hf upload <user>/acoustic-spaces space . --type space
```

The Space builds the image from the `Dockerfile`; it then shows a runtime error, as it
serves no web page, which does not matter: Jobs only take its image. Upload again after
updating the checkout, and Jobs take the new build. Then, for each run:

```sh
mkdir -p jobs && cp hall.zip jobs/
hf jobs run --flavor l4x1 --timeout 2h -v ./jobs:/jobs:rw \
    hf.co/spaces/<user>/acoustic-spaces \
    acoustic-spaces simulate /jobs/hall.zip --out /jobs/results
```

The folder `jobs` is copied to a storage bucket of the account and mounted in the job,
which writes the results there; the command prints how to sync the bucket's folder back
when the job starts. Check the GPU first with a short job running `acoustic-spaces info`;
a job stops at its timeout (30 minutes unless `--timeout` says otherwise) and is billed
while it starts and runs.
