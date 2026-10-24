# The command line in a container, to run packs (acoustic-spaces pack) on a server or a
# cloud job: headless, on the machine's GPU through Vulkan when it has one, else on its
# CPU cores. See docs/remote.md.
#
#   docker build -t acoustic-spaces .
#   docker run --rm --gpus all acoustic-spaces info
#   docker run --rm --gpus all -v "$PWD:/work" --user "$(id -u):$(id -g)" \
#       acoustic-spaces simulate hall.zip --out results

FROM ubuntu:24.04 AS build
# No questions from apt, nor notes on the terminal it would ask them in.
ARG DEBIAN_FRONTEND=noninteractive
# Ubuntu's g++-13 itself: the g++ package's c++ link warns that the image has no manual
# pages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++-13 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*
ENV CXX=g++-13
RUN python3 -m venv /opt/build
WORKDIR /src
COPY . .
# CMake, Ninja and nanobind come from PyPI as build requirements.
RUN /opt/build/bin/pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
# The Vulkan loader and Mesa's drivers: AMD and Intel GPUs, and lavapipe, a software
# renderer that device auto never picks. NVIDIA's driver comes from the host.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv libvulkan1 \
        mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*
# The optional parts installed with it: gpu (wgpu) and sofa (h5py) by default; add ifc
# or sketchup for packs whose model is an IFC or SketchUp file.
ARG EXTRAS=gpu,sofa
COPY --from=build /wheels /wheels
RUN python3 -m venv /opt/acoustic-spaces \
    && /opt/acoustic-spaces/bin/pip install --no-cache-dir "$(ls /wheels/*.whl)[${EXTRAS}]" \
    && rm -rf /wheels
# NVIDIA's container toolkit mounts the driver's Vulkan part only with the graphics
# capability; many CUDA images ask for compute alone.
ENV PATH=/usr/local/bin:/opt/acoustic-spaces/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    XDG_CACHE_HOME=/tmp/cache \
    XDG_CONFIG_HOME=/tmp/config
COPY docker/acoustic-spaces.sh /usr/local/bin/acoustic-spaces
# Ubuntu's image has a user with id 1000, as Hugging Face's Docker Spaces expect.
USER ubuntu
WORKDIR /work
ENTRYPOINT ["acoustic-spaces"]
CMD ["info"]
