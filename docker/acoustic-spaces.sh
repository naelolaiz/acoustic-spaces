#!/bin/sh
# acoustic-spaces in the container, ahead of the installed one on the PATH, so that it
# runs whether a service starts the image with its arguments or with a whole command
# (acoustic-spaces simulate ...).
#
# NVIDIA's container toolkit mounts the driver's Vulkan library (with the graphics
# capability) but, in older versions, not the file telling the Vulkan loader where it
# is: write one when the library is there and no such file is, so wgpu finds the GPU.
set -e
if [ "$1" = acoustic-spaces ]; then
    shift
fi
if [ -z "${VK_ICD_FILENAMES}${VK_DRIVER_FILES}" ] \
    && ! ls /usr/share/vulkan/icd.d/nvidia_icd*.json /etc/vulkan/icd.d/nvidia_icd*.json \
        >/dev/null 2>&1 \
    && ldconfig -p 2>/dev/null | grep -q 'libGLX_nvidia\.so\.0 '; then
    icd="${TMPDIR:-/tmp}/nvidia_icd.json"
    printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}\n' >"$icd"
    export VK_ADD_DRIVER_FILES="$icd${VK_ADD_DRIVER_FILES:+:$VK_ADD_DRIVER_FILES}"
fi
exec /opt/acoustic-spaces/bin/acoustic-spaces "$@"
