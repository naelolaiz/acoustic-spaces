#!/usr/bin/env bash
# Turn a mix into a video (for YouTube, say): a still picture with a scrolling spectrogram
# of the sound in the band at its bottom. Needs ffmpeg.
#
#   bash make_video.sh MIX [COVER] [OUT]
#
# MIX    the WAV written by auralize.sh
# COVER  the picture, 1920 x 1080 (default cover.png, as make_images.py writes it)
# OUT    the video to write (default: MIX with .mp4 for .wav)
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,9p' "$0"
    exit 2
fi
mix=$1
cover=${2:-cover.png}
out=${3:-$(basename "${mix%.wav}").mp4}

ffmpeg -loop 1 -framerate 30 -i "$cover" -i "$mix" \
    -filter_complex "[1:a]showspectrum=s=1720x512:slide=scroll:color=magma:scale=log:fscale=log:start=40:stop=16000:drange=80:overlap=0.5:fps=30,scale=1720:96,format=rgba,colorkey=0x000000:0.08:0.25[s];[0:v][s]overlay=100:892:shortest=1,format=yuv420p[v]" \
    -map "[v]" -map 1:a -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 384k \
    -movflags +faststart "$out"
