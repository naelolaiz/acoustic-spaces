#!/usr/bin/env bash
# Mix the 61 anechoic tracks of one movement as heard at one seat of scene.yaml.
#
#   bash auralize.sh RESULTS TRACKS [LISTENER] [FORMAT] [OUT]
#
# RESULTS  folder written by `acoustic-spaces simulate scene.yaml --out RESULTS`
# TRACKS   a movement's `anechoic` folder: unzip movement1.zip (or 2, 4) from
#          https://doi.org/10.14279/depositonce-6729.2 into movement1, say
# LISTENER conductor, stalls-front, stalls-middle (default) or stalls-rear
# FORMAT   binaural (default) or mono, as the scene writes them
# OUT      the WAV to write (default beethoven8_<movement>_<listener>.<format>.wav)
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,11p' "$0"
    exit 2
fi
results=$1
tracks=$2
listener=${3:-stalls-middle}
format=${4:-binaural}
folder=$(cd "$tracks" && pwd)
movement=$(basename "$folder")
if [ "$movement" = anechoic ]; then
    movement=$(basename "$(dirname "$folder")")
fi
out=${5:-beethoven8_${movement}_${listener}.${format}.wav}

if [ ! -f "$results/parameters.json" ]; then
    echo "error: $results holds no simulation results (no parameters.json); run first:" >&2
    echo "  acoustic-spaces simulate $(dirname "$0")/scene.yaml --out $results" >&2
    exit 1
fi

# source name = track file
pairs=(
    fl1=01_Fl_1.wav
    fl2=02_Fl_2.wav
    ob1=03_Ob_1.wav
    ob2=04_Ob_2.wav
    cl1=05_Cl_1.wav
    cl2=06_Cl_2.wav
    bn1=07_Bn_1.wav
    bn2=08_Bn_2.wav
    hn1=09_Hrn_1.wav
    hn2=10_Hrn_2.wav
    tp1=11_Trp_1.wav
    tp2=12_Trp_2.wav
    timpani=13_Tp.wav
    vl1_01=14_Vl_1_1_1.wav
    vl1_02=15_Vl_1_1_2.wav
    vl1_03=16_Vl_1_1_3.wav
    vl1_04=17_Vl_1_1_4.wav
    vl1_05=18_Vl_1_1_5.wav
    vl1_06=19_Vl_1_1_6.wav
    vl1_07=20_Vl_1_1_7.wav
    vl1_08=21_Vl_1_2_1.wav
    vl1_09=22_Vl_1_2_2.wav
    vl1_10=23_Vl_1_2_3.wav
    vl1_11=24_Vl_1_2_4.wav
    vl1_12=25_Vl_1_2_5.wav
    vl1_13=26_Vl_1_2_6.wav
    vl1_14=27_Vl_1_2_7.wav
    vl2_01=28_Vl_2_1_1.wav
    vl2_02=29_Vl_2_1_2.wav
    vl2_03=30_Vl_2_1_3.wav
    vl2_04=31_Vl_2_1_4.wav
    vl2_05=32_Vl_2_1_5.wav
    vl2_06=33_Vl_2_1_6.wav
    vl2_07=34_Vl_2_2_1.wav
    vl2_08=35_Vl_2_2_2.wav
    vl2_09=36_Vl_2_2_3.wav
    vl2_10=37_Vl_2_2_4.wav
    vl2_11=38_Vl_2_2_5.wav
    vla_01=39_Vla_1_1.wav
    vla_02=40_Vla_1_2.wav
    vla_03=41_Vla_1_3.wav
    vla_04=42_Vla_1_4.wav
    vla_05=43_Vla_1_5.wav
    vla_06=44_Vla_1_6.wav
    vla_07=45_Vla_2_1.wav
    vla_08=46_Vla_2_2.wav
    vla_09=47_Vla_2_3.wav
    vla_10=48_Vla_2_4.wav
    vla_11=49_Vla_2_5.wav
    vc_1=50_Vc_1.wav
    vc_2=51_Vc_2.wav
    vc_3=52_Vc_3.wav
    vc_4=53_Vc_4.wav
    vc_5=54_Vc_5.wav
    vc_6=55_Vc_6.wav
    db_1=56_Db_1.wav
    db_2=57_Db_2.wav
    db_3=58_Db_3.wav
    db_4=59_Db_4.wav
    db_5=60_Db_5.wav
    db_6=61_Db_6.wav
)

signals=()
for pair in "${pairs[@]}"; do
    signals+=("${pair%%=*}=$tracks/${pair#*=}")
done
acoustic-spaces auralize "$results" "${signals[@]}" \
    --listener "$listener" --format "$format" --normalize --out "$out"
