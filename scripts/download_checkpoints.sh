#!/usr/bin/env bash

set -euo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/env.sh"

readonly url_dr_spaam="https://drive.google.com/uc?id=1gH_pdxUC8diIFUe3_Ky8QiVVCyiR3500"
readonly url_drow3="https://drive.google.com/uc?id=1QuB9mwm8h46vIiynFsDKKxlEiUYC0quN"
readonly url_dr_spaam_bev="https://drive.google.com/uc?id=14JpNtIFOGu2i4rIzIx9RVMYm4hmIjuO1"
path_dir_checkpoints=""

show_help() {
    echo "Usage:"
    echo "  ./download_checkpoints.sh [-h | --help] <path_dir_checkpoints>"
    echo
    echo "Download checkpoints of pretrained models to <path_dir_checkpoints>."
    echo
}

parse_args() {
    local arg=""
    while [[ "$#" -gt 0 ]]; do
        arg="$1"
        shift
        case "$arg" in
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$path_dir_checkpoints" ]]; then
                path_dir_checkpoints="$arg"
            else
                echo "Unexpected positional argument"
                exit 1
            fi
            ;;
        esac
    done
}

download_checkpoints() {
    mkdir --parents "$path_dir_checkpoints"

    cd "$path_dir_checkpoints"
    download_checkpoint_dr_spaam
    download_checkpoint_drow3
    download_checkpoint_dr_spaam_bev
}

download_checkpoint_dr_spaam() {
    local path_checkpoint="$path_dir_checkpoints/ckpt_jrdb_ann_dr_spaam_e20.pth"

    if [ ! -f "$path_checkpoint" ]; then
        echo "Downloading to $path_dir_checkpoints ..."
        gdown "$url_dr_spaam"
    else
        echo "$path_checkpoint already exists"
    fi
}

download_checkpoint_drow3() {
    local path_checkpoint="$path_dir_checkpoints/ckpt_jrdb_ann_drow3_e40.pth"

    if [ ! -f "$path_checkpoint" ]; then
        echo "Downloading to $path_dir_checkpoints ..."
        gdown "$url_drow3"
    else
        echo "$path_checkpoint already exists"
    fi
}

download_checkpoint_dr_spaam_bev() {
    local path_checkpoint="$path_dir_checkpoints/jrdb_dr_spaam_with_bev_box_e20.pth"

    if [ ! -f "$path_checkpoint" ]; then
        echo "Downloading to $path_dir_checkpoints ..."
        gdown "$url_dr_spaam_bev"
    else
        echo "$path_checkpoint already exists"
    fi
}

main() {
    parse_args "$@"
    download_checkpoints
}

main "$@"
