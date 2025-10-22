#!/usr/bin/env bash

set -euo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/env.sh"

readonly url_drowv2="https://github.com/VisualComputingInstitute/DROW/releases/download/v2/DROWv2-data.zip"
path_dir_data=""

show_help() {
    echo "Usage:"
    echo "  ./download_data.sh [-h | --help] <path_dir_data>"
    echo
    echo "Download data to <path_dir_data>."
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
            if [[ -z "$path_dir_data" ]]; then
                path_dir_data="$arg"
            else
                echo "Unexpected positional argument"
                exit 1
            fi
            ;;
        esac
    done
}

download_data() {
    mkdir --parents "$path_dir_data"

    download_data_drowv2
}

download_data_drowv2() {
    local path_dir_drowv2="$path_dir_data/DROWv2"

    if [ ! -d "$path_dir_drowv2" ]; then
        echo "Downloading to $path_dir_drowv2 ..."
        wget "$url_drowv2" --directory-prefix "$path_dir_data"
        unzip data/DROWv2-data.zip -d data
        mv "$path_dir_data/DROWv2-data" "$path_dir_drowv2"
        rm "$path_dir_data/DROWv2-data.zip"
    else
        echo "$path_dir_drowv2 already exists"
    fi
}

main() {
    parse_args "$@"
    download_data
}

main "$@"
