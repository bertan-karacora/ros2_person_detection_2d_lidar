#!/usr/bin/env bash

set -eo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/env.sh"

source "/opt/ros/$DISTRIBUTION_ROS/setup.bash"
source "$HOME/colcon_ws/install/setup.bash"

set -u

args_detection=""

show_help() {
    echo "Usage:"
    echo "  ./start_detection.sh [-h | --help] [<args_detection>]"
    echo
    echo "Start detection."
    echo
}

parse_args() {
    local arg=""
    while [[ "$#" -gt 0 ]]; do
        arg="$1"
        shift
        case $arg in
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$args_detection" ]]; then
                args_detection="$arg"
            else
                args_detection="$args_detection $arg"
            fi
            ;;
        esac
    done
}

start_detection() {
    ros2 launch ros2_person_detection_2d_lidar person_detection_2d_lidar_launch.py ${args_detection:+$args_detection}
}

main() {
    parse_args "$@"
    start_detection
}

main "$@"
