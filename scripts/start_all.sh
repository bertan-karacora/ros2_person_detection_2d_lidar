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
    echo "  ./start_all.sh [-h | --help] [<args_detection>]"
    echo
    echo "Start all processes."
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

start_tmux_detection() {
    local path_log="$HOME/.ros/log/$NAME_CONTAINER_ROS2_PERSON_DETECTION_2D_LIDAR.log"

    if [ -f "$path_log" ]; then
        >"$path_log"
        echo "Reset log file $path_log"
    else
        touch "$path_log"
        echo "Created log file $path_log"
    fi

    tmux new -d -s person_detection_2d_lidar "$path_repo/scripts/start_detection.sh" ${args_detection:+$args_detection}
    tmux pipe-pane -t person_detection_2d_lidar -o "cat >> $path_log"
}

attach_to_tmux_detection() {
    tmux a -t person_detection_2d_lidar
}

main() {
    parse_args "$@"
    "$path_repo/scripts/download_checkpoints.sh" "$path_repo/resources/checkpoints"
    "$path_repo/scripts/download_data.sh" "$path_repo/data"

    start_tmux_detection
    attach_to_tmux_detection
}

main "$@"
