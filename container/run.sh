#!/usr/bin/env bash

set -euo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/env.sh"

readonly name_container="$NAME_CONTAINER_ROS2_PERSON_DETECTION_2D_LIDAR"
command=""
use_detach=1
is_found_overlay_maps=""
is_found_overlay_markers_server=""
is_found_overlay_ros2_config=""
is_found_overlay_ros2_interfaces=""
is_found_overlay_ros2_utils=""

show_help() {
    echo "Usage:"
    echo "  ./run.sh [-h | --help] [-a | --use_attach] [<command>]"
    echo
    echo "Run the container."
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
        -a | --use_attach)
            use_detach=""
            ;;
        *)
            if [[ -z "$command" ]]; then
                command="$arg"
            else
                command="$command $arg"
            fi
            ;;
        esac
    done
}

check_overlays() {
    if [[ ! -z "$PATH_MAPS" ]]; then
        if [[ -d "$PATH_MAPS" ]]; then
            is_found_overlay_maps=1
            echo "Overlaying data at $PATH_MAPS"
        else
            echo "Data at $PATH_MAPS not found"
        fi
    fi
    if [[ ! -z "$PATH_MARKERS_SERVER" ]]; then
        if [[ -d "$PATH_MARKERS_SERVER" ]]; then
            is_found_overlay_markers_server=1
            echo "Overlaying repo at $PATH_MARKERS_SERVER"
        else
            echo "Repo at $PATH_MARKERS_SERVER not found"
        fi
    fi
    if [[ ! -z "$PATH_ROS2_CONFIG" ]]; then
        if [[ -d "$PATH_ROS2_CONFIG" ]]; then
            is_found_overlay_ros2_config=1
            echo "Overlaying repo at $PATH_ROS2_CONFIG"
        else
            echo "Repo at $PATH_ROS2_CONFIG not found"
        fi
    fi
    if [[ ! -z "$PATH_ROS2_INTERFACES" ]]; then
        if [[ -d "$PATH_ROS2_INTERFACES" ]]; then
            is_found_overlay_ros2_interfaces=1
            echo "Overlaying repo at $PATH_ROS2_INTERFACES"
        else
            echo "Repo at $PATH_ROS2_INTERFACES not found"
        fi
    fi
    if [[ ! -z "$PATH_ROS2_UTILS" ]]; then
        if [[ -d "$PATH_ROS2_UTILS" ]]; then
            is_found_overlay_ros2_utils=1
            echo "Overlaying repo at $PATH_ROS2_UTILS"
        else
            echo "Repo at $PATH_ROS2_UTILS not found"
        fi
    fi
}

run_container() {
    local arch="$(arch)"
    local name_repo="$(basename "$path_repo")"

    docker run \
        --name "$name_container" \
        --shm-size 12G \
        --gpus all \
        --ipc host \
        --interactive \
        --tty \
        --net host \
        --rm \
        --env DISPLAY \
        ${use_detach:+"--detach"} \
        --volume /etc/localtime:/etc/localtime:ro \
        --volume /tmp/.X11-unix/:/tmp/.X11-unix/:ro \
        --volume "$HOME/.Xauthority:/home/$USER/.Xauthority:ro" \
        --volume "$HOME/.ros/:/home/$USER/.ros/" \
        --volume "$path_repo:/home/$USER/colcon_ws/src/$name_repo" \
        ${is_found_overlay_maps:+--volume "$PATH_MAPS:/home/$USER/data/maps"} \
        ${is_found_overlay_markers_server:+--volume "$PATH_MARKERS_SERVER:/home/$USER/colcon_ws/src/markers_server"} \
        ${is_found_overlay_ros2_config:+--volume "$PATH_ROS2_CONFIG:/home/$USER/colcon_ws/src/$name_repo/libs/ros2_config"} \
        ${is_found_overlay_ros2_interfaces:+--volume "$PATH_ROS2_INTERFACES:/home/$USER/colcon_ws/src/ros2_interfaces"} \
        ${is_found_overlay_ros2_utils:+--volume "$PATH_ROS2_UTILS:/home/$USER/colcon_ws/src/ros2_utils"} \
        "$name_container:$arch" \
        ${command:+"$command"}
}

main() {
    parse_args "$@"
    check_overlays
    run_container
}

main "$@"
