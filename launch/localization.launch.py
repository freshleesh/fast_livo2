#!/usr/bin/python3
# -- coding: utf-8 --**
#
# Usage:
#   ros2 launch fast_livo localization.launch.py                 # yaml default (hall_0521_1)
#   ros2 launch fast_livo localization.launch.py map:=my_seat    # override → stack_master/maps/my_seat
#
# `map` 인자는 stack_master/maps/<name>/ 디렉터리 이름. middle_level_mac.launch.py 와 같은 패턴.

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


# Canonical map storage: stack_master/maps/<name>/ (src side — fast_livo
# 가 SaveMap 으로 새 파일 쓰는 곳과 동일).
# 머신마다 다르면 FASTLIVO_MAPS_ROOT 환경변수로 덮어쓴다 (기본값 = 실차 mac).
MAPS_ROOT = os.environ.get(
    "FASTLIVO_MAPS_ROOT",
    "/Users/mini/ros2_ws/src/IFAC2026_SH/src/system/stack_master/maps")


def launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory("fast_livo")
    config_dir = os.path.join(pkg_dir, "config")
    src_pkg_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    rviz_config_file = os.path.join(src_pkg_dir, "rviz_cfg", "relocalization.rviz")

    main_params = os.path.join(config_dir, "mid360_localization.yaml")
    camera_params = os.path.join(config_dir, "camera_see3cam.yaml")

    # 머신별 오버라이드 (git 미추적, repo 밖). 경로·img_en 등 머신마다 다른
    # 값은 base yaml이 아니라 여기에 둔다. 템플릿: config/local.yaml.example
    params = [main_params, camera_params]
    local_params = os.path.expanduser("~/.config/fast_livo/local.yaml")
    if os.path.exists(local_params):
        params.append(local_params)

    map_name = LaunchConfiguration("map").perform(context)

    # map:= 인자는 local.yaml보다 우선 (마지막에 얹음)
    if map_name:
        prior_map_dir = os.path.join(MAPS_ROOT, map_name)
        params.append({"relocalization.prior_map_dir": prior_map_dir})

    reloc_node = Node(
        package="fast_livo",
        executable="fastlivo_mapping",
        name="laserMapping",
        parameters=params,
        output="screen",
    )

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_file],
        output="screen",
    )

    return [reloc_node, rviz_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_rviz", default_value="True",
            description="Whether to launch RViz2",
        ),
        DeclareLaunchArgument(
            "map", default_value="",
            description="Prior map name under stack_master/maps/<name>/. "
                        "Empty = yaml default (relocalization.prior_map_dir).",
        ),
        OpaqueFunction(function=launch_setup),
    ])
