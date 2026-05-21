#!/usr/bin/python3
# -- coding: utf-8 --**

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("fast_livo")
    config_dir = os.path.join(pkg_dir, "config")
    # ★ rviz config 는 install/share 가 아닌 src/ 의 파일을 직접 봐서, 편집 시 rebuild 없이 반영되도록.
    src_pkg_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    rviz_config_file = os.path.join(src_pkg_dir, "rviz_cfg", "fast_livo2.rviz")

    main_params = os.path.join(config_dir, "mid360_mapping.yaml")
    camera_params = os.path.join(config_dir, "camera_see3cam.yaml")

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="True",
        description="Whether to launch RViz2",
    )

    mapping_node = Node(
        package="fast_livo",
        executable="fastlivo_mapping",
        name="laserMapping",
        parameters=[main_params, camera_params],
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

    return LaunchDescription([
        use_rviz_arg,
        mapping_node,
        rviz_node,
    ])
