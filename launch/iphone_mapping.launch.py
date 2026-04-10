#!/usr/bin/python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    config_file_dir = os.path.join(
        get_package_share_directory("fast_livo"), "config")
    rviz_config_file = os.path.join(
        get_package_share_directory("fast_livo"), "rviz_cfg", "fast_livo2.rviz")

    livo_config = os.path.join(config_file_dir, "iphone_offline_mapping.yaml")
    camera_config = os.path.join(config_file_dir, "camera_iphone.yaml")

    use_rviz_arg = DeclareLaunchArgument("use_rviz", default_value="False")

    livo_config_arg = DeclareLaunchArgument(
        'avia_params_file', default_value=livo_config)
    camera_config_arg = DeclareLaunchArgument(
        'camera_params_file', default_value=camera_config)
    use_respawn_arg = DeclareLaunchArgument('use_respawn', default_value='False')

    return LaunchDescription([
        use_rviz_arg,
        livo_config_arg,
        camera_config_arg,
        use_respawn_arg,

        Node(
            package="fast_livo",
            executable="fastlivo_mapping",
            name="laserMapping",
            parameters=[
                LaunchConfiguration('avia_params_file'),
                LaunchConfiguration('camera_params_file'),
            ],
            output="screen"
        ),

        Node(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_file],
            output="screen"
        ),
    ])
