#!/usr/bin/python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    config_file_dir = os.path.join(
        get_package_share_directory("fast_livo"), "config")
    rviz_config_file = os.path.join(
        get_package_share_directory("fast_livo"), "rviz_cfg", "fast_livo2.rviz")

    livo_config = os.path.join(config_file_dir, "mid360_livo_mapping.yaml")
    camera_config = os.path.join(config_file_dir, "camera_see3cam.yaml")

    bag_path = os.path.expanduser("~/bag/rosbag2_2026_04_28-21_24_16")

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="False", description="Whether to launch Rviz2")

    return LaunchDescription([
        use_rviz_arg,

        # Play only LiDAR, IMU, and compressed image
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "play", bag_path,
                "--clock", "--rate", "0.5",
                "--topics",
                "/livox/lidar", "/livox/imu", "/image_raw/compressed",
            ],
            output="screen",
        ),

        Node(
            package="fast_livo",
            executable="fastlivo_mapping",
            name="laserMapping",
            parameters=[
                livo_config,
                camera_config,
                {"use_sim_time": True},
            ],
            output="screen",
        ),

        Node(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_file],
            parameters=[{"use_sim_time": True}],
            output="screen",
        ),
    ])
