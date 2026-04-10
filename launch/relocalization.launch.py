from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.conditions import IfCondition


def generate_launch_description():
    pkg_dir = get_package_share_directory('fast_livo')

    rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to start RViz'
    )

    param_file = os.path.join(pkg_dir, 'config', 'mid360_relocalization.yaml')
    camera_file = os.path.join(pkg_dir, 'config', 'camera_see3cam.yaml')

    # Single node: fastlivo_mapping in localization mode
    # ESIKF matches against pre-loaded prior map with LIVO (LIO + VIO)
    reloc_node = Node(
        package='fast_livo',
        executable='fastlivo_mapping',
        name='laserMapping',
        output='screen',
        parameters=[param_file, camera_file],
    )

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_dir, 'rviz_cfg', 'relocalization.rviz')],
        prefix='nice',
    )

    return LaunchDescription([
        rviz_arg,
        reloc_node,
        rviz_node,
    ])
