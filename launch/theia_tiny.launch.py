"""Launch Theia-Tiny inference node (+ optional OpenCV viewer).

Usage:
  ros2 launch vision_encoders theia_tiny.launch.py              # node only
  ros2 launch vision_encoders theia_tiny.launch.py viewer:=true # node + viewer
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_encoders')
    config_path = os.path.join(pkg_share, 'config', 'theia_tiny.yaml')

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_path = cfg['theia_tiny_node']['ros__parameters']['model_path']
    if not os.path.isabs(model_path):
        real_launch = os.path.realpath(__file__)
        src_root = os.path.abspath(os.path.join(os.path.dirname(real_launch), '..'))
        model_path = os.path.join(src_root, model_path)

    viewer_script = os.path.join(pkg_share, 'scripts', 'theia_viewer.py')

    return LaunchDescription([
        DeclareLaunchArgument('viewer', default_value='false',
                              description='launch the OpenCV viewer alongside the node'),

        Node(
            package='vision_encoders',
            executable='theia_tiny_node',
            name='theia_tiny_node',
            output='screen',
            parameters=[
                config_path,
                {'model_path': model_path},
            ],
        ),

        ExecuteProcess(
            cmd=['python3', viewer_script, '--config', config_path],
            output='screen',
            condition=IfCondition(LaunchConfiguration('viewer')),
        ),
    ])
