"""Launch Theia-Tiny inference node (+ optional viewer).

The camera streamer is NOT launched here — run it manually in its own terminal
*before* invoking this launch (sudo needs a tty + the camera release is racy
when both are spawned together):
    python3 src/vision_encoders/scripts/camera_streamer.py

Usage:
  ros2 launch vision_encoders theia_tiny.launch.py
  ros2 launch vision_encoders theia_tiny.launch.py viewer:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_encoders')
    config_path = os.path.join(pkg_share, 'config', 'theia_tiny.yaml')
    viewer_script = os.path.join(pkg_share, 'scripts', 'theia_viewer.py')

    def _build(context):
        viewer = LaunchConfiguration('viewer').perform(context).lower() in ('true', '1')

        # The node resolves a relative `model_path` against its package share
        # dir, so the yaml's "models/theia_tiny.onnx" works as-is.
        actions = [
            Node(
                package='vision_encoders',
                executable='theia_tiny_node',
                name='theia_tiny_node',
                output='screen',
                parameters=[config_path],
            ),
        ]
        if viewer:
            actions.append(ExecuteProcess(
                cmd=['python3', viewer_script, '--config', config_path],
                output='screen',
            ))
        return actions

    return LaunchDescription([
        DeclareLaunchArgument('viewer', default_value='false',
                              description='launch the OpenCV viewer alongside the node'),
        OpaqueFunction(function=_build),
    ])
