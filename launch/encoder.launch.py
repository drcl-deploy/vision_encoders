"""Launch the generic encoder node (+ optional viewer).

The camera streamer is NOT launched here — run it manually in its own terminal
*before* invoking this launch (sudo needs a tty + the camera release is racy
when both are spawned together):
    python3 src/vision_encoders/scripts/camera_streamer.py

Usage:
  ros2 launch vision_encoders encoder.launch.py                     # yaml's model
  ros2 launch vision_encoders encoder.launch.py model:=dinov2-s
  ros2 launch vision_encoders encoder.launch.py model:=theia-tiny viewer:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_encoders')
    config_path = os.path.join(pkg_share, 'config', 'encoder.yaml')
    viewer_script = os.path.join(pkg_share, 'scripts', 'viewer.py')

    def _build(context):
        viewer = LaunchConfiguration('viewer').perform(context).lower() in ('true', '1')
        model = LaunchConfiguration('model').perform(context)

        params = [config_path]
        if model:
            params.append({'model': model})

        actions = [
            Node(
                package='vision_encoders',
                executable='encoder_node',
                name='encoder_node',
                output='screen',
                parameters=params,
            ),
        ]
        if viewer:
            actions.append(ExecuteProcess(
                cmd=['python3', viewer_script, '--config', config_path],
                output='screen',
            ))
        return actions

    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='',
                              description="artifact tag (default: the yaml's `model`)"),
        DeclareLaunchArgument('viewer', default_value='false',
                              description='launch the OpenCV viewer alongside the node'),
        OpaqueFunction(function=_build),
    ])
