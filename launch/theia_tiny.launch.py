"""Launch Theia-Tiny inference node (+ optional viewer / recorder).

The camera streamer is NOT launched here — run it manually in its own terminal
*before* invoking this launch (sudo needs a tty + the camera release is racy
when both are spawned together):
    python3 src/vision_encoders/scripts/camera_streamer.py

Usage:
  ros2 launch vision_encoders theia_tiny.launch.py
  ros2 launch vision_encoders theia_tiny.launch.py viewer:=true
  ros2 launch vision_encoders theia_tiny.launch.py record:=/path/to/dir
      # ^ enables publish_input_frame, spawns recording_node.py listening to
      #   /joy + /lowstate (Unitree gamepad). The bag lands at
      #   <dir>/<YYYYMonDD_HHMM>/, written between an A (start) and Y/R1 (stop).
  ros2 launch vision_encoders theia_tiny.launch.py viewer:=true record:=/tmp/run
      # ^ all of the above
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_encoders')
    config_path = os.path.join(pkg_share, 'config', 'theia_tiny.yaml')
    viewer_script = os.path.join(pkg_share, 'scripts', 'theia_viewer.py')
    recorder_script = os.path.join(pkg_share, 'scripts', 'recording_node.py')

    def _build(context):
        record_path = LaunchConfiguration('record').perform(context)
        recording = bool(record_path)
        # No existence check on record_path: recording_node appends _<datetime>
        # to it, so launching with the same record:= arg never collides.

        # The node resolves a relative `model_path` against its package share
        # dir, so the yaml's "models/theia_tiny.onnx" works as-is.
        node_params = [config_path]
        # recording requires the post-resize frame on the wire
        if recording:
            node_params.append({'publish_input_frame': True})

        actions = [
            Node(
                package='vision_encoders',
                executable='theia_tiny_node',
                name='theia_tiny_node',
                output='screen',
                parameters=node_params,
            ),
            ExecuteProcess(
                cmd=['python3', viewer_script, '--config', config_path],
                output='screen',
                condition=IfCondition(LaunchConfiguration('viewer')),
            ),
        ]
        if recording:
            actions.append(ExecuteProcess(
                cmd=['python3', recorder_script,
                     '--ros-args', '-p', f'output_dir:={record_path}'],
                output='screen',
            ))
        return actions

    return LaunchDescription([
        DeclareLaunchArgument('viewer', default_value='false',
                              description='launch the OpenCV viewer alongside the node'),
        DeclareLaunchArgument('record', default_value='',
                              description='if non-empty, spawn the joystick-gated recorder '
                                          'and write per-motion segments under this dir '
                                          '(also forces publish_input_frame=true)'),
        OpaqueFunction(function=_build),
    ])
