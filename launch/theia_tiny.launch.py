"""Launch Theia-Tiny inference node (+ optional viewer / recorder).

Usage:
  ros2 launch vision_encoders theia_tiny.launch.py
  ros2 launch vision_encoders theia_tiny.launch.py viewer:=true
  ros2 launch vision_encoders theia_tiny.launch.py record:=/path/to/dir
      # ^ enables publish_input_frame, spawns recording_node.py
      #   listening to /joy + /lowstate (Unitree gamepad). Each
      #   A -> Y/R1 cycle creates <dir>/segNN/ as its own bag.
  ros2 launch vision_encoders theia_tiny.launch.py viewer:=true record:=/tmp/run
      # ^ all of the above

  # Onboard mode (streamer + node on same Orin, no ethernet):
  ros2 launch vision_encoders theia_tiny.launch.py onboard:=true
"""
import os

import yaml
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
    streamer_script = os.path.join(pkg_share, 'scripts', 'camera_streamer.py')

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_path = cfg['theia_tiny_node']['ros__parameters']['model_path']
    if not os.path.isabs(model_path):
        real_launch = os.path.realpath(__file__)
        src_root = os.path.abspath(os.path.join(os.path.dirname(real_launch), '..'))
        model_path = os.path.join(src_root, model_path)

    def _build(context):
        record_path = LaunchConfiguration('record').perform(context)
        recording = bool(record_path)
        onboard = LaunchConfiguration('onboard').perform(context).lower() == 'true'
        # No existence check: recording_node appends _<datetime> to the path,
        # so launching with the same record:= arg never collides.

        node_params = [
            config_path,
            {'model_path': model_path},
        ]
        if onboard:
            node_params.append({'camera_ip': '127.0.0.1'})
        # recording requires the post-resize frame on the wire
        if recording:
            node_params.append({'publish_input_frame': True})

        streamer_cmd = ['python3', streamer_script]
        if onboard:
            streamer_cmd.append('--onboard')

        actions = []
        if onboard:
            actions.append(ExecuteProcess(
                cmd=streamer_cmd,
                output='screen',
            ))
        actions += [
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
        DeclareLaunchArgument('onboard', default_value='false',
                              description='run streamer + node on same device (loopback, no ethernet)'),
        DeclareLaunchArgument('record', default_value='',
                              description='if non-empty, spawn the joystick-gated recorder '
                                          'and write per-motion segments under this dir '
                                          '(also forces publish_input_frame=true)'),
        OpaqueFunction(function=_build),
    ])
