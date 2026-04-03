import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_encoders')
    config_path = os.path.join(pkg_share, 'config', 'theia_tiny_tcp.yaml')

    # Resolve relative model_path to absolute (relative to package source root)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_path = cfg['theia_tiny_tcp_node']['ros__parameters']['model_path']
    if not os.path.isabs(model_path):
        real_launch = os.path.realpath(__file__)
        src_root = os.path.abspath(os.path.join(os.path.dirname(real_launch), '..'))
        model_path = os.path.join(src_root, model_path)

    return LaunchDescription([
        Node(
            package='vision_encoders',
            executable='theia_tiny_tcp_node',
            name='theia_tiny_tcp_node',
            output='screen',
            parameters=[
                config_path,
                {'model_path': model_path},
            ],
        ),
    ])
