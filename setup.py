from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vision_encoders'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'extern']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('vision_encoders/config/*.yaml')),
    ],
    install_requires=['setuptools'],  # ONLY setuptools here!
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Vision encoder nodes for multiple architectures',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dino_encoder_node = vision_encoders.dino_encoder_node:main',
            'theia_encoder_node = vision_encoders.theia_encoder_node:main',
        ],
    },
)
