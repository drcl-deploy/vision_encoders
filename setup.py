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
    install_requires=[
        'torch==2.8.0',
        'torchvision==0.15.0',
        'pillow==9.0.0',
        'requests==2.28.0',
        'tqdm==4.64.0',
        'numpy==1.21.0',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Vision encoder nodes for multiple architectures',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'dino_encoder_node = vision_encoders.dino_encoder_node:main',
            'theia_encoder_node = vision_encoders.theia_encoder_node:main',
        ],
    },
)
