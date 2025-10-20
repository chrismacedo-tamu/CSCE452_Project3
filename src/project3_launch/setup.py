from setuptools import setup

package_name = 'project3_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/project3.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='madmac',
    maintainer_email='cmacedo99@tamu.edu',
    description='Launch package for Project 3 nodes and bag playback',
    license='MIT',
    entry_points={
        'console_scripts': [],
    },
)
