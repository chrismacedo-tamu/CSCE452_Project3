from setuptools import setup

package_name = "publish_markers"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Christian Macedo',
    maintainer_email='cmacedo99@tamu.edu',
    description='Node that publishes markers',
    license='MIT',
    entry_points={
        'console_scripts': [
            'publish_markers_node = publish_markers.publish_markers_node:main',
        ],
    },
)