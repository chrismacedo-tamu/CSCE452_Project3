from setuptools import setup, find_packages

package_name = "read_scan"

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Christian Macedo',
    maintainer_email='cmacedo99@tamu.edu',
    description='Node that reads /scan',
    license='MIT',
    entry_points={
        'console_scripts': [
            'read_scan_node = read_scan.read_scan_node:main',
        ],
    },
)