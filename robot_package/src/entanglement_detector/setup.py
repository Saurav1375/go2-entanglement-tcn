import os
from glob import glob

from setuptools import setup

package_name = "entanglement_detector"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "models"), glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Saurav Gupta",
    maintainer_email="sauravgupta1375@gmail.com",
    description="Real-time leg-entanglement detector for the Unitree GO2 (multi-task TCN).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "entanglement_node = entanglement_detector.node:main",
        ],
    },
)
