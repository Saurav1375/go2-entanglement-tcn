import os
from glob import glob

from setuptools import setup

package_name = "entanglement_recovery"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Saurav Gupta",
    maintainer_email="sauravgupta1375@gmail.com",
    description="Recovery state machine for the Unitree Go2 driven by the entanglement detector.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "recovery_node = entanglement_recovery.recovery_node:main",
        ],
    },
)
