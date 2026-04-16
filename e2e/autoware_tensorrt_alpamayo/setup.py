from setuptools import setup

package_name = "autoware_tensorrt_alpamayo"

setup(
    name=package_name,
    version="0.1.0",
    packages=["tensorrt_alpamayo"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/tensorrt_alpamayo.param.yaml"]),
        ("share/" + package_name + "/launch", ["launch/tensorrt_alpamayo.launch.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="binwang",
    maintainer_email="bin.wang@tier4.jp",
    description="Alpamayo E2E trajectory planner with TensorRT Expert optimization",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "tensorrt_alpamayo_node = tensorrt_alpamayo.alpamayo_node:main",
        ],
    },
)
