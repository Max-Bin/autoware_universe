from setuptools import setup

package_name = "autoware_alpamayo_trt"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/alpamayo_trt.param.yaml"]),
        ("share/" + package_name + "/launch", ["launch/alpamayo_trt.launch.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="binwang",
    maintainer_email="bin.wang@tier4.jp",
    description="Alpamayo E2E trajectory planner with TRT Expert optimization",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "alpamayo_trt_node = autoware_alpamayo_trt.alpamayo_node:main",
        ],
    },
)
