import os
import os.path as op
from distutils.core import setup

from setuptools import PEP420PackageFinder

ROOT = op.dirname(op.abspath(__file__))
SRC = op.join(ROOT, "src")


def get_version_info():
    """Extract version information as a dictionary from version.py."""
    version_info = {}
    version_filename = os.path.join("src", "ta_lib", "version.py")
    with open(version_filename, "r") as version_module:
        version_code = compile(version_module.read(), "version.py", "exec")
    exec(version_code, version_info)
    return version_info


setup(
    name="ta_lib",
    version=get_version_info()["version"],
    package_dir={"": "src"},
    description="DS Templates",
    author="Tiger Analytics",
    packages=PEP420PackageFinder.find(where=str(SRC)),
    package_data={
        "ta_lib": [
            "_vendor/tigerml/core/reports/html/report_resources/*",
            "_vendor/tigerml/viz/static_resources/*",
            "_vendor/tigerml/automl/backends/conf.json",
        ]
    },
    include_package_data=True,
    python_requires=">=3.8, <=3.11.3",  # Specify the supported Python versions
)
