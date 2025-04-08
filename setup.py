from setuptools import setup, find_namespace_packages

setup(
    packages=find_namespace_packages(include=["homelab_ai*"]),
    package_dir={"": "src"},
)
