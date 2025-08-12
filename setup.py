from setuptools import setup, find_packages

setup(
    name="diffusion-on-the-edge",
    version="0.1.0",
    description="Project for exploring maximum entropy diffusion models",
    author="UrjalaCoder",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
