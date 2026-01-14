# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as f:
    install_requires = [str(requirement) for requirement in
                        pkg_resources.parse_requirements(f)]

setup(
    name="linghe",
    version="0.3.0",
    license="MIT",
    license_files=("LICENSE",),
    description="LLM traning kernels",
    URL="https://github.com/inclusionAI/linghe",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.8",
)
