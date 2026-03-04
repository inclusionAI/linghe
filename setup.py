# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pathlib

from setuptools import find_packages, setup

setup(
    name="linghe",
    version="0.3.5",
    license="MIT",
    license_files=("LICENSE",),
    description="LLM traning kernels",
    URL="https://github.com/inclusionAI/linghe",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.8",
)
