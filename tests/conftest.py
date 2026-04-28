# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest

from linghe.tools.benchmark import benchmark_func


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run benchmark tests",
    )


@pytest.fixture
def benchmark(request):
    class _Benchmark:
        def __call__(self, *args, **kwargs):
            if not request.config.getoption("--benchmark"):
                return None
            return benchmark_func(*args, **kwargs)
    return _Benchmark()
