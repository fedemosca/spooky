import os

# JAX defaults to float32, which is insufficient for the precision expected by
# these tests. Enable float64 before JAX is imported (conftest.py is loaded by
# pytest before any test modules are collected).
_backend = os.environ.get('NUMPY_BACKEND', 'numpy').lower()
if _backend == 'jax':
    os.environ.setdefault('JAX_ENABLE_X64', '1')


def pytest_report_header(config):
    return f"spooky backend: {_backend}"
