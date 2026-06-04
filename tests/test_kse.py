"""Tests for the KuramotoSivashinsky solver."""
import numpy as np
import pytest

import spooky as sp
from spooky.solvers import KuramotoSivashinsky

# Shared parameters — small grid so tests run fast
LX   = 22.0
NX   = 64
DT   = 1e-3
NU   = 1.0

FIXTURES = "tests/fixtures"


def _make_solver(rkord=2):
    grid   = sp.Grid1D(LX, NX, DT)
    solver = KuramotoSivashinsky(grid, nu=NU, rkord=rkord)
    return grid, solver


def _ic(grid):
    """Deterministic initial condition (same as in examples/kse)."""
    Lx = grid.Lx
    return (0.3 * np.cos(2 * np.pi * 3 * grid.xx / Lx) +
            0.4 * np.cos(2 * np.pi * 5 * grid.xx / Lx) +
            0.5 * np.cos(2 * np.pi * 4 * grid.xx / Lx))


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_zero_mean_is_conserved():
    """The spatial mean must stay exactly zero throughout the evolution.

    KuramotoSivashinsky.rkstep explicitly zeros the k=0 Fourier mode after
    every sub-step, so the mean of the returned physical field should remain at
    the level of irfftn round-off (~1e-17).
    """
    grid, solver = _make_solver()
    fields = solver.evolve([_ic(grid)], T=1.0, write_outputs=False)
    fu = grid.forward(fields[0])
    # The zero mode in Fourier space must be identically zero (up to round-off
    # introduced solely by the final inverse transform).
    assert abs(fu[0]) < 1e-14, f"k=0 mode is {fu[0]:.2e}, expected ~0"


def test_regression():
    """Evolved solution matches a precomputed reference to machine precision.

    Reference was generated with:
        Lx=22, Nx=64, dt=1e-3, nu=1, rkord=2, T=1.0
    and the deterministic IC from examples/kse/time_marching.py.
    Any change that alters the numerics (wrong sign, wrong dealiasing, etc.)
    will break this test.
    """
    ref = np.load(f"{FIXTURES}/kse_reference.npy")

    grid, solver = _make_solver(rkord=2)
    fields = solver.evolve([_ic(grid)], T=1.0, write_outputs=False)

    np.testing.assert_allclose(fields[0], ref, atol=1e-14)
