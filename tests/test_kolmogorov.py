"""Tests for the KolmogorovFlow solver."""
import numpy as np
import pytest

import spooky as sp
from spooky.solvers import KolmogorovFlow

# Shared parameters — small grid so tests run fast
LX, LY   = 2 * np.pi, 2 * np.pi
NX, NY   = 32, 32
DT       = 0.01
NU       = 0.1
KF       = 4
F0       = 1.0

FIXTURES = "tests/fixtures"


def _make_solver():
    grid   = sp.Grid2D(LX, LY, NX, NY, DT)
    solver = KolmogorovFlow(grid, kf=KF, f0=F0, nu=NU, rkord=2)
    return grid, solver


def _laminar_ic(grid):
    """Exact laminar Kolmogorov steady state: u = (f0/nu/K²)·sin(K·y), v = 0."""
    K  = 2 * np.pi * KF / LY
    U0 = F0 / (NU * K ** 2)
    uu = U0 * np.sin(K * grid.yy)
    vv = np.zeros_like(uu)
    return uu, vv


def _perturbed_ic(grid):
    """Laminar steady state plus a small divergence-free perturbation."""
    uu, vv = _laminar_ic(grid)
    uu += 0.01 * np.cos(2 * grid.xx) * np.sin(grid.yy)
    vv += 0.01 * np.sin(grid.xx) * np.cos(2 * grid.yy)
    # Project to enforce exact divergence-free condition
    fu, fv = grid.forward(uu), grid.forward(vv)
    fu, fv = grid.inc_proj([fu, fv])
    return grid.inverse(fu), grid.inverse(fv)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_divergence_free_is_maintained():
    """∂u/∂x + ∂v/∂y = 0 must hold throughout the evolution.

    The pressure projector applied to the nonlinear terms in rkstep keeps the
    velocity field on the divergence-free manifold. We start from a projected IC
    and verify the condition still holds after T=1.0.
    """
    grid, solver = _make_solver()
    uu, vv = _perturbed_ic(grid)
    fields = solver.evolve([uu, vv], T=1.0, write_outputs=False)
    fu, fv = grid.forward(fields[0]), grid.forward(fields[1])
    div = grid.kx * fu + grid.ky * fv
    np.testing.assert_allclose(np.abs(div), 0.0, atol=1e-11)


def test_energy_balance_at_steady_state():
    """At the laminar steady state injection exactly balances dissipation.

    The laminar solution u = (f0/nu/K²)·sin(K·y), v=0 is a fixed point of the
    PDE: the nonlinear term vanishes and the forcing balances viscous dissipation.
    This verifies that the solver computes both terms correctly.
    """
    grid, solver = _make_solver()
    uu, vv = _laminar_ic(grid)
    fu, fv = grid.forward(uu), grid.forward(vv)

    ens = grid.enstrophy([fu, fv])
    dis = -2 * NU * ens
    inj = grid.avg(grid.inner([fu, fv], [solver.fx, solver.fy]))

    # At the steady state inj + dis = 0 to machine precision
    np.testing.assert_allclose(inj + dis, 0.0, atol=1e-14)


def test_regression():
    """Evolved solution matches a precomputed reference to machine precision.

    Reference was generated with:
        Lx=Ly=2π, Nx=Ny=32, dt=0.01, nu=0.1, kf=4, f0=1, rkord=2, T=1.0
    and the laminar steady state + small projected perturbation as IC.
    """
    ref_uu = np.load(f"{FIXTURES}/kolmogorov_reference_uu.npy")
    ref_vv = np.load(f"{FIXTURES}/kolmogorov_reference_vv.npy")

    grid, solver = _make_solver()
    uu, vv = _perturbed_ic(grid)
    fields = solver.evolve([uu, vv], T=1.0, write_outputs=False)

    np.testing.assert_allclose(fields[0], ref_uu, atol=1e-14)
    np.testing.assert_allclose(fields[1], ref_vv, atol=1e-14)
