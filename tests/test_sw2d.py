"""Tests for the SWHD_2D shallow water solver.

The physical case is the one the solver is built for: a pulse propagating in +x that
runs over a Gaussian bump at the centre of the domain and deforms. Grids are small so
the whole file runs in a few seconds; the full-resolution study lives in
examples/sw2d/residuals.py.

Conventions, which the assertions depend on: `h` is the free-surface elevation measured
from a datum, `hb` the bottom measured from the same datum, so `H = h - hb` is the water
column and must stay positive.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import spooky as sp
from spooky.solvers import SWHD_2D

# Shared parameters — cgs units, small grid so tests run fast
LX, LY   = 30.0, 30.0
NX, NY   = 64, 64
DT       = 5e-4
T        = 0.2
G        = 981.0
H_REST   = 3.0

# Pulse: amplitude, half-length, launch position
A, S, X0 = 0.3, 1.5, 8.0
# Gaussian bump: height and width, centred in the domain
HB_HEIGHT, HB_WIDTH = 1.0, 3.0

FIXTURES = "tests/fixtures"


def _make_params(dt=DT):
    """SWHD_2D reads these off pm in __init__; only g and dt affect the physics here."""
    return SimpleNamespace(
        g=G, dt=dt, T=T, ostep=1, bstep=1,
        out_path='.', data_path='.', hb_path='.',
        make_data=False, noise=False,
        uum_noise_std=0.0, vvm_noise_std=0.0, hhm_noise_std=0.0,
        iit=0, iit0=0, iitN=1,
    )


def _make_solver(bump_height=HB_HEIGHT, n=NX, dt=DT):
    grid = sp.Grid2D(Lx=LX, Ly=LY, Nx=n, Ny=n, dt=dt)
    solver = SWHD_2D(grid, _make_params(dt))
    xx, yy = np.asarray(grid.xx), np.asarray(grid.yy)
    hb = bump_height * np.exp(-((xx - LX/2)**2 + (yy - LY/2)**2) / HB_WIDTH**2)
    solver.update_hb(hb)
    return grid, solver


def _pulse_ic(grid):
    """Right-moving pulse. u = eta*sqrt(g/H) is the linearised simple-wave relation,
    so the left-going component is nonlinear in A/H_REST and stays small."""
    xx = np.asarray(grid.xx)
    pulse = A * np.exp(-((xx - X0) / S)**2)
    uu = np.sqrt(G / H_REST) * pulse
    vv = np.zeros_like(xx)
    hh = H_REST + pulse
    return [uu, vv, hh]


def _evolve(solver, grid, T=T):
    """Returns diagnostics before and after, plus the final physical fields."""
    ic = _pulse_ic(grid)
    before = solver.diagnostics([grid.forward(ff) for ff in ic])
    fields = solver.evolve(ic, T=T, write_outputs=False)
    after = solver.diagnostics([grid.forward(ff) for ff in fields])
    return before, after, fields


# ── Conservation ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('bump_height', [0.0, HB_HEIGHT], ids=['flat', 'bump'])
def test_mass_is_conserved(bump_height):
    """Total water column <h-hb> is conserved to machine precision.

    Exactly, not approximately: the continuity equation is a pure divergence, whose
    domain average vanishes identically in Fourier space, so nothing but roundoff can
    change the mean. A failure here means the flux term is malformed.
    """
    grid, solver = _make_solver(bump_height)
    before, after, _ = _evolve(solver, grid)
    np.testing.assert_allclose(after[0], before[0], rtol=1e-12)


@pytest.mark.parametrize('bump_height', [0.0, HB_HEIGHT], ids=['flat', 'bump'])
def test_energy_is_conserved(bump_height):
    """E = <(h-hb)|u|^2>/2 + g<h^2>/2 is invariant for the continuous equations.

    There is no viscosity and no forcing, so the only drift is time-discretisation
    error. Measured drift is ~5e-7 at this dt; the tolerance leaves an order of margin.
    """
    grid, solver = _make_solver(bump_height)
    before, after, _ = _evolve(solver, grid)
    E0, E1 = before[1] + before[2], after[1] + after[2]
    np.testing.assert_allclose(E1, E0, rtol=1e-5)


def test_energy_conservation_converges_with_dt():
    """Energy drift must shrink as dt does, confirming it is discretisation error.

    Without this, the conservation tests above could be passed by a solver that simply
    fails to evolve anything. Halving dt drops the drift by ~7x in practice.
    """
    drifts = []
    for dt in (1e-3, 5e-4):
        grid, solver = _make_solver(HB_HEIGHT, dt=dt)
        before, after, _ = _evolve(solver, grid)
        E0, E1 = before[1] + before[2], after[1] + after[2]
        drifts.append(abs(E1/E0 - 1))
    assert drifts[0] > 4 * drifts[1], f'drift did not converge: {drifts}'


def test_energy_partition_actually_moves():
    """Guards the conservation tests against being trivially satisfied.

    Kinetic and potential energy each change by ~1% over the run while their sum holds
    to ~5e-7, so the invariance above reflects a real cancellation rather than a static
    solution.
    """
    grid, solver = _make_solver(HB_HEIGHT)
    before, after, _ = _evolve(solver, grid)
    assert abs(after[1]/before[1] - 1) > 1e-3


# ── Physical admissibility ────────────────────────────────────────────────────

def test_water_column_stays_positive():
    """H = h - hb > 0 everywhere; the equations are ill-posed once it is not.

    With h_rest=3 and a 1 cm bump the column is ~2 cm at the crest. Dropping the rest
    height from the initial condition is what drives this negative and produces NaNs.
    """
    grid, solver = _make_solver(HB_HEIGHT)
    _, _, fields = _evolve(solver, grid)
    assert np.min(fields[2] - solver.hb) > 0.0


def test_topography_deforms_the_wave():
    """The bump must change the solution, and be re-read on every evolve().

    Deliberately reuses one solver and swaps hb between runs: under jit `self` is a
    static argument, so an hb read inside the compiled function is baked in at trace
    time and the id-keyed cache never refreshes it. That made the bumped run bit-for-bit
    identical to the flat one, which is also what silently breaks the hb update loop in
    the assimilation solvers. Two separate solver instances would not catch this — they
    hash differently and trigger a recompile.
    """
    grid, solver = _make_solver(HB_HEIGHT)
    xx, yy = np.asarray(grid.xx), np.asarray(grid.yy)

    solver.update_hb(np.zeros_like(xx))
    _, _, flat = _evolve(solver, grid)

    solver.update_hb(HB_HEIGHT * np.exp(-((xx - LX/2)**2 + (yy - LY/2)**2) / HB_WIDTH**2))
    _, _, bumped = _evolve(solver, grid)

    assert np.max(np.abs(flat[2] - bumped[2])) > 1e-3


def test_mean_velocity_is_not_removed():
    """<u> is dynamical here, and must not be projected out.

    A localised pulse carries net momentum, so <u> > 0. Averaging the momentum equation
    over the periodic domain kills <grad h> and leaves d<u>/dt = -<u.grad u>, which is
    second-order small — so <u> stays near its initial value rather than at zero.
    Zeroing the velocity zero mode (as Kolmogorov flow does, where a uniform shift is a
    genuine symmetry) would instead pin it to zero and slow the front.
    """
    grid, solver = _make_solver(HB_HEIGHT)
    ic = _pulse_ic(grid)
    umean0 = np.mean(ic[0])
    _, _, fields = _evolve(solver, grid)
    assert umean0 > 0.4
    np.testing.assert_allclose(np.mean(fields[0]), umean0, rtol=1e-3)


def test_balance_file_is_truncated_between_runs(tmp_path):
    """Each run must start balance.dat fresh.

    Opening in append mode concatenates successive runs into one file, which then reads
    as a time series that jumps backwards partway through.
    """
    grid, solver = _make_solver(HB_HEIGHT)
    solver.pm.out_path = str(tmp_path)
    for _ in range(2):
        solver.evolve(_pulse_ic(grid), T=10*DT, bstep=1, ostep=None, bpath=str(tmp_path))

    times = np.atleast_2d(np.loadtxt(tmp_path / 'balance.dat'))[:, 0]
    assert np.all(np.diff(times) >= 0), f'time runs backwards: {times}'


# ── Regression ────────────────────────────────────────────────────────────────

def test_regression():
    """Evolved solution matches a precomputed reference.

    Reference generated with the module constants above at T=0.05. The tolerance is
    loose enough to hold across the NumPy and JAX backends, which agree to ~1e-15 in
    float64, but tight enough to catch any change in the scheme.
    """
    ref = np.load(f"{FIXTURES}/sw2d_reference.npz")
    grid, solver = _make_solver(HB_HEIGHT)
    _, _, fields = _evolve(solver, grid, T=0.05)
    for name, got in zip(('uu', 'vv', 'hh'), fields):
        np.testing.assert_allclose(got, ref[name], atol=1e-10, err_msg=name)
