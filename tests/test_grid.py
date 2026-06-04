"""Tests for Grid1D and Grid2D."""
import numpy as np
import pytest

import spooky as sp
from spooky._backend import index_update, xnp


# ── Grid1D ───────────────────────────────────────────────────────────────────

class TestGrid1D:
    @pytest.fixture
    def g(self):
        return sp.Grid1D(Lx=2 * np.pi, Nx=32, dt=0.01)

    # -- FFT round-trips -------------------------------------------------------

    def test_inverse_forward_roundtrip(self, g):
        """inverse(forward(u)) recovers u to machine precision."""
        u = np.random.default_rng(0).standard_normal(g.Nx)
        np.testing.assert_allclose(g.inverse(g.forward(u)), u, atol=1e-12)

    def test_forward_inverse_roundtrip(self, g):
        """forward(inverse(forward(u))) recovers forward(u) to machine precision.

        inverse(fu) requires conjugate-symmetric input (i.e. fu must come from a
        real field). We construct u as a bandlimited sum of sinusoids so it is
        smooth, periodic, and guaranteed conjugate-symmetric after forward().
        """
        rng = np.random.default_rng(1)
        ks = np.arange(1, g.Nx // 3)   # keep well below Nyquist
        u = sum(rng.standard_normal() * np.sin(k * g.xx) +
                rng.standard_normal() * np.cos(k * g.xx) for k in ks)
        fu = g.forward(u)
        np.testing.assert_allclose(g.forward(g.inverse(fu)), fu, atol=1e-12)

    # -- Spectral derivative ---------------------------------------------------

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_spectral_derivative(self, g, k):
        """d/dx[sin(k*x)] == k*cos(k*x) to near-machine precision.

        Lx = 2π so the physical wavenumber equals the mode index k.
        """
        u = np.sin(k * g.xx)
        du = g.inverse(g.deriv(g.forward(u), g.kx))
        np.testing.assert_allclose(du, k * np.cos(k * g.xx), atol=1e-11)

    # -- Energy / enstrophy of known fields ------------------------------------

    @pytest.mark.parametrize("A, k", [(1.0, 2), (3.0, 4)])
    def test_energy_sinusoid(self, g, A, k):
        """Energy of A*sin(k*x) is A²/4."""
        fu = g.forward(A * np.sin(k * g.xx))
        np.testing.assert_allclose(g.energy([fu]), A ** 2 / 4, rtol=1e-10)

    @pytest.mark.parametrize("A, k", [(1.0, 2), (2.0, 3)])
    def test_enstrophy_sinusoid(self, g, A, k):
        """Enstrophy of A*sin(k*x) is k²·A²/4 (Lx=2π so physical wavenumber = k)."""
        fu = g.forward(A * np.sin(k * g.xx))
        np.testing.assert_allclose(g.enstrophy([fu]), k ** 2 * A ** 2 / 4, rtol=1e-10)

    def test_energy_zero_field(self, g):
        """Energy of the zero field is zero."""
        fu = np.zeros(g.Nx // 2 + 1, dtype=complex)
        assert g.energy([fu]) == 0.0

    # -- Dealiasing ------------------------------------------------------------

    def test_dealias_zeros_high_modes(self, g):
        """Applying dealias_modes zeros all modes above the 2/3 threshold."""
        fu = xnp.ones(g.Nx // 2 + 1, dtype=complex)
        fu_d = index_update(fu, g.dealias_modes, 0.0)
        assert np.all(fu_d[g.dealias_modes] == 0.0), "High modes not zeroed"
        assert np.all(fu_d[~g.dealias_modes] == 1.0), "Low modes incorrectly modified"

    def test_dealias_threshold_location(self, g):
        """The 2/3-rule boundary sits at mode Nx/3."""
        threshold = g.Nx // 3
        assert not g.dealias_modes[threshold],     "Mode at threshold should be kept"
        assert g.dealias_modes[threshold + 1],     "Mode just above threshold should be zeroed"

    # -- Translation -----------------------------------------------------------

    def test_translate_full_period(self, g):
        """Translating by Lx (one full period) returns the original field."""
        u = np.sin(3 * g.xx) + 0.5 * np.cos(5 * g.xx)
        np.testing.assert_allclose(g.translate([u], g.Lx)[0], u, atol=1e-11)

    def test_translate_quarter_period(self, g):
        """Translating sin(x) by Lx/4 gives cos(x).

        sin(x + π/2) = cos(x), with Lx=2π so Lx/4 = π/2.
        """
        u = np.sin(g.xx)
        u_shifted = g.translate([u], g.Lx / 4)[0]
        np.testing.assert_allclose(u_shifted, np.cos(g.xx), atol=1e-11)


# ── Grid2D ───────────────────────────────────────────────────────────────────

class TestGrid2D:
    @pytest.fixture
    def g(self):
        return sp.Grid2D(Lx=2 * np.pi, Ly=2 * np.pi, Nx=32, Ny=32, dt=0.01)

    # -- FFT round-trips -------------------------------------------------------

    def test_inverse_forward_roundtrip(self, g):
        """inverse(forward(u)) recovers u to machine precision."""
        u = np.random.default_rng(0).standard_normal((g.Nx, g.Ny))
        np.testing.assert_allclose(g.inverse(g.forward(u)), u, atol=1e-11)

    # -- Spectral derivatives --------------------------------------------------

    @pytest.mark.parametrize("k", [1, 3])
    def test_spectral_derivative_x(self, g, k):
        """d/dx[sin(k*x)] == k*cos(k*x) on a 2D grid."""
        u = np.sin(k * g.xx)
        du = g.inverse(g.deriv(g.forward(u), g.kx))
        np.testing.assert_allclose(du, k * np.cos(k * g.xx), atol=1e-10)

    @pytest.mark.parametrize("k", [1, 3])
    def test_spectral_derivative_y(self, g, k):
        """d/dy[sin(k*y)] == k*cos(k*y) on a 2D grid."""
        u = np.sin(k * g.yy)
        du = g.inverse(g.deriv(g.forward(u), g.ky))
        np.testing.assert_allclose(du, k * np.cos(k * g.yy), atol=1e-10)

    # -- Energy of known fields ------------------------------------------------

    @pytest.mark.parametrize("A, k", [(1.0, 2), (3.0, 3)])
    def test_energy_sinusoid(self, g, A, k):
        """Energy of (A*sin(k*x), 0) is A²/4."""
        fu = g.forward(A * np.sin(k * g.xx))
        fv = np.zeros_like(fu)
        np.testing.assert_allclose(g.energy([fu, fv]), A ** 2 / 4, rtol=1e-10)

    # -- Dealiasing ------------------------------------------------------------

    def test_dealias_zeros_high_modes(self, g):
        """Applying dealias_modes on a 2D spectral array zeros all flagged modes."""
        rng = np.random.default_rng(0)
        shape = (g.Nx, g.Ny // 2 + 1)
        fu = xnp.array(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        fu_d = index_update(fu, g.dealias_modes, 0.0)
        assert np.all(fu_d[g.dealias_modes] == 0.0), "High modes not zeroed"

    # -- Incompressibility projector -------------------------------------------

    def test_inc_proj_divergence_free(self, g):
        """inc_proj output satisfies kx·fu + ky·fv = 0 everywhere in Fourier space."""
        rng = np.random.default_rng(42)
        shape = (g.Nx, g.Ny // 2 + 1)
        fu = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        fv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        fu_p, fv_p = g.inc_proj([fu, fv])
        div = g.kx * fu_p + g.ky * fv_p
        np.testing.assert_allclose(np.abs(div), 0.0, atol=1e-11)

    def test_inc_proj_idempotent(self, g):
        """P(P(f)) == P(f): applying inc_proj twice gives the same result as once."""
        rng = np.random.default_rng(7)
        shape = (g.Nx, g.Ny // 2 + 1)
        fu = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        fv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        fu_p,  fv_p  = g.inc_proj([fu,   fv])
        fu_pp, fv_pp = g.inc_proj([fu_p, fv_p])
        np.testing.assert_allclose(fu_pp, fu_p, atol=1e-12)
        np.testing.assert_allclose(fv_pp, fv_p, atol=1e-12)

    def test_inc_proj_preserves_divergence_free_field(self, g):
        """A field already satisfying ∂u/∂x + ∂v/∂y = 0 is unchanged by inc_proj.

        u = cos(y), v = -cos(x) is exactly divergence-free (∂u/∂x = ∂v/∂y = 0).
        """
        fu = g.forward(np.cos(g.yy))
        fv = g.forward(-np.cos(g.xx))
        fu_p, fv_p = g.inc_proj([fu, fv])
        np.testing.assert_allclose(fu_p, fu, atol=1e-12)
        np.testing.assert_allclose(fv_p, fv, atol=1e-12)

    # -- Translation -----------------------------------------------------------

    def test_translate2d_full_period_x(self, g):
        """Translating by (Lx, 0) is the identity."""
        u = np.sin(3 * g.xx) + np.cos(2 * g.yy)
        np.testing.assert_allclose(g.translate2D([u], g.Lx, 0.0)[0], u, atol=1e-11)

    def test_translate2d_full_period_y(self, g):
        """Translating by (0, Ly) is the identity."""
        u = np.cos(g.xx) + np.sin(2 * g.yy)
        np.testing.assert_allclose(g.translate2D([u], 0.0, g.Ly)[0], u, atol=1e-11)
