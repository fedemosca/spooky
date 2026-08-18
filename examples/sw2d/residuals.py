'''
PDE residual and conservation diagnostics for the 2D shallow water solver.

Evolves the wavefront-over-bump case and, at a set of probe times, measures how well the
discrete solution satisfies the continuous equations:

    r_u = du/dt + u ux + v uy + g hx
    r_v = dv/dt + u vx + v vy + g hy
    r_h = dh/dt + ( u(h-hb) )x + ( v(h-hb) )y

The time derivative is a centred difference over three *consecutive* steps, so its own
truncation error is O(dt^2) rather than O(ostep*dt)^2 -- taking the difference over saved
snapshots instead would measure the snapshot spacing, not the scheme.

Residuals are reported normalised by the largest individual term in the same equation,
which is what makes the numbers interpretable: 1e-6 means the equation is satisfied to
one part in a million of the terms being balanced.

Meant for the cluster. Run with:
    sbatch job.sh          (after pointing it at this script)
'''

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spooky as sp
from spooky._backend import index_update
from spooky.solvers import SWHD_2D

import params as pm

# Number of times at which to evaluate the residual, spread over the run
N_PROBES = 6

# The initial condition below adds the rest height that params.py omits: h is the free
# surface, so h - hb is the water column and must stay positive. Without the offset it
# goes negative over the bump and the run turns into NaNs.
H_REST = pm.h_rest


def make_fields(grid) -> list:
    ''' Right-moving pulse, linearised simple-wave relation between u and the elevation. '''
    xx = np.asarray(grid.xx)
    pulse = pm.A * np.exp(-((xx - pm.x0) / pm.s) ** pm.n)
    uu = np.sqrt(pm.g / H_REST) * pulse
    vv = np.zeros_like(xx)
    hh = H_REST + pulse
    return [uu, vv, hh]


def make_hb(grid) -> np.ndarray:
    xx, yy = np.asarray(grid.xx), np.asarray(grid.yy)
    return pm.H0 * np.exp(-((xx - pm.xb) ** 2 + (yy - pm.yb) ** 2) / pm.R ** 2)


def rhs_terms(solver, ffields) -> tuple:
    ''' Individual right-hand-side terms in physical space, dealiased as in rkstep.

    Returns (rhs, terms) where rhs[i] is the full right-hand side of equation i and
    terms[i] is the list of its separate contributions, kept apart so the residual can be
    normalised by the largest term actually being balanced.
    '''
    grid, hb = solver.grid, solver.hb
    fu, fv, fh = ffields

    uu, vv, hh = (grid.inverse(ff) for ff in ffields)
    ux = grid.inverse(grid.deriv(fu, grid.kx))
    uy = grid.inverse(grid.deriv(fu, grid.ky))
    vx = grid.inverse(grid.deriv(fv, grid.kx))
    vy = grid.inverse(grid.deriv(fv, grid.ky))

    def dealias(ff):
        return grid.inverse(index_update(ff, grid.dealias_modes, 0.0))

    adv_u = dealias(grid.forward(uu * ux + vv * uy))
    adv_v = dealias(grid.forward(uu * vx + vv * vy))
    grad_h_x = dealias(pm.g * grid.deriv(fh, grid.kx))
    grad_h_y = dealias(pm.g * grid.deriv(fh, grid.ky))
    flux_x = dealias(grid.deriv(grid.forward(uu * (hh - hb)), grid.kx))
    flux_y = dealias(grid.deriv(grid.forward(vv * (hh - hb)), grid.ky))

    rhs = [-(adv_u + grad_h_x), -(adv_v + grad_h_y), -(flux_x + flux_y)]
    terms = [[adv_u, grad_h_x], [adv_v, grad_h_y], [flux_x, flux_y]]
    return rhs, terms


def residual_at(solver, grid, fields) -> tuple:
    ''' Residual of each equation at the current state, via three consecutive steps.

    Advances two single steps so the centred time derivative is taken at spacing dt.
    Returns (residual fields, normalised residual norms, fields advanced by one step).
    '''
    dt = grid.dt
    prev = fields
    curr = solver.evolve(prev, T=dt, write_outputs=False)
    nxt = solver.evolve(curr, T=dt, write_outputs=False)

    fcurr = [grid.forward(ff) for ff in curr]
    rhs, terms = rhs_terms(solver, fcurr)

    residuals, norms = [], []
    for i in range(3):
        ddt = (np.asarray(nxt[i]) - np.asarray(prev[i])) / (2.0 * dt)
        res = ddt - np.asarray(rhs[i])
        scale = max(np.max(np.abs(np.asarray(t))) for t in terms[i])
        residuals.append(res)
        norms.append(np.max(np.abs(res)) / scale if scale > 0 else np.nan)
    return residuals, norms, curr


def main() -> None:
    os.makedirs(pm.out_path, exist_ok=True)

    grid = sp.Grid2D(Lx=pm.Lx, Ly=pm.Ly, Nx=pm.Nx, Ny=pm.Ny, dt=pm.dt)
    solver = SWHD_2D(grid, pm)
    solver.update_hb(make_hb(grid))

    fields = make_fields(grid)
    c = np.sqrt(pm.g * H_REST)
    reach = pm.x0 + c * pm.T
    print(f'wave speed {c:.2f} cm/s; front reaches x = {reach:.1f} cm by T = {pm.T}')
    if reach < pm.xb:
        print(f'WARNING: bump is at x = {pm.xb}, the front never gets there. '
              f'T >= {(pm.xb - pm.x0) / c:.3f} is needed for any interaction.')

    f0 = [grid.forward(ff) for ff in fields]
    mass0, ekin0, epot0 = solver.diagnostics(f0)

    probe_times = np.linspace(pm.T / N_PROBES, pm.T, N_PROBES)
    history = {'t': [], 'res': [], 'mass': [], 'energy': [], 'hmin': []}
    snapshots = []

    t = 0.0
    for t_probe in probe_times:
        # Two of the steps to t_probe are taken inside residual_at
        span = t_probe - t - 2 * pm.dt
        if span > 0:
            fields = solver.evolve(fields, T=span, write_outputs=False)
        residuals, norms, fields = residual_at(solver, grid, fields)
        t = t_probe

        ff = [grid.forward(x) for x in fields]
        mass, ekin, epot = solver.diagnostics(ff)
        hcol = np.asarray(fields[2]) - np.asarray(solver.hb)

        history['t'].append(t)
        history['res'].append(norms)
        history['mass'].append(float(mass))
        history['energy'].append(float(ekin + epot))
        history['hmin'].append(float(np.min(hcol)))
        snapshots.append((t, np.asarray(fields[2]), residuals[2]))

        print(f't={t:7.4f}  res(u,v,h) = {norms[0]:.2e} {norms[1]:.2e} {norms[2]:.2e}   '
              f'min(h-hb) = {np.min(hcol):7.3f}')

    res = np.array(history['res'])
    mass = np.array(history['mass'])
    energy = np.array(history['energy'])
    E0 = float(ekin0 + epot0)

    print('\n--- summary ---')
    print(f'{"t":>9} {"res_u":>10} {"res_v":>10} {"res_h":>10} {"dM/M":>10} {"dE/E":>10}')
    for i, t in enumerate(history['t']):
        print(f'{t:>9.4f} {res[i,0]:>10.2e} {res[i,1]:>10.2e} {res[i,2]:>10.2e} '
              f'{abs(mass[i]/float(mass0)-1):>10.2e} {abs(energy[i]/E0-1):>10.2e}')

    _plot(history, res, mass, energy, float(mass0), E0, snapshots)
    print(f'\nfigures written to {pm.out_path}')


def _plot(history, res, mass, energy, mass0, E0, snapshots) -> None:
    t = history['t']

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for i, name in enumerate(('u', 'v', 'h')):
        ax[0].semilogy(t, res[:, i], 'o-', label=f'$r_{name}$')
    ax[0].set(xlabel='t (s)', ylabel='max|residual| / max|term|',
              title='PDE residual, normalised')
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].semilogy(t, np.abs(mass / mass0 - 1) + 1e-18, 'o-', label='mass')
    ax[1].semilogy(t, np.abs(energy / E0 - 1) + 1e-18, 's-', label='energy')
    ax[1].set(xlabel='t (s)', ylabel='relative drift', title='Conservation')
    ax[1].legend(); ax[1].grid(alpha=.3)

    ax[2].plot(t, history['hmin'], 'o-')
    ax[2].axhline(0.0, color='r', ls='--', label='ill-posed below')
    ax[2].set(xlabel='t (s)', ylabel='min(h - hb)', title='Water column')
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{pm.out_path}/residual_summary.png', dpi=150)
    plt.close(fig)

    n = len(snapshots)
    fig, ax = plt.subplots(2, n, figsize=(3.1 * n, 6.4), squeeze=False)
    for j, (tj, hh, rh) in enumerate(snapshots):
        im = ax[0, j].imshow(hh.T, origin='lower', cmap='viridis',
                             extent=[0, pm.Lx, 0, pm.Ly])
        ax[0, j].set_title(f'h, t={tj:.3f}')
        fig.colorbar(im, ax=ax[0, j], fraction=.046)
        im = ax[1, j].imshow(rh.T, origin='lower', cmap='RdBu_r',
                             extent=[0, pm.Lx, 0, pm.Ly])
        ax[1, j].set_title(f'$r_h$, t={tj:.3f}')
        fig.colorbar(im, ax=ax[1, j], fraction=.046)
    fig.tight_layout()
    fig.savefig(f'{pm.out_path}/residual_fields.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
