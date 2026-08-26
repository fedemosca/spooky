'''
2D Shallow water equations over a Gaussian bump

Usage: python3 time_marching.py ./outs/1

The case directory must already exist and hold a params.py. Every output of the run is
written back into it, so the parameters that produced a run always sit beside it.
'''

import os
import sys
import importlib.util

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spooky as sp
from spooky.solvers import SWHD_2D


def load_params(case_dir: str):
    ''' Imports <case_dir>/params.py and points its output paths at <case_dir>. '''
    if not os.path.isdir(case_dir):
        raise SystemExit(f'case directory not found: {case_dir}')
    params_file = os.path.join(case_dir, 'params.py')
    if not os.path.isfile(params_file):
        raise SystemExit(f'no params.py in {case_dir}')

    spec = importlib.util.spec_from_file_location('params', params_file)
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)

    # The case owns its directory: the run lands here no matter what the case file says,
    # so copying a params.py between cases can never redirect output to the wrong run.
    pm.out_path = pm.hb_path = pm.data_path = case_dir
    return pm


if len(sys.argv) != 2:
    raise SystemExit(f'usage: {sys.argv[0]} <case_dir>')

case_dir = os.path.normpath(sys.argv[1])
pm = load_params(case_dir)

grid = sp.Grid2D(Lx=pm.Lx, Ly=pm.Ly, Nx=pm.Nx, Ny=pm.Ny, dt=pm.dt)
solver = SWHD_2D(grid, pm)

Xs, Ys = grid.xx, grid.yy

# Bottom topography
hb = pm.H0*np.exp(-((Xs-pm.xb)**2 + (Ys-pm.yb)**2)/pm.R**2)
np.save(f'{pm.hb_path}/hb.npy', hb)
solver.update_hb(hb)
solver.update_true_hb()

# Initial conditions: super-Gaussian pulse travelling in +x. h is the free surface, so
# it sits on top of the rest height: without that offset h-hb is negative over the bump,
# the water column is unphysical and the run turns into NaNs.
pulse = np.exp(-((Xs-pm.x0)/pm.s)**pm.n)
h0 = pm.h_rest + pm.A*pulse
u0 = pm.U*pulse
v0 = np.zeros_like(Xs)
fields = [u0, v0, h0]

# Evolve
fields = solver.evolve(fields, T=pm.T, bstep=pm.bstep, ostep=pm.ostep,
                       bpath=pm.out_path, opath=pm.out_path)

# # Plot final fields
# uu, vv, hh = fields
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# for axi, (ff, name) in zip(ax, [(uu, 'uu'), (vv, 'vv'), (hh, 'hh')]):
#     im = axi.imshow(ff.T, origin='lower', cmap='viridis',
#                     extent=[0, pm.Lx, 0, pm.Ly])
#     axi.set_title(name)
#     fig.colorbar(im, ax=axi)
# plt.savefig(f'{pm.out_path}/fields.png', dpi=300)
# plt.close()
