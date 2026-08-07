'''
2D Shallow water equations over a Gaussian bump
'''

import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spooky as sp
from spooky.solvers import SWHD_2D

import params as pm

os.makedirs(pm.out_path, exist_ok=True)

grid = sp.Grid2D(Lx=pm.Lx, Ly=pm.Ly, Nx=pm.Nx, Ny=pm.Ny, dt=pm.dt)
solver = SWHD_2D(grid, pm)

Xs, Ys = grid.xx, grid.yy

# Bottom topography
hb = pm.H0*np.exp(-((Xs-pm.xb)**2 + (Ys-pm.yb)**2)/pm.R**2)
np.save(f'{pm.hb_path}/hb.npy', hb)
solver.update_hb(hb)
solver.update_true_hb()

# Initial conditions: super-Gaussian pulse travelling in +x
h0 = pm.A*np.exp(-((Xs-pm.x0)/pm.s)**pm.n)
u0 = pm.U*np.exp(-((Xs-pm.x0)/pm.s)**pm.n)
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
