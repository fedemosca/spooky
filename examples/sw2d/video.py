'''
Animate a stacked output field (nt, Nx, Ny) as an mp4, with a profile cut along x.

The mp4 is written next to the input array, and params.py and hb.npy are read from that
same directory.

Usage: python video.py ./outs/1/hhms.npy
'''

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
# The cluster has no ffmpeg on PATH; imageio-ffmpeg ships a static binary, but matplotlib
# only ever spawns what this rcParam points at.
import imageio_ffmpeg
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

fps = 10
dpi = 150

npy_path = os.path.abspath(sys.argv[1])
run_path = os.path.dirname(npy_path)
name = os.path.basename(npy_path).removesuffix('.npy')

# Snapshot times come from the params.py sitting next to the data, so the labels match
# the run that produced it rather than whatever is current in this directory.
sys.path.insert(0, run_path)
import params as pm
dt_out = pm.dt*pm.ostep

field = np.load(npy_path, mmap_mode='r')
if field.ndim != 3:
    raise ValueError(f'expected a (nt, Nx, Ny) array, got shape {field.shape}')

# h is the free surface, so it sits around h_rest: without removing that offset the
# diverging map is uniformly red and the wave is invisible. The velocities are already
# centered on zero.
is_h = name.startswith('hh')
offset = pm.h_rest if is_h else 0.0
label = f'{name} - h_rest' if offset else name

# Axes in cm. The grid spacing comes from the array shape rather than pm.Nx/pm.Ny so that
# the extent is right even if a params.py was edited after the run that produced the data.
dx = pm.Lx/field.shape[1]
dy = pm.Ly/field.shape[2]
xs = np.arange(field.shape[1])*dx

jcut = field.shape[2]//2
ycut = jcut*dy
if is_h:
    hb_cut = np.load(os.path.join(run_path, 'hb.npy'))[:, jcut] - offset

# Fixed color scale across the whole series, otherwise each frame renormalizes and the
# amplitude evolution is invisible. Percentiles rather than min/max so that a single
# steep frame does not flatten the rest, and symmetric so that white is exactly zero.
lo, hi = np.percentile(np.asarray(field) - offset, [0.5, 99.5])
vmax = max(abs(lo), abs(hi))

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]},
                               constrained_layout=True)

# Cell centers sit at i*dx, so the pixel edges run half a cell either side of the domain.
extent = [-0.5*dx, pm.Lx - 0.5*dx, -0.5*dy, pm.Ly - 0.5*dy]
im = ax0.imshow(field[0].T - offset, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                extent=extent)
ax0.axhline(ycut, color='k', ls='--', lw=0.8)
ax0.set_ylabel('y [cm]')
title = ax0.set_title(f'{label}  |  t = 0.0000 s')
fig.colorbar(im, ax=[ax0, ax1], label=label)

prof, = ax1.plot(xs, field[0, :, jcut] - offset, color='k', lw=1.0)
ax1.set_xlabel('x [cm]')
ax1.set_ylabel(f'{label} at y = {ycut:.1f} cm')
ax1.axhline(0.0, color='0.7', lw=0.5)

if is_h:
    # Show the water column: the surface line above, the bump below, fluid shaded
    # between. That fixes the panel range to the full depth, so the wave itself is a
    # small wiggle near the top -- the point here is the wave relative to the bottom.
    ax1.plot(xs, hb_cut, color='0.4', lw=1.0)
    fill = ax1.fill_between(xs, hb_cut, field[0, :, jcut] - offset, color='tab:blue', alpha=0.3)
    ymin, ymax = hb_cut.min(), vmax
else:
    ymin, ymax = -vmax, vmax
pad = 0.05*(ymax - ymin)
ax1.set_ylim(ymin - pad, ymax + pad)

out_file = os.path.join(run_path, f'{name}.mp4')

writer = FFMpegWriter(fps=fps, bitrate=-1)
with writer.saving(fig, out_file, dpi):
    for i, frame in enumerate(field):
        im.set_data(frame.T - offset)
        cut = frame[:, jcut] - offset
        prof.set_ydata(cut)
        if is_h:
            fill.remove()
            fill = ax1.fill_between(xs, hb_cut, cut, color='tab:blue', alpha=0.3)
        title.set_text(f'{label}  |  t = {i*dt_out:.4f} s')
        writer.grab_frame()
plt.close(fig)

print(f'wrote {out_file}')
