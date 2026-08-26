import numpy as np

# Case 1 stopped at the resolved-front limit. Identical in every respect except T, so its
# 111 snapshots are the leading 111 of case 1 and the two can be compared directly.

# Domain and grid
# The domain is periodic and purely numerical, so it matches neither the 170 x 70 cm tank
# nor the 30 x 30 cm measurement window: it only has to hold the pulse, the bump and the
# travel distance without anything wrapping into the region of interest. Square cells,
# dx = dy = 0.078 cm. The resolution is set by the steepened front, not the bump: the front
# narrows to ~0.6 cm by t = 0.42 s, which is ~8 points here but only 4 at half this
# resolution, too coarse for a spectral method and enough to ring. Note the measured front
# sharpens as the grid refines, so 8 points is a floor, not a margin: T is set to stop at
# that limit rather than past it, which is the whole point of this case.
Lx = 80.0            # domain size in x (cm)
Ly = 40.0            # domain size in y (cm)
Nx = 1024
Ny = 512

# Physical parameters
g = 981.0            # gravitational acceleration (cm/s^2)
h_rest = 2.0         # rest water height (cm)

# Bottom topography (Gaussian bump)
H0 = 1.0             # bump height, half the rest depth
R = 1.25             # the tank bump is 1 cm tall on a circular base 5 cm across, so it
                     # vanishes at r = 2.5 cm. A Gaussian never does, so R is set to put
                     # that base at 2R, where exp(-r^2/R^2) has fallen to 1.8% of H0.
xb = 40.0
yb = Ly/2

# Initial pulse
# s = 3 cm matches the wave in the tank: the visible hump is about 3s = 9 cm across, with
# a dominant wavelength of 23 cm. Note this is an intermediate-depth wave, not a strictly
# shallow one: k*h = 0.53, and the shallow water phase speed is 5.5% high against the
# exact linear dispersion relation. That error is the dominant modelling limitation here
# and should be quoted alongside any comparison with experiment. A wider pulse would
# reduce it -- 2.3% at s=5, 1.0% at s=8, both still fitting the 30 cm window -- but it
# also weakens the measurement: the 5 cm bump is already well below the wavelength, so
# lengthening the wave cuts the deformation being measured. At s=3 the peak leaves the
# bump at 109% of A; widening buys accuracy that the experiment cannot resolve.
A = 0.2             # height amplitude (cm)
s = 3.0              # half-length: the surface hump is ~3s = 9 cm across
n = 2                # exponent
x0 = 27.5            # pulse center, 12.5 cm upstream of the bump
U = np.sqrt(g/h_rest)*A   # velocity amplitude

# Time integration
# c = sqrt(g*h_rest) = 44.3 cm/s. The crest rides deeper water than the base and so runs
# faster, shearing the wave forward until its leading face goes vertical. Whitham (Linear
# and Nonlinear Waves, 1974, section 2.1) puts that gradient catastrophe at
#     t_break = 2*h_rest / (3*c*|d(eta)/dx|_max) = 0.53 s
# using the exact Gaussian maximum slope A*sqrt(2/e)/s. The face narrows roughly as
# (1 - t/t_break), and the grid needs ~8 points across it, which caps a trustworthy run
# near T = 0.43 s here. Note dt does nothing for this -- the face narrows in x, so only Nx
# helps, and only up to t_break, beyond which the solution is genuinely discontinuous.
#
# This is where this case differs from outs/1: it stops at that limit instead of running
# to 1 s, so every snapshot it holds is quantitative and none of them shows the ringing
# that follows once the front goes under ~8 points. The measured front gives 7 points
# across at t = 0.42 s, so 0.44 s sits right at the edge -- the last few snapshots are the
# marginal ones and are worth checking rather than trusting. Use outs/1 to look at the
# breakdown itself; use this case for anything measured.
#
# Unlike T = 1, this T divides dt exactly: Nt = 22000 with no truncation, 111 snapshots at
# t = 0.004*k for k = 0..110, and the last one lands on t = 0.44 s exactly.
dt = 2e-5
T = 0.44            # total simulated time (s), at the resolved-front limit
ostep = 200         # output step (111 snapshots, ~1.4 GB across the three fields)
bstep = 50          # balance step

# No paths here: time_marching.py sets out_path, hb_path and data_path to the case
# directory this file was loaded from, so a params.py copied between cases can never
# redirect output into the wrong run.

# Data generation / assimilation flags (used by SWHD_2D)
make_data = True
noise = False
uum_noise_std = 0.0
vvm_noise_std = 0.0
hhm_noise_std = 0.0
iit = 0
iit0 = 0
iitN = 1


# Fields. time_marching.py calls these with the grid meshes and uses whatever they
# return, so the shape of the run lives here rather than in the driver.

def bathymetry(X, Y):
    ''' Isolated Gaussian bump centered on (xb, yb). '''
    return H0*np.exp(-((X-xb)**2 + (Y-yb)**2)/R**2)


def initial_fields(X, Y):
    ''' Super-Gaussian pulse travelling in +x, uniform in y. Returns [u0, v0, h0].

    h is the free surface, so it sits on top of the rest height: without that offset
    h-hb goes negative over the bump and the run turns into NaNs.
    '''
    pulse = np.exp(-((X-x0)/s)**n)
    return [U*pulse, np.zeros_like(X), h_rest + A*pulse]
