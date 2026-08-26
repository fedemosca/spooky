import numpy as np

# Case 1 with the bump extended into a ridge. Every parameter below is identical to
# outs/1, and so is initial_fields; only bathymetry differs. The three cases differ only
# in the bottom: outs/0 flat, outs/1 an isolated bump, outs/2 that same bump profile
# spanning the width.

# Domain and grid
# The domain is periodic and purely numerical, so it matches neither the 170 x 70 cm tank
# nor the 30 x 30 cm measurement window: it only has to hold the pulse, the ridge and the
# travel distance without anything wrapping into the region of interest. Square cells,
# dx = dy = 0.078 cm. The resolution is set by the steepened front, not the ridge: the
# front narrows to ~0.6 cm by t = 0.42 s, which is ~8 points here but only 4 at half this
# resolution, too coarse for a spectral method and enough to ring. Note the measured front
# sharpens as the grid refines, so 8 points is a floor, not a margin: the run continues to
# T = 1 s past the point where the front is resolved, so treat anything after ~0.43 s as a
# study of the breakdown rather than as a solution.
Lx = 80.0            # domain size in x (cm)
Ly = 40.0            # domain size in y (cm)
Nx = 1024
Ny = 512

# Physical parameters
g = 981.0            # gravitational acceleration (cm/s^2)
h_rest = 2.0         # rest water height (cm)

# Bottom topography (Gaussian ridge)
# The case 1 bump profile along y = yb, extended over the whole width:
#     hb = H0*exp(-(x-xb)^2/R^2)
# Same height and same along-x width as the bump, but no y dependence, so it is a
# submerged bar across the domain rather than an isolated obstacle. Being independent of
# y it is exactly periodic in y, which the isolated bump is only approximately.
H0 = 1.0             # ridge height, half the rest depth
R = 1.25             # the tank bump is 1 cm tall on a base 5 cm across, so it vanishes at
                     # r = 2.5 cm. A Gaussian never does, so R is set to put that base at
                     # 2R, where exp(-r^2/R^2) has fallen to 1.8% of H0.
xb = 40.0
                     # no yb: a ridge has no y dependence to center

# Initial pulse
# s = 3 cm matches the wave in the tank: the visible hump is about 3s = 9 cm across, with
# a dominant wavelength of 23 cm. Note this is an intermediate-depth wave, not a strictly
# shallow one: k*h = 0.53, and the shallow water phase speed is 5.5% high against the
# exact linear dispersion relation. That error is the dominant modelling limitation here
# and should be quoted alongside any comparison with experiment. A wider pulse would
# reduce it -- 2.3% at s=5, 1.0% at s=8, both still fitting the 30 cm window -- but it
# also weakens the measurement: the 5 cm ridge is already well below the wavelength, so
# lengthening the wave cuts the deformation being measured.
A = 0.2             # height amplitude (cm)
s = 3.0              # half-length: the surface hump is ~3s = 9 cm across
n = 2                # exponent
x0 = 27.5            # pulse center, 12.5 cm upstream of the ridge
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
# T = 1 s runs deliberately well past that, matching cases 0 and 1 step for step so the
# three can be compared at every snapshot: oscillations behind the crest after ~0.43 s are
# the expected outcome, not a bug. Use only the early snapshots for anything quantitative.
# The crest reaches x ~ 74 cm by t = 1 s, still short of the periodic edge at Lx = 80, so
# nothing wraps back into the region of interest.
dt = 2e-5
T = 1               # total simulated time (s)
ostep = 200         # output step (250 snapshots, ~3.1 GB across the three fields)
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
    ''' Case 1's bump profile along y = yb, extended over the full width.

    Same height and same along-x width as the bump, with no y dependence: a submerged
    bar across the domain rather than an isolated obstacle.
    '''
    return H0*np.exp(-(X-xb)**2/R**2)


def initial_fields(X, Y):
    ''' Identical to case 1: super-Gaussian pulse travelling in +x, uniform in y.

    h is the free surface, so it sits on top of the rest height: without that offset
    h-hb goes negative over the ridge and the run turns into NaNs.
    '''
    pulse = np.exp(-((X-x0)/s)**n)
    return [U*pulse, np.zeros_like(X), h_rest + A*pulse]
