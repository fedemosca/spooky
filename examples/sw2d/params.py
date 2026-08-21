import numpy as np

# Domain and grid
# The domain is periodic and purely numerical, so it matches neither the 170 x 70 cm tank
# nor the 30 x 30 cm measurement window: it only has to hold the pulse, the bump and the
# travel distance without anything wrapping into the region of interest. Square cells,
# dx = dy = 0.078 cm. The resolution is set by the steepened front, not the bump: the
# front narrows to ~0.6 cm by the end of the run, which is ~8 points here but only 4 at
# half this resolution, too coarse for a spectral method and enough to ring. Note the
# measured front sharpens as the grid refines, so 8 points is a floor, not a margin: if
# the production run shows oscillations behind the crest, shorten T rather than trusting
# the late snapshots.
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
# and should be quoted alongside any comparison with experiment. Reaching 1% would need
# s > 8 cm, whose 29 cm span no longer fits the 30 x 30 cm measurement window.
A = 0.25             # height amplitude (cm)
s = 3.0              # half-length: the surface hump is ~3s = 9 cm across
n = 2                # exponent
x0 = 27.5            # pulse center, 12.5 cm upstream of the bump
U = np.sqrt(g/h_rest)*A   # velocity amplitude

# Time integration
# c = sqrt(g*h_rest) = 44.3 cm/s. The wave steepens as it travels: a simple wave shocks
# after t_shock = s/(1.5*(A/h_rest)*c) = 0.36 s, and stays resolved on this grid until
# roughly 0.8 of that. T = 0.28 s carries the crest 12.5 cm, clear of the bump, and stops
# inside that limit. A narrow pulse shocks sooner than a wide one at the same amplitude,
# so raising A much above 0.25 cm leaves no room to follow the wave past the bump.
dt = 2e-5
T = 0.28            # total simulated time (s)
ostep = 200         # output step (71 snapshots, ~0.9 GB across the three fields)
bstep = 50          # balance step

# Paths
out_path = './'
data_path = './'
hb_path = './'

# Data generation / assimilation flags (used by SWHD_2D)
make_data = True
noise = False
uum_noise_std = 0.0
vvm_noise_std = 0.0
hhm_noise_std = 0.0
iit = 0
iit0 = 0
iitN = 1
