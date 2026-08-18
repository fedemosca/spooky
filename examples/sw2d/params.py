import numpy as np

# Domain and grid
# The domain is periodic and purely numerical, so it matches neither the 170 x 70 cm tank
# nor the 30 x 30 cm measurement window: it only has to hold the pulse, the bump and the
# travel distance without anything wrapping into the region of interest. Square cells,
# dx = dy = 0.234 cm, which puts ~10 points across the bump radius.
Lx = 120.0           # domain size in x (cm)
Ly = 60.0            # domain size in y (cm)
Nx = 512
Ny = 256

# Physical parameters
g = 981.0            # gravitational acceleration (cm/s^2)
h_rest = 2.0         # rest water height (cm)

# Bottom topography (Gaussian bump)
H0 = 1.0             # bump height, half the rest depth
R = 2.5              # bump width: exp(-r^2/R^2) is 5 cm across at 1/e, as in the tank
xb = 60.0
yb = Ly/2

# Initial pulse
# s is a compromise between two opposed constraints. Shallow water assumes k*h << 1, and
# the spectrum-weighted error in the wave speed falls as the pulse widens: 21% at s=1,
# 3.4% at s=4, 1.3% at s=7. Against that, the 30 x 30 cm measurement window has to hold
# the pulse and the bump at once, and the visible span is 3s + 5 cm. s=7 gives 26 cm,
# which fits with margin, at an error small enough that the bump geometry and the
# amplitude dominate any comparison with experiment.
A = 0.25             # height amplitude (cm)
s = 7.0              # half-length
n = 2                # exponent
x0 = 37.5            # pulse center, 22.5 cm upstream of the bump
U = np.sqrt(g/h_rest)*A   # velocity amplitude

# Time integration
# c = sqrt(g*h_rest) = 44.3 cm/s. The wave steepens as it travels: a simple wave shocks
# after t_shock = s/(1.5*(A/h_rest)*c) = 0.84 s, and stays resolved on this grid until
# roughly 0.8 of that. T = 0.6 s carries the crest 27 cm, clear of the bump, and stays
# inside that limit. Amplitude trades directly against how far the wave can be followed:
# A = 0.25 cm lets the crest clear the bump, while following the whole 35 cm envelope
# past it would need A <= 0.17 cm. At the 5 mm end of the experimental range the wave
# breaks before reaching the bump at all, which this solver cannot represent as it has
# no dissipation.
dt = 5e-5
T = 0.6             # total simulated time (s)
ostep = 80          # output step (151 snapshots, ~0.48 GB across the three fields)
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
