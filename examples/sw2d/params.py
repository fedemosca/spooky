import numpy as np

# Domain and grid
# The domain is periodic and purely numerical: it does not have to match the tank, it
# only has to be long enough that nothing wraps into the region of interest during the
# run. A shallow water pulse is long (the envelope below spans ~50 cm), so once the
# travel distance and the bump are added the requirement lands near 150 cm -- close
# enough to the real 170 x 70 cm tank that matching it costs nothing and keeps the
# geometry comparable. Cells are near-square: dx = 0.332, dy = 0.313.
Lx = 170.0           # domain size in x (cm)
Ly = 70.0            # domain size in y (cm)
Nx = 512
Ny = 224

# Physical parameters
g = 981.0            # gravitational acceleration (cm/s^2)
h_rest = 2.0         # rest water height (cm)

# Bottom topography (Gaussian bump)
H0 = 1.0             # bump height, half the rest depth
R = 5.0              # bump width
xb = 65.0
yb = Ly/2

# Initial pulse
# s is set by shallow water validity, not by the tank: the equations assume k*h << 1,
# and the spectrum-weighted error in the wave speed is 21% at s=1, 2.3% at s=5 and 0.6%
# at s=10. Below ~1% the approximation stops being the limiting error in any comparison
# with experiment, so s=10 (five times the depth) is the cheapest width that qualifies.
A = 0.2              # height amplitude (cm)
s = 10.0             # half-length
n = 2                # exponent
x0 = 35.0            # pulse center, 30 cm upstream of the bump
U = np.sqrt(g/h_rest)*A   # velocity amplitude

# Time integration
# c = sqrt(g*h_rest) = 44.3 cm/s. The wave steepens as it travels: a simple wave shocks
# after t_shock = s/(1.5*(A/h_rest)*c) = 1.5 s, and stays resolved on this grid until
# roughly 0.8 of that. T is set there, which carries the crest 53 cm -- over the bump and
# ~23 cm beyond it. Raising A shortens t_shock in proportion: at the 5 mm end of the
# experimental range it drops to 0.6 s and the wave breaks before clearing the bump,
# which this solver cannot represent since it has no dissipation.
dt = 5e-5
T = 1.2             # total simulated time (s)
ostep = 120         # output step (201 snapshots, ~0.55 GB across the three fields)
bstep = 100         # balance step

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
