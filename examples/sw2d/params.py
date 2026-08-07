import numpy as np

# Domain and grid
Lx = 30.0            # domain size in x (cm)
Ly = 30.0            # domain size in y (cm)
Nx = 512
Ny = 512

# Physical parameters
g = 981.0            # gravitational acceleration (cm/s^2)
h_rest = 3.0         # rest water height (cm)

# Bottom topography (Gaussian bump)
H0 = 2.0             # bump height
R = 5.0              # bump width
xb = Lx/2
yb = Ly/2

# Initial pulse
A = 0.3              # height amplitude
s = 1.0              # half-length
n = 2                # exponent
x0 = Lx/5            # pulse center
U = np.sqrt(g/h_rest)*A   # velocity amplitude

# Time integration
dt = 1e-5
T = 0.06          # total simulated time (s)
ostep = 100         # output step
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
