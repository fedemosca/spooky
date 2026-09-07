import numpy as np

# Case 3 stopped short: case 2_short with a richer bar. Same grid, same initial state,
# same T = 0.46 s; only the bathymetry differs. Instead of a plain Gaussian ridge the bar
# is an enveloped sum of two cosines -- one long, one short -- renormalized so its crest
# is still exactly 1 cm. Everything else is copied from outs/2_short unchanged.

# Domain and grid
# The domain is periodic and purely numerical, so it matches neither the 170 x 70 cm tank
# nor the 30 x 30 cm measurement window: it only has to hold the pulse, the bar and the
# travel distance without anything wrapping into the region of interest. Square cells,
# dx = dy = 0.078 cm. The resolution is set by the steepened front, not the bar: the
# front narrows to ~0.6 cm by t = 0.42 s, which is ~8 points here but only 4 at half this
# resolution, too coarse for a spectral method and enough to ring. Note the measured front
# sharpens as the grid refines, so 8 points is a floor, not a margin: T is set to stop
# near that limit rather than well past it, which is the point of this case.
Lx = 80.0            # domain size in x (cm)
Ly = 40.0            # domain size in y (cm)
Nx = 1024
Ny = 512

# Physical parameters
g = 981.0            # gravitational acceleration (cm/s^2)
h_rest = 2.0         # rest water height (cm)

# Bottom topography (modulated bar)
# A Gaussian envelope carrying two cosines, uniform in y like case 2's ridge:
#     raw = exp(-(x-xb)^2/R^2) * (a1*cos(2*pi*(x-xb)/L1) + a2*cos(2*pi*(x-xb)/L2))
#     hb  = H0 * raw / (a1 + a2)
# Both cosines peak at x = xb, so raw peaks there too at exactly a1 + a2, and dividing by
# that puts the crest at H0 = 1 cm with no dependence on the grid. a1 and a2 therefore set
# no amplitude of their own -- only their ratio, together with L1, L2 and R, sets the
# shape. The envelope is what keeps the bar compact: bare cosines would tile the whole
# periodic domain with topography.
#
# The oscillation is not a small ripple on a Gaussian. The two cosines go negative away
# from the crest, so the bar is flanked by shallow trenches reaching -0.126 cm -- a
# genuinely different bottom from case 2, not a perturbation of it.
H0 = 1.0             # crest height (cm), half the rest depth -- same as cases 1 and 2
R = 2.5              # envelope width (cm). Twice case 2's R: at R = 1.25 the envelope is
                     # narrower than L1 and crushes the oscillation back into a plain
                     # Gaussian (trenches of only -0.05 cm). The bar is correspondingly
                     # wider -- |hb| > 1e-3 out to 5.86 cm from the center, against 2.9 cm
                     # for case 2 -- so this case breaks the "same along-x width as the
                     # case 1 bump" invariant that cases 1 and 2 share.
a1 = 0.15            # long-cosine weight
L1 = 8.0             # long wavelength (cm), somewhat wider than the envelope
a2 = 0.08            # short-cosine weight
L2 = 2.5             # short wavelength (cm), 32 grid points across -- well resolved
xb = 40.0
                     # no yb: the bar has no y dependence to center

# Initial pulse
# s = 3 cm matches the wave in the tank: the visible hump is about 3s = 9 cm across, with
# a dominant wavelength of 23 cm. Note this is an intermediate-depth wave, not a strictly
# shallow one: k*h = 0.53, and the shallow water phase speed is 5.5% high against the
# exact linear dispersion relation. That error is the dominant modelling limitation here
# and should be quoted alongside any comparison with experiment. A wider pulse would
# reduce it -- 2.3% at s=5, 1.0% at s=8, both still fitting the 30 cm window -- but it
# also weakens the measurement: the bar is already well below the wavelength, so
# lengthening the wave cuts the deformation being measured.
#
# The wider bar leaves less clear water ahead of the pulse than in case 2: its leading
# edge sits at x = 34.1 instead of 37.1, where the pulse is still 0.0074 of its peak. That
# is small enough not to matter but is no longer entirely negligible.
A = 0.2             # height amplitude (cm)
s = 3.0              # half-length: the surface hump is ~3s = 9 cm across
n = 2                # exponent
x0 = 27.5            # pulse center, 12.5 cm upstream of the bar crest
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
# T is kept at case 2_short's 0.46 s so the two are directly comparable. Take the 0.43 s
# figure as a flat-bottom guide only: this bar shallows the whole width at the crest just
# as case 2's ridge does, so the transmitted wave steepens faster and the front is
# expected to go under ~8 points somewhat earlier. T = 0.46 s therefore sits at or
# slightly past the edge: measure the front on hhms.npy before treating the last snapshots
# as quantitative.
#
# This T divides dt exactly: Nt = 23000 with no truncation, 116 snapshots at
# t = 0.004*k for k = 0..115, and the last one lands on t = 0.46 s exactly.
dt = 2e-5
T = 0.46            # total simulated time (s), near the resolved-front limit
ostep = 200         # output step (116 snapshots, ~1.5 GB across the three fields)
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
    ''' Gaussian-enveloped sum of two cosines, uniform in y, crest normalized to H0.

    Both cosines are 1 at X = xb, so the unnormalized field peaks there at exactly
    a1 + a2; dividing by that sum pins the crest at H0 independently of the grid.
    '''
    raw = np.exp(-(X-xb)**2/R**2)*(a1*np.cos(2*np.pi*(X-xb)/L1)
                                   + a2*np.cos(2*np.pi*(X-xb)/L2))
    return H0*raw/(a1 + a2)


def initial_fields(X, Y):
    ''' Identical to cases 1 and 2: super-Gaussian pulse travelling in +x, uniform in y.

    h is the free surface, so it sits on top of the rest height: without that offset
    h-hb goes negative over the bar and the run turns into NaNs.
    '''
    pulse = np.exp(-((X-x0)/s)**n)
    return [U*pulse, np.zeros_like(X), h_rest + A*pulse]
