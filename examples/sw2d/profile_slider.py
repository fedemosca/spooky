'''
Interactive view of the initial wave profile along x, with the pulse half-length on a
slider. The wave is uniform in y, so the cut through the middle of the domain is the
whole story.

Everything except s is read from params.py. The readout tracks the three quantities that
decide whether the simulation is worth running: how much of the 30 cm measurement window
the wave occupies, how far outside the shallow water regime it sits, and how large an
amplitude it can carry before steepening into a shock.

Local tool -- needs an interactive matplotlib backend, unlike the cluster scripts.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import params as pm

FOV = 30.0          # measurement window (cm)
S_MIN, S_MAX = 1.0, 10.0


def surface(xx: np.ndarray, s: float) -> np.ndarray:
    ''' Free surface: rest height plus the pulse. '''
    return pm.h_rest + pm.A * np.exp(-((xx - pm.x0)/s)**pm.n)


def bottom(xx: np.ndarray) -> np.ndarray:
    return pm.H0 * np.exp(-(xx - pm.xb)**2 / pm.R**2)


def dispersion_error(s: float) -> float:
    ''' Spectrum-weighted error of the shallow water phase speed against the exact
    linear dispersion relation, which is what the model gets wrong at finite k*h. '''
    k = np.linspace(1e-6, 40/s, 20000)
    weight = np.exp(-k**2 * s**2 / 2)
    c_exact = np.sqrt(pm.g * np.tanh(k*pm.h_rest) / k)
    return 1 - np.sum(weight*c_exact)/np.sum(weight)/np.sqrt(pm.g*pm.h_rest)


def max_amplitude(s: float) -> float:
    ''' Largest A whose crest still clears the bump within 0.8 of the shock time. '''
    return 0.8 * pm.h_rest / (1.5 * (2.5*s + 2*pm.R) / s)


def readout(s: float) -> str:
    lam = 7.85 * s                      # dominant wavelength of a Gaussian of width s
    kh = 2*np.pi * pm.h_rest / lam
    span = 3*s + 2*pm.R
    return (f'hump 3s = {3*s:.0f} cm      wavelength = {lam:.0f} cm      k*h = {kh:.2f}\n'
            f'shallow water error = {dispersion_error(s):.1%}      '
            f'max A = {10*max_amplitude(s):.1f} mm      '
            f'visible span = {span:.0f} cm {"(fits)" if span <= FOV else "(EXCEEDS window)"}')


def main() -> None:
    xx = np.linspace(0, pm.Lx, 2000)
    hb = bottom(xx)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(bottom=0.26, top=0.86)

    ax.axvspan(pm.xb - FOV/2, pm.xb + FOV/2, color='0.9', label=f'{FOV:.0f} cm window')
    ax.fill_between(xx, 0, hb, color='0.45', label='bottom')
    line, = ax.plot(xx, surface(xx, pm.s), lw=2, color='tab:blue', label='free surface')
    water = ax.fill_between(xx, hb, surface(xx, pm.s), color='tab:blue', alpha=.25)
    ax.axhline(pm.h_rest, ls=':', color='0.5', lw=1)

    ax.set(xlabel='x (cm)', ylabel='height (cm)', xlim=(0, pm.Lx),
           ylim=(0, pm.h_rest + 4*pm.A))
    ax.legend(loc='upper right', fontsize=9)
    title = ax.set_title(readout(pm.s), fontsize=9, family='monospace', loc='left')

    slider = Slider(fig.add_axes([0.12, 0.10, 0.76, 0.03]), 'half-length s (cm)',
                    S_MIN, S_MAX, valinit=pm.s, valstep=0.1)

    def update(_) -> None:
        nonlocal water
        hh = surface(xx, slider.val)
        line.set_ydata(hh)
        water.remove()
        water = ax.fill_between(xx, hb, hh, color='tab:blue', alpha=.25)
        title.set_text(readout(slider.val))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
    main()
