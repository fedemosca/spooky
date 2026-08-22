'''
Post-process filter for saved sw2d snapshots.

Applies an exponential filter exp(-alpha*(k/k_dealias)^P) to each frame and writes
*_filtered.npy alongside the originals, which are left untouched. Nothing here feeds back
into the solver: this is cosmetic, for frames whose tail has picked up grid-scale ringing
as the wave front steepens.

The filter is smooth on purpose. Zeroing the modes above a cutoff instead convolves the
field with a sinc and manufactures the very oscillations it is meant to remove -- on a
front at 0.9 t_break that puts 2e-5 cm of ripple into water that should be flat, against
3e-17 for the exponential.

Read the diagnostics it prints before trusting a frame. Measured on the A = 2 mm run in
2.0/: while the front is still resolved the filter removes essentially nothing, which is
the point -- there is no grid-scale content to take. Once the front falls below a few
points it removes ~79% of the ringing but costs 15% of the front slope and 3% of the peak
height. So it cleans the picture without repairing the solution: ringing only appears
after the front stops being resolved, and by then the dynamics that produced the frame
were already wrong. Treat a frame that needs heavy filtering as a marker of where to stop
trusting the run, not as a frame that has been fixed.
'''

import numpy as np

import params as pm

# P is calibrated on the A = 2 mm run in 2.0/, at 1024x512 out to t = 1.9 t_break. On a
# broken frame the removed ripple is flat at ~79% for every P up to 32 and only falls off
# beyond it, while the damage to the wave shrinks steadily with P: at P=8 the front slope
# loses 37% and the peak 6%, at P=32 only 15% and 3%. Since the cleanup is the same, the
# gentler filter is strictly better, and P=32 is where it stops being free. Much larger P
# starts leaving ringing behind (67% removed at 48, 43% at 64).
#
# This is a single pass over saved frames. An in-solver filter fires once per substep --
# tens of thousands of times -- where sigma^N compounds and these values would be far too
# aggressive.
P = 32               # filter order: higher confines the damping nearer the grid scale
ALPHA = 36.0         # -log(eps) for float64, so the last retained mode lands at roundoff
FIELDS = ('hhms', 'uums', 'vvms')


def filter_mask(shape: tuple) -> np.ndarray:
    ''' Exponential filter on the grid's dealiasing normalisation.

    Grid2D dealiases at kr > 1/9, so 3*sqrt(kr) runs 0..1 across the retained band and the
    filter reaches full strength exactly where the solver already truncates.
    '''
    nx, ny = shape
    ki = np.fft.fftfreq(nx, 1/nx)
    kj = np.fft.rfftfreq(ny, 1/ny)
    ki, kj = np.meshgrid(ki, kj, indexing='ij')
    kn = 3.0 * np.sqrt((ki/nx)**2 + (kj/ny)**2)
    return np.exp(-ALPHA * kn**P)


def apply_filter(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.fft.irfft2(np.fft.rfft2(frame) * mask, s=frame.shape)


def main() -> None:
    for name in FIELDS:
        try:
            data = np.load(f'{pm.out_path}/{name}.npy')
        except FileNotFoundError:
            print(f'{name}.npy not found, skipping')
            continue

        mask = filter_mask(data.shape[1:])
        out = np.empty_like(data)
        changed = np.empty(len(data))
        for i, frame in enumerate(data):
            out[i] = apply_filter(frame, mask)
            changed[i] = np.max(np.abs(out[i] - frame))

        np.save(f'{pm.out_path}/{name}_filtered.npy', out)

        scale = pm.A if name == 'hhms' else pm.U
        print(f'{name}: {len(data)} frames -> {name}_filtered.npy')
        print(f'  largest change {changed.max():.2e} '
              f'({100*changed.max()/scale:.2f}% of {"A" if name == "hhms" else "U"}), '
              f'first appreciable at frame {int(np.argmax(changed > 0.001*scale))}')
        if changed.max() > 0.02 * scale:
            print(f'  WARNING: the filter is removing more than 2% somewhere. That is not '
                  f'ringing on top of a good solution,\n           it is a frame whose '
                  f'front the grid never resolved. Shorten T instead of filtering harder.')


if __name__ == '__main__':
    main()
