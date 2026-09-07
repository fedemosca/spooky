# SWHD_2D Simulation — Parameter Reference

Case `outs/3_short`. Mirrors the `params.py` in this directory and `time_marching.py`.
Everything below is either read from `params.py` or built in `time_marching.py`; nothing
here is an attribute of `SWHD_2D` unless the table says so.

This is **`outs/2_short` with a richer bar**: same grid, same initial state, same
`T = 0.46` s, same output cadence. The single change is `bathymetry`, which replaces the
plain Gaussian ridge with a Gaussian-enveloped sum of two cosines — one long, one short —
renormalized so the crest is still exactly 1 cm. Anything that differs from `outs/2_short`
is the bottom.

## 1. Domain & Grid

| Parameter | Value | Meaning |
|---|---|---|
| `Lx, Ly` | 80 × 40 | domain size (cm) |
| `Nx, Ny` | 1024 × 512 | grid resolution |
| `dx, dy` | 0.078125 | grid spacing (cm), square cells, `np.linspace(0, L, N, endpoint=False, retstep=True)` |
| `Xs, Ys` | `grid.xx`, `grid.yy` | `np.meshgrid(xi, yi, indexing='ij')` |

Unchanged from `outs/2_short`. The domain is periodic and purely numerical: it matches
neither the 170 × 70 cm tank nor the 30 × 30 cm measurement window, it only has to hold
the pulse, the bar and the travel distance without anything wrapping into the region of
interest.

The resolution is still set by the steepened front (~0.6 cm wide by `t = 0.42` s, ~8
points here), not by the bar: the shortest bottom feature is `L2 = 2.5` cm, 32 points
across.

The bar is independent of `y` and so is the initial pulse, so the whole problem is uniform
in `y`: `v ≡ 0` and every `x`-line is the same. `Ny = 512` is kept only so the cases share
a grid.

## 2. Physical Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `g` | 981 | gravitational acceleration (cm/s²) |
| `h_rest` | 2 | rest water height (cm) |
| `c` (derived) | ≈ 44.29 cm/s | gravity wave speed, `sqrt(g·h_rest)` |
| `c` over the crest (derived) | ≈ 31.32 cm/s | `sqrt(g·(h_rest - H0))`, identical to `outs/2_short` — the crest height is normalized to the same 1 cm |
| `c` over the trenches (derived) | ≈ 45.67 cm/s | `sqrt(g·(h_rest + 0.126))`, **new to this case**: the flanks are deeper than the far field, so the wave speeds up before and after slowing over the crest |

Not present in this implementation: `nu` (no viscosity term in `rkstep`).

## 3. Bottom Topography

The bar is

```
raw = exp(-(x-xb)²/R²) · (a1·cos(2π(x-xb)/L1) + a2·cos(2π(x-xb)/L2))
hb  = H0 · raw / (a1 + a2)
```

uniform in `y`, like case 2's ridge.

| Parameter | Value | Meaning |
|---|---|---|
| `H0` | 1.0 | crest height (cm), half the rest depth — same as cases 1 and 2 |
| `R` | 2.5 | Gaussian envelope width (cm), **twice** case 2's |
| `a1`, `L1` | 0.15, 8.0 | long-cosine weight and wavelength (cm) |
| `a2`, `L2` | 0.08, 2.5 | short-cosine weight and wavelength (cm) |
| `xb` | 40 | bar center in x (cm) |
| `yb` | — | absent: the bar has no y dependence to center |
| `max(hb)` (derived) | 1.000000 cm, at `x = xb` | exact, see the normalization note below |
| `min(hb)` (derived) | −0.126143 cm | the flanking trenches |
| footprint (derived) | `\|hb\| > 1e-3` out to 5.86 cm from `xb` | against 2.9 cm for case 2 |
| points per `L2` (derived) | 32 | `L2/dx` |
| `hb` | — | saved to `{hb_path}/hb.npy` and passed in via `solver.update_hb(hb)` |
| `true_hb` | — | reloaded from `{data_path}/hb.npy` by `solver.update_true_hb()` |

**Normalization.** Both cosines equal 1 at `x = xb`, so `raw` attains its maximum there at
exactly `a1 + a2 = 0.23`. Dividing by that constant rather than by `raw.max()` pins the
crest at `H0` analytically, with no dependence on whether `xb` happens to land on a grid
point (it does — `xb/dx = 512` — so the two agree here to the last bit, but the constant
is the honest form).

`a1` and `a2` therefore set **no amplitude of their own**. Only their ratio, together with
`L1`, `L2` and `R`, sets the shape; the height is fixed at 1 cm by construction.

Three consequences worth keeping in mind when comparing against `outs/2_short`:

- **The bar is flanked by trenches.** The cosines go negative away from the crest, down to
  −0.126 cm, so the bottom dips *below* the far-field level on both sides before rising to
  the crest. This is a different bottom in kind, not a perturbation of the Gaussian ridge.
  The rest column ranges over 1.000 – 2.126 cm instead of 1.000 – 2.000 cm.
- **The footprint roughly doubles**, from ±2.9 cm to ±5.9 cm about `xb`. This case
  therefore breaks the "same along-x width as the case 1 bump" invariant that `outs/1` and
  `outs/2` share — `H0` and `xb` still match them, `R` does not. `R = 1.25` was tried and
  rejected: an envelope narrower than `L1` crushes the oscillation back into a plain
  Gaussian, leaving trenches of only −0.05 cm.
- **It is exactly periodic in y**, like case 2's ridge and unlike case 1's isolated bump.
  In `x` it decays to 6.6e-112 at the domain edge, so the periodic image is absent for
  every practical purpose.

## 4. Initial Conditions

Identical to `outs/2_short`, and so to cases 1 and 2.

| Parameter | Value | Meaning |
|---|---|---|
| `A` | 0.2 | pulse height amplitude (cm) |
| `s` | 3.0 | pulse half-length (cm); the surface hump is ~3s = 9 cm across |
| `n` | 2 | pulse exponent |
| `x0` | 27.5 | pulse center (cm), 12.5 cm upstream of the bar crest |
| `U` (derived) | ≈ 4.43 cm/s | velocity amplitude, `sqrt(g/h_rest)·A` |
| `pulse` | — | `exp(-((Xs-x0)/s)^n)` — a function of x only, uniform in y |
| `h0` | — | `h_rest + A·pulse` — free surface, **offset by the rest height** |
| `u0` | — | `U·pulse` |
| `v0` | — | zeros |

`h` is the free surface, so it sits on top of `h_rest`: without that offset `h - hb` is
negative over the bar and the run turns into NaNs. The minimum initial column is
`min(h0 - hb) = 1.0000` cm, along the crest line — the same figure as `outs/2_short`,
since the crest height is the same.

**One thing the wider bar changes.** Its leading edge now sits at `x = 34.1` instead of
`37.1`, where the initial pulse is still 0.0074 of its peak. The pulse and the bottom
therefore overlap slightly at `t = 0`, which they effectively did not in case 2. It is
small enough not to matter for the dynamics, but it is no longer exactly zero, and it is
worth stating rather than assuming the initial state is "over flat bottom".

The pulse matches the wave in the tank (dominant wavelength 23 cm), but it is an
intermediate-depth wave: `k·h = 0.53`, and the shallow-water phase speed is 5.5% high
against the exact linear dispersion relation. That is the dominant modelling limitation
and should be quoted alongside any comparison with experiment.

Fields are handed to the solver as `fields = [u0, v0, h0]`, matching
`fields[0]=fu, fields[1]=fv, fields[2]=fh` inside `rkstep`.

## 5. Time Integration

Identical to `outs/2_short` in every entry.

| Parameter | Value | Meaning |
|---|---|---|
| `dt` | 2e-5 | timestep (s) |
| `T` | 0.46 | total simulated time (s) |
| `ostep` | 200 | steps between saved snapshots |
| `bstep` | 50 | steps between balance writes |
| `rkord` | 2 | Runge-Kutta order, `SWHD_2D` constructor default |
| `Nt` (derived) | 23 000 | `int(T/dt)`, exactly — `0.46/2e-5` lands on 23 000.0 in binary, so nothing truncates |
| `total_steps` (derived) | 23 001 | `int(T/dt) + 1` |
| snapshots saved (derived) | 116 | `(total_steps - 1)//ostep + 1`, length of `uus`/`vvs`/`hhs` |
| snapshot memory (derived) | ≈ 1.5 GB | 116 × 1024 × 512 × 8 B × 3 fields |
| balance rows (derived) | 462 | steps 0, 50, …, 23 000 (461 rows) plus the `final=True` write, also at step 23 000 |
| CFL number (derived) | ≈ 1.13 × 10⁻² | `c·dt/dx` |

Snapshot `k` is the field at step `k·ostep`, i.e. `t = 0.004·k`, for `k = 0…115`, and
`k = 115` is `t = 0.46` s exactly. Slot 115 is written twice with the same field — once in
the loop at step 23 000 and again by the `final=True` write at that same step — so the
duplicate is a no-op. The same is true of the last row of `balance.dat`, which repeats
`t = 0.46`.

Since `T` here is a multiple of `dt`, the `dt = T - Nt·dt` short final step is zero and
`evolve` breaks out rather than taking it. The `make_data` dump fires because the loop
reaches `step == total_steps - 1`.

`T = 0.46` s is kept from `outs/2_short` so the two runs are snapshot-for-snapshot
comparable. It stops near the resolved-front limit rather than well past it: the crest
rides deeper water than the base and shears the wave forward until its leading face goes
vertical, and Whitham (*Linear and Nonlinear Waves*, 1974, §2.1) puts that gradient
catastrophe at

```
t_break = 2·h_rest / (3·c·|dη/dx|_max) = 0.53 s
```

using the exact Gaussian maximum slope `A·sqrt(2/e)/s = 0.0572 cm⁻¹`. The face narrows
roughly as `(1 - t/t_break)`, and the grid needs ~8 points across it, which caps a
trustworthy run near `t = 0.43` s.

That estimate is the flat-bottom one and is only a guide here. This bar shallows the whole
width at the crest exactly as case 2's ridge does, so the transmitted wave steepens faster
than in case 0; the trenches work the other way over their own span, but they are shallow
and the crest dominates. Expect the resolved window to close somewhat earlier than
`t = 0.43` s, as in `outs/2_short`, which puts `T = 0.46` s at or slightly past the edge
rather than comfortably inside it. Measure the front on `hhms.npy` before treating the
last snapshots as quantitative.

## 6. Data / Assimilation Flags

Read by `SWHD_2D.__init__`; only `make_data` affects this forward run. Unchanged from
`outs/2_short`.

| Parameter | Value | Meaning |
|---|---|---|
| `make_data` | True | dump `uums`/`vvms`/`hhms` on the final step |
| `noise` | False | add Gaussian noise to the saved measurements |
| `uum_noise_std`, `vvm_noise_std`, `hhm_noise_std` | 0.0 | noise levels |
| `iit`, `iit0`, `iitN` | 0, 0, 1 | assimilation iteration counters |

## 7. Paths & Output

`out_path`, `hb_path` and `data_path` are not set in `params.py`: `time_marching.py`
sets all three to the case directory it was given, which for this run is `./outs/3_short`.

| Field | Storage attr | File |
|---|---|---|
| `u` | `self.uus` | `{out_path}/uums.npy` |
| `v` | `self.vvs` | `{out_path}/vvms.npy` |
| `h` | `self.hhs` | `{out_path}/hhms.npy` |
| topography | `self.hb` | `{hb_path}/hb.npy` |
| mass, `E_kin`, `E_pot` | — | `{out_path}/balance.dat` (truncated on step 0) |
