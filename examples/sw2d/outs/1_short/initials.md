# SWHD_2D Simulation — Parameter Reference

Case `outs/1_short`. Mirrors the `params.py` in this directory and `time_marching.py`.
Everything below is either read from `params.py` or built in `time_marching.py`; nothing
here is an attribute of `SWHD_2D` unless the table says so.

This is **case 1 truncated at the resolved-front limit**: the same bump, the same initial
state and the same grid, stopped at `T = 0.44` s instead of 1 s. Its 111 snapshots are
the leading 111 of `outs/1`, so the two are directly comparable; this case just holds
none of the post-breakdown ones.

## 1. Domain & Grid

| Parameter | Value | Meaning |
|---|---|---|
| `Lx, Ly` | 80 × 40 | domain size (cm) |
| `Nx, Ny` | 1024 × 512 | grid resolution |
| `dx, dy` | 0.078125 | grid spacing (cm), square cells, `np.linspace(0, L, N, endpoint=False, retstep=True)` |
| `Xs, Ys` | `grid.xx`, `grid.yy` | `np.meshgrid(xi, yi, indexing='ij')` |

The domain is periodic and purely numerical: it matches neither the 170 × 70 cm tank nor
the 30 × 30 cm measurement window, it only has to hold the pulse, the bump and the travel
distance without anything wrapping into the region of interest. The resolution is set by
the steepened front (~0.6 cm wide by the end of the run, ~8 points here), not by the bump.

## 2. Physical Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `g` | 981 | gravitational acceleration (cm/s²) |
| `h_rest` | 2 | rest water height (cm) |
| `c` (derived) | ≈ 44.29 cm/s | gravity wave speed, `sqrt(g·h_rest)` |

Not present in this implementation: `nu` (no viscosity term in `rkstep`).

## 3. Bottom Topography

| Parameter | Value | Meaning |
|---|---|---|
| `H0` | 1.0 | bump height (cm), half the rest depth |
| `R` | 1.25 | Gaussian width (cm); the 5 cm-wide tank bump is placed at 2R, where `exp(-r²/R²)` has fallen to 1.8% of `H0` |
| `xb, yb` | 40, 20 | bump center (cm), `yb = Ly/2` |
| `bathymetry(X, Y)` | — | `H0·exp(-((X-xb)² + (Y-yb)²)/R²)` |
| `hb` | — | saved to `{hb_path}/hb.npy` and passed in via `solver.update_hb(hb)` |
| `true_hb` | — | reloaded from `{data_path}/hb.npy` by `solver.update_true_hb()` |

## 4. Initial Conditions

| Parameter | Value | Meaning |
|---|---|---|
| `A` | 0.2 | pulse height amplitude (cm) |
| `s` | 3.0 | pulse half-length (cm); the surface hump is ~3s = 9 cm across |
| `n` | 2 | pulse exponent |
| `x0` | 27.5 | pulse center (cm), 12.5 cm upstream of the bump |
| `U` (derived) | ≈ 4.43 cm/s | velocity amplitude, `sqrt(g/h_rest)·A` |
| `pulse` | — | `exp(-((Xs-x0)/s)^n)` — a function of x only, uniform in y |
| `h0` | — | `h_rest + A·pulse` — free surface, **offset by the rest height** |
| `u0` | — | `U·pulse` |
| `v0` | — | zeros |

`h` is the free surface, so it sits on top of `h_rest`: without that offset `h - hb` is
negative over the bump and the run turns into NaNs.

The pulse matches the wave in the tank (dominant wavelength 23 cm), but it is an
intermediate-depth wave: `k·h = 0.53`, and the shallow-water phase speed is 5.5% high
against the exact linear dispersion relation. That is the dominant modelling limitation
and should be quoted alongside any comparison with experiment.

Fields are handed to the solver as `fields = [u0, v0, h0]`, matching
`fields[0]=fu, fields[1]=fv, fields[2]=fh` inside `rkstep`.

## 5. Time Integration

| Parameter | Value | Meaning |
|---|---|---|
| `dt` | 2e-5 | timestep (s) |
| `T` | 0.44 | total simulated time (s) |
| `ostep` | 200 | steps between saved snapshots |
| `bstep` | 50 | steps between balance writes |
| `rkord` | 2 | Runge-Kutta order, `SWHD_2D` constructor default |
| `Nt` (derived) | 22 000 | `int(T/dt)`, exactly — unlike case 1, `0.44/2e-5` lands on 22 000.0 in binary, so nothing truncates |
| `total_steps` (derived) | 22 001 | `int(T/dt) + 1` |
| snapshots saved (derived) | 111 | `(total_steps - 1)//ostep + 1`, length of `uus`/`vvs`/`hhs` |
| snapshot memory (derived) | ≈ 1.4 GB | 111 × 1024 × 512 × 8 B × 3 fields |
| balance rows (derived) | 442 | steps 0, 50, …, 22 000 (441 rows) plus the `final=True` write, also at step 22 000 |
| CFL number (derived) | ≈ 1.13 × 10⁻² | `c·dt/dx` |

Snapshot `k` is the field at step `k·ostep`, i.e. `t = 0.004·k`, for `k = 0…110`, and
`k = 110` is `t = 0.44` s exactly. Slot 110 is written twice with the same field — once
in the loop at step 22 000 and again by the `final=True` write at that same step — so
unlike case 1 there is no odd last time label; the duplicate is a no-op. The same is true
of the last row of `balance.dat`, which repeats `t = 0.44`.

Since `T` here is a multiple of `dt`, the `dt = T - Nt·dt` short final step is zero and
`evolve` breaks out rather than taking it. The `make_data` dump fires because the loop
reaches `step == total_steps - 1`.

`T = 0.44` s stops at the resolved-front limit rather than past it. The crest rides deeper
water than the base and shears the wave forward until its leading face goes vertical;
Whitham (*Linear and Nonlinear Waves*, 1974, §2.1) puts that gradient catastrophe at

```
t_break = 2·h_rest / (3·c·|dη/dx|_max) = 0.53 s
```

using the exact Gaussian maximum slope `A·sqrt(2/e)/s = 0.0572 cm⁻¹`. The face narrows
roughly as `(1 - t/t_break)`, and the grid needs ~8 points across it, which caps a
trustworthy run near `t = 0.43` s. Measured on case 1's `hhms.npy`, the front spans 45
points at `t = 0`, 18 at `t = 0.28` s and 7 at `t = 0.42` s — close agreement with the
estimate, and the reason `T` is set here rather than at some round number.

`T = 0.44` s sits right at that edge, so the last handful of snapshots are the marginal
ones: check them rather than assume them. Everything before ~0.42 s is comfortably
resolved. Note `dt` does nothing for this — the face narrows in `x`, so only `Nx` helps,
and only up to `t_break`, beyond which the solution is genuinely discontinuous.

Use `outs/1` to look at how the scheme breaks down past this point; use this case for
anything measured. Scaling case 1's crest travel (27.5 → ≈ 74 cm over 1 s) the crest is
near `x ≈ 48` cm at `t = 0.44` s — just past the bump, and far from the periodic edge at
`Lx = 80`.

## 6. Data / Assimilation Flags

Read by `SWHD_2D.__init__`; only `make_data` affects this forward run.

| Parameter | Value | Meaning |
|---|---|---|
| `make_data` | True | dump `uums`/`vvms`/`hhms` on the final step |
| `noise` | False | add Gaussian noise to the saved measurements |
| `uum_noise_std`, `vvm_noise_std`, `hhm_noise_std` | 0.0 | noise levels |
| `iit`, `iit0`, `iitN` | 0, 0, 1 | assimilation iteration counters |

## 7. Paths & Output

`out_path`, `hb_path` and `data_path` are not set in `params.py`: `time_marching.py`
sets all three to the case directory it was given, which for this run is `./outs/1_short`.

| Field | Storage attr | File |
|---|---|---|
| `u` | `self.uus` | `{out_path}/uums.npy` |
| `v` | `self.vvs` | `{out_path}/vvms.npy` |
| `h` | `self.hhs` | `{out_path}/hhms.npy` |
| topography | `self.hb` | `{hb_path}/hb.npy` |
| mass, `E_kin`, `E_pot` | — | `{out_path}/balance.dat` (truncated on step 0) |
