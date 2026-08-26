# SWHD_2D Simulation — Parameter Reference

Case `outs/0`. Mirrors the `params.py` in this directory and `time_marching.py`.
Everything below is either read from `params.py` or built in `time_marching.py`; nothing
here is an attribute of `SWHD_2D` unless the table says so.

This is the **flat-bottom control for case 1**: identical grid, identical time
integration and a byte-identical `initial_fields`, with a `bathymetry` that returns zero.
Anything that differs between `outs/0` and `outs/1` is the bump.

## 1. Domain & Grid

| Parameter | Value | Meaning |
|---|---|---|
| `Lx, Ly` | 80 × 40 | domain size (cm) |
| `Nx, Ny` | 1024 × 512 | grid resolution |
| `dx, dy` | 0.078125 | grid spacing (cm), square cells, `np.linspace(0, L, N, endpoint=False, retstep=True)` |
| `Xs, Ys` | `grid.xx`, `grid.yy` | `np.meshgrid(xi, yi, indexing='ij')` |

The domain is periodic and purely numerical: it matches neither the 170 × 70 cm tank nor
the 30 × 30 cm measurement window, it only has to hold the pulse and the travel distance
without anything wrapping into the region of interest. The resolution is set by the
steepened front (~0.6 cm wide by `t = 0.42` s, ~8 points here).

With a flat bottom the problem is uniform in `y`: `v ≡ 0` and every `x`-line of `u` and
`h` is the same, so the run is effectively 1D. `Ny = 512` is kept only so the two cases
share a grid.

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
| `bathymetry(X, Y)` | `zeros_like(X)` | flat bottom |
| `hb` | — | zero everywhere, still saved to `{hb_path}/hb.npy` and passed in via `solver.update_hb(hb)` |
| `true_hb` | — | reloaded from `{data_path}/hb.npy` by `solver.update_true_hb()` |

There are no topography parameters in this case: `H0`, `R`, `xb` and `yb` are dropped
rather than set to zero, since nothing reads them.

## 4. Initial Conditions

Identical to case 1.

| Parameter | Value | Meaning |
|---|---|---|
| `A` | 0.2 | pulse height amplitude (cm) |
| `s` | 3.0 | pulse half-length (cm); the surface hump is ~3s = 9 cm across |
| `n` | 2 | pulse exponent |
| `x0` | 27.5 | pulse center (cm), 12.5 cm upstream of where case 1 puts the bump |
| `U` (derived) | ≈ 4.43 cm/s | velocity amplitude, `sqrt(g/h_rest)·A` |
| `pulse` | — | `exp(-((Xs-x0)/s)^n)` — a function of x only, uniform in y |
| `h0` | — | `h_rest + A·pulse` — free surface, **offset by the rest height** |
| `u0` | — | `U·pulse` |
| `v0` | — | zeros |

`h` is the free surface, so it sits on top of `h_rest`. The offset matters less here than
in case 1 — with `hb = 0` the column `h - hb` is never driven negative by topography —
but it is kept so the two runs start from exactly the same state.

The pulse matches the wave in the tank (dominant wavelength 23 cm), but it is an
intermediate-depth wave: `k·h = 0.53`, and the shallow-water phase speed is 5.5% high
against the exact linear dispersion relation. That is the dominant modelling limitation
and should be quoted alongside any comparison with experiment.

Fields are handed to the solver as `fields = [u0, v0, h0]`, matching
`fields[0]=fu, fields[1]=fv, fields[2]=fh` inside `rkstep`.

## 5. Time Integration

Identical to case 1, so the two runs can be compared snapshot for snapshot.

| Parameter | Value | Meaning |
|---|---|---|
| `dt` | 2e-5 | timestep (s) |
| `T` | 1 | total simulated time (s) |
| `ostep` | 200 | steps between saved snapshots |
| `bstep` | 50 | steps between balance writes |
| `rkord` | 2 | Runge-Kutta order, `SWHD_2D` constructor default |
| `Nt` (derived) | 49 999 | `int(T/dt)`, **not** 50 000 — `2e-5` is above its decimal value in binary, so `1/2e-5` falls just short and truncates down |
| `total_steps` (derived) | 50 000 | `int(T/dt) + 1` |
| snapshots saved (derived) | 250 | `(total_steps - 1)//ostep + 1`, length of `uus`/`vvs`/`hhs` |
| snapshot memory (derived) | ≈ 3.1 GB | 250 × 1024 × 512 × 8 B × 3 fields |
| balance rows (derived) | 1001 | steps 0, 50, …, 49 950 plus the final write at step 49 999 |
| CFL number (derived) | ≈ 1.13 × 10⁻² | `c·dt/dx` |

Snapshot `k` is the field at step `k·ostep`, i.e. `t = 0.004·k`, for `k = 0…248`. The last
slot is written twice — once at step 49 800 and again by the `final=True` write at step
49 999 — so slot 249 holds `t = 0.99998`, not `t = 0.996`.

`T = 1` s deliberately overruns the resolved-front limit. The crest rides deeper water
than the base and shears the wave forward until its leading face goes vertical; Whitham
(*Linear and Nonlinear Waves*, 1974, §2.1) puts that gradient catastrophe at

```
t_break = 2·h_rest / (3·c·|dη/dx|_max) = 0.53 s
```

using the exact Gaussian maximum slope `A·sqrt(2/e)/s = 0.0572 cm⁻¹`. The face narrows
roughly as `(1 - t/t_break)`, and the grid needs ~8 points across it, which caps a
trustworthy run near `t = 0.43` s. Ringing behind the crest after that is expected, not a
bug, and only the early snapshots are usable quantitatively — the same caveat as case 1,
kept deliberately so the control breaks down the same way the bumped run does.

With no bump nothing slows or reflects the pulse, so its crest travels marginally further
by `t = 1` s than case 1's, still short of the periodic edge at `Lx = 80`.

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
sets all three to the case directory it was given, which for this run is `./outs/0`.

| Field | Storage attr | File |
|---|---|---|
| `u` | `self.uus` | `{out_path}/uums.npy` |
| `v` | `self.vvs` | `{out_path}/vvms.npy` |
| `h` | `self.hhs` | `{out_path}/hhms.npy` |
| topography | `self.hb` | `{hb_path}/hb.npy` |
| mass, `E_kin`, `E_pot` | — | `{out_path}/balance.dat` (truncated on step 0) |
