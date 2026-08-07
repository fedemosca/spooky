# SWHD_2D Simulation — Parameter Reference

## 1. Domain & Grid

| Parameter | Value | Meaning |
|---|---|---|
| `Nx, Ny` | 512 × 512 | grid resolution |
| `Lx, Ly` | 30 × 30 | domain size (cm) |
| `dx, dy` | ≈ 0.0586 | grid spacing, `np.linspace(0, L, N, endpoint=False, retstep=True)` (note: `ys`/`dy` is built with `Nx` samples in the init-condition script, not `Ny` — no numeric effect here since `Nx == Ny`) |
| `Xs, Ys` | — | `np.meshgrid(xs, ys, indexing='ij')` |

## 2. Physical Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `pm.g` | 981 | gravitational acceleration (cm/s²) |
| `h_rest` | 3 | rest water height; used only in the init-condition script (for `U`), not stored on `SWHD_2D` |
| `c` (derived) | ≈ 54.25 cm/s | gravity wave speed, `sqrt(pm.g · h_rest)` |

Not present in this implementation: `nu` (no viscosity term in `rkstep`).

## 3. Bottom Topography

| Parameter | Value | Meaning |
|---|---|---|
| `H0` | 2 | bump height (init-condition script) |
| `R` | 5 | bump width (init-condition script) |
| center | (15, 15) | `(Lx/2, Ly/2)` |
| `hb` | — | init-condition script's `H`, passed in via `self.update_hb(hb)`; stored as `self.hb` |
| `true_hb` | — | loaded from `{pm.hb_path}/hb.npy` via `self.update_true_hb()` |

## 4. Initial Conditions

| Parameter | Value | Meaning |
|---|---|---|
| `A` | 0.3 | pulse height amplitude |
| `s` | 1 | pulse half-length |
| `x0` | 6 | pulse center, `Lx/5` |
| `n` | 2 | pulse exponent |
| `U` (derived) | ≈ 5.42 cm/s | velocity amplitude, `sqrt(pm.g/h_rest)·A` |
| `h0` | — | `A·exp(-((Xs-x0)/s)^n)` — pulse only, no `h_rest` offset |
| `u0` | — | `U·exp(-((Xs-x0)/s)^n)` |
| `v0` | — | zeros |

None of these are attributes of `SWHD_2D` — they're computed in the init-condition
script, then handed to the solver as `fields = [u0, v0, h0]`, matching
`fields[0]=fu, fields[1]=fv, fields[2]=fh` inside `rkstep`.

## 5. Time Integration

| Parameter | Value | Meaning |
|---|---|---|
| `pm.dt` | 1e-6 | timestep (s) |
| `pm.T` | 0.06 | total simulated time (s), `= steps · dt` from the init-condition script |
| `pm.ostep` | 1,000 | steps between saved snapshots, `= skip` |
| `total_steps` (derived) | `int(pm.T/pm.dt) + 1` | total timesteps |
| `rkord` | 2 | Runge-Kutta order, constructor arg |
| CFL number (derived) | ≈ 9.3 × 10⁻⁴ | `c·pm.dt/dx` |
| snapshots saved (derived) | `int(total_steps/pm.ostep)` | array length of `uus`/`vvs`/`hhs` |

## 6. Output

| Field | Storage attr | File |
|---|---|---|
| `h` | `self.hhs` | `{pm.out_path}/hhms.npy` |
| `u` | `self.uus` | `{pm.out_path}/uums.npy` |
| `v` | `self.vvs` | `{pm.out_path}/vvms.npy` |