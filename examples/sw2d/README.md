# sw2d — 2D shallow water over topography

Cases so far: `0` flat bottom, `1` an isolated Gaussian bump, `2` that same bump profile
extended into a ridge across the full width. They share every other parameter, so they
are comparable snapshot for snapshot. `1_short` is case 1 stopped at `T = 0.44` s, the
resolved-front limit, for when only the trustworthy part of the run is wanted.

## Workflow

One directory per run, under `outs/`. The name is free-form — an integer for a new
setup, or a suffixed variant like `1_short` for a run that differs from an existing case
in one parameter. A case directory holds its
own `params.py` and every file the run produced, so the parameters that generated a
result always sit next to the result.

```
examples/sw2d/
├── time_marching.py       # the run script, takes a case directory
├── video.py               # animates one stacked field, takes a .npy inside a case dir
├── submit.sh              # submits one case to SLURM
├── job.sh                 # the batch script submit.sh hands to sbatch
├── README.md              # this file
└── outs/
    ├── 1/
    │   ├── params.py      # the only source of parameters for this case
    │   ├── initials.md    # parameter reference for this case
    │   ├── hb.npy         # topography, written by the run
    │   ├── uums.npy       # stacked (nt, Nx, Ny) fields, written by the run
    │   ├── vvms.npy
    │   ├── hhms.npy
    │   ├── balance.dat    # t, mass, E_kin, E_pot
    │   ├── hhms.mp4       # written by video.py
    │   └── sw2d-<jid>.out # SLURM log, one per submitted job
    ├── 2/
    └── ...
```

There is no top-level `params.py`. Every `params.py` in the tree describes exactly one
run.

## Running a case

```bash
python3 time_marching.py ./outs/1
```

The directory must already exist and contain a `params.py`; `time_marching.py` errors out
if either is missing rather than creating anything. Everything the run writes goes into
that directory.

Then animate a field:

```bash
python3 video.py ./outs/1/hhms.npy
```

`video.py` reads `params.py` and `hb.npy` from the directory holding the array, so its
time labels come from the run that produced the data, never from some other case.

On the cluster, submit the case:

```bash
./submit.sh ./outs/1
```

`submit.sh` checks the directory, then calls `sbatch` with `CASE` exported and the log
redirected to `<case_dir>/sw2d-%j.out`, so the job's stdout ends up with the run it
produced. The `%j` is the job id: re-submitting a case adds a log rather than
overwriting the previous one.

This cannot be done from `job.sh` alone. `#SBATCH` directives are read by `sbatch` at
submit time and do not expand shell variables, so `#SBATCH -o "$CASE/sw2d.out"` would
write to a directory literally named `$CASE`. The `-o` line still in `job.sh` is only a
fallback for a bare `sbatch job.sh`, which fails immediately on the missing `CASE`.

## Adding a case

```bash
mkdir outs/2
cp outs/1/params.py outs/2/params.py
```

Then edit `outs/2/params.py` and run it. Copying from the nearest existing case is the
intended way to start — there is no template to keep in sync.

## Writing a params.py

The file is a flat module of module-level names, imported by `time_marching.py` from the
case directory. It must define:

| Group | Names |
|---|---|
| Domain & grid | `Lx`, `Ly`, `Nx`, `Ny` |
| Physical | `g`, `h_rest` |
| Time integration | `dt`, `T`, `ostep`, `bstep` |
| Data flags | `make_data`, `noise`, `uum_noise_std`, `vvm_noise_std`, `hhm_noise_std`, `iit`, `iit0`, `iitN` |
| Fields | `bathymetry(X, Y)`, `initial_fields(X, Y)` |

**Do not set `out_path`, `hb_path` or `data_path`.** `time_marching.py` sets all three on
the loaded module to the case directory it was given, so a `params.py` copied between
cases can never redirect output into the wrong run. Anything written for those names in
the file is discarded.

The two functions are where a case says what it actually simulates. `time_marching.py`
calls them with the grid meshes `Xs, Ys` and uses whatever comes back, so it never knows
what shape a case has and adding a case never touches it:

- `bathymetry(X, Y)` returns `hb`, saved to `hb.npy`.
- `initial_fields(X, Y)` returns `[u0, v0, h0]`, in the order `rkstep` expects.

Any scalar the functions use — bump height, pulse width, whatever the case needs — is a
module-level name in the same file, so the numbers stay readable at the top and the
shapes stay explicit at the bottom. Nothing is required beyond the two functions, so
case 0 simply has no topography parameters at all.

The three current cases are

```
outs/0   hb = 0
outs/1   hb = H0*exp(-((X-xb)^2 + (Y-yb)^2)/R^2)      isolated bump
outs/2   hb = H0*exp(-(X-xb)^2/R^2)                   that profile, extended in y
```

all three with the same initial state

```
pulse = exp(-((X-x0)/s)^n)
h0    = h_rest + A*pulse        # free surface, offset by the rest height
u0    = U*pulse
v0    = 0
```

`h` is the free surface, not the water column: without the `h_rest` offset, `h - hb` goes
negative over the topography and the run turns into NaNs.

Two derived quantities are worth checking by hand before submitting:

- **Snapshot count and size.** `Nt = int(T/dt)`, and the run saves
  `(Nt//ostep) + 1` snapshots of `Nx × Ny × 8` bytes for each of the three fields. Note
  `int(T/dt)` truncates a float division, so it can land one below the round number you
  expect — at `T = 1`, `dt = 2e-5` it gives 49 999, not 50 000.
- **How long the front stays resolved.** The pulse steepens as it travels until its
  leading face goes vertical, at Whitham's gradient catastrophe
  `t_break = 2*h_rest/(3*c*slope_max)` with `c = sqrt(g*h_rest)` and, for the Gaussian
  pulse, `slope_max = A*sqrt(2/e)/s`. The face narrows roughly as `(1 - t/t_break)` from
  an initial width `A/slope_max`, and a spectral method needs ~8 points across it, so a
  trustworthy run stops near `0.82*t_break`. Only `Nx` buys more — `dt` does nothing,
  since the face narrows in `x`. Running past that is fine if the breakdown is the point
  (case 1 does exactly this); it is not fine if the late snapshots are meant to be
  quantitative.

## Version control

Outputs are gitignored (`*.npy`, `*.png`, `balance.dat`) since they are regenerable from
the case's `params.py`. The `params.py` and `initials.md` in each case directory are
tracked — they are the record of what was run. SLURM logs (`sw2d-*.out`) are not
currently ignored; add them to `.gitignore` if you would rather not carry them.
