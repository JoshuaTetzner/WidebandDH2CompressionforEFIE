# WidebandDH2CompressionforEFIE

This repository contains all simulation scripts required to reproduce the results
presented in the paper:

**"<!-- TODO: paper title -->"**
*Joshua Tetzner, Simon Adrian*

## 1. Overview

The repository provides the simulation scripts used to generate the figures and tables
of the paper. All implementations are written in **Julia** and the workflow is fully
script-driven. The compression algorithms themselves live in the separate package
[NestedCrossApproximation.jl](https://github.com/JoshuaTetzner/NestedCrossApproximation.jl);
this repository only drives them.

The experiments fall into four groups:

| Group | Directory | What it studies |
| --- | --- | --- |
| Cube | `simulations/cube` | All five compressions on the unit cube, including the two that are not error-controlled |
| Sphere | `simulations/sphere` | The three error-controlled compressions on the icosphere, plus the tolerance sweep and the Mie validation |
| Refined rafale | `simulations/rafalefuel` | The same three on a multiscale rafale with fuel tanks and a locally refined nozzle |
| Solved rafale | `simulations/rafaleopen` | Calderón-preconditioned EFIE solve of the open-nose rafale |

At the top level, `runall.sh` runs every simulation sequentially with 16 threads and
shows how each script is launched individually.

## 2. Setup

### 2.1 Install Julia

The code was developed and tested with **Julia ≥ 1.10**. Julia can be downloaded from
[julialang.org/downloads](https://julialang.org/downloads).

### 2.2 Clone the repository

```
git clone https://github.com/JoshuaTetzner/WidebandDH2CompressionforEFIE.git
```

### 2.3 Dependencies

Start a Julia REPL in the repository root and run

```julia
using Pkg
Pkg.instantiate()
```

This installs all Julia dependencies pinned in `Project.toml` and `Manifest.toml`.
The two compression packages are fetched straight from GitHub at the exact commit
recorded in the manifest, so this step needs network access. To verify the
installation:

```julia
using NestedCrossApproximation
```

This should load the package without throwing an error.

## 3. External requirements

The cube and sphere geometries are meshed on the fly by
[CompScienceMeshes.jl](https://github.com/krcools/CompScienceMeshes.jl), which
interfaces with [Gmsh](https://gmsh.info/). Please install Gmsh and make sure the
`gmsh` executable is on your path:

```
gmsh --version
```

On Linux, Gmsh is usually available through the package manager:

```
sudo apt install gmsh
```

On Windows, download the precompiled binary from the
[Gmsh website](https://gmsh.info/#Download).

The rafale meshes are **not** generated at run time. They ship with the repository
under `simulations/<case>/geometry/`, gzip-compressed as `*.msh.gz` (63 MB in total
instead of 283 MB). `geometrypath` in `src/geometry.jl` expands an archive the first
time a script asks for it and reuses the expanded mesh afterwards, so no manual step
and no configuration is needed.

The expanded `*.msh` files are build artefacts: they are excluded by `.gitignore` and
can be deleted at any time to reclaim disk space. Expanding all of them needs about
283 MB.

## 4. Hardware requirements

The largest runs (rafale with 1.85 M unknowns, sphere with 768 k unknowns) need a
machine with several hundred GB of RAM at peak.

Run the simulations multithreaded; the reported assembly and matrix-vector times are
wall-clock times at that thread count and are not meant as a scaling study. Keep the
thread count fixed within a series of runs so the timings stay comparable.

## 5. Repository layout

```
simulations/           one directory per experiment group; geometry/ holds its meshes (*.msh.gz)
src/                   shared assembly, timing and post-processing drivers
results/               the data behind the figures (see results/README.md)
runall.sh              runs every simulation sequentially
```

`src/` contains the drivers shared by the scripts:

| File | Role |
| --- | --- |
| `simcompare.jl` | Builds the selected compressions on one discretization and reports storage, far-field error, assembly time and mat-vec time for each |
| `simsweep.jl` | Error and storage over a range of ACA tolerances, for one tree and admissibility per call |
| `simdetails.jl` | High-/low-frequency node counts and maximum direction count per tree |
| `trees.jl` | Builds the k-means and octree `BlockTree`s and the NCA pivoting/convergence pair |
| `utils.jl`, `geo.jl`, `poweriteration.jl` | Assembly helpers, mesh metrics, norm and difference estimation |
| `bcmap.jl`, `loopsstars.jl`, `graminverse.jl` | Buffa-Christiansen map, loop/star operators and Gram inverse for the Calderón-preconditioned solves |
| `geometry.jl` | Resolves a mesh path, transparently expanding the shipped `*.msh.gz` |

## 6. Running the simulations

### 6.1 Run everything

```bash
bash runall.sh
```

Each script recreates its own output file in `results/`, so a full run regenerates
every data file committed here.

### 6.2 Run individual simulations

<!-- TODO: fill in the figure/table number each script belongs to. -->

**Cube**

```julia
julia --project=. --threads=16 simulations/cube/cube.jl   # results/cube.csv
```

`cube.jl` runs all five compressions of the same operator on each discretization
and writes one row per discretization:

| Column suffix | Label | Method |
| --- | --- | --- |
| `hmataca` | H, ACA | H-matrix, standard ACA (Frobenius-norm stopping criterion) |
| `hmatrs` | H, ACA-RS | H-matrix, ACA with the additional random-sampling criterion |
| `nca` | DH², NCA | directional H², wideband admissibility, plain tree mimicry pivoting |
| `ncaefie` | DH², NCA-EFIE | the same, with the EFIE directional filter |
| `ncaefieoct` | DH², NCA-EFIE (octree) | `ncaefie` on an octree instead of the k-means tree |

`hmatrs`, `ncaefie` and `ncaefieoct` are the three **error-controlled** compressions:
their stopping criterion tracks the true far-field error, so they attain the requested
tolerance. `hmataca` and `nca` do not -- their error stagnates about an order of
magnitude above `tol` and grows with N.

Each method contributes a storage (`stor…`, GB), a far-field error (`err…`), an
assembly time (`tass…`, s) and a matrix-vector time (`tmv…`, s). The first four
share one k-means `BlockTree`; the octree variant gets its own tree and therefore
its own reference matrix, since an error is only meaningful against a reference
with the same near/far split.

The `methods` keyword of `simcompare` selects any subset of these, and
`compareframe(methods)` writes the matching header. A group is skipped entirely
when nothing in it is requested, so asking only for `"ncaefieoct"` builds no
k-means tree and no k-means reference matrix.

*Reproducibility.* The `seed` keyword of `simcompare` (default `1`) fixes every
random draw: the k-means clustering, the sample positions of the `RandomSampling`
convergence criterion, and the start vectors of the power iterations behind the
error estimates. Two runs at the same thread count give bit-identical `stor…` and
`err…` columns; the `tass…`/`tmv…` columns are wall-clock times and vary.
Across different thread counts the `hmatrs` columns and the reference matrix
shift slightly, because the assembly spawns one task per chunk and Julia derives
each task's RNG from the root RNG at spawn time, so the number of chunks changes
which entries get sampled. Use the same thread count for a series of runs that has
to be comparable -- any fixed count will do.

**Sphere**

```julia
julia --project=. --threads=16 simulations/sphere/sphere.jl        # results/sphere.csv
julia --project=. --threads=16 simulations/sphere/spheresweep.jl   # results/spheresweep.csv
julia --project=. --threads=16 simulations/sphere/spheremie.jl     # results/rcs_xy_mie.csv, solve_sphere_<N>.vtu
```

`sphere.jl` uses the same driver as the cube, restricted to the three error-controlled
approaches -- `hmatrs`, `ncaefie` and `ncaefieoct`. The two that are not
error-controlled are demonstrated on the cube and are not repeated here.

`spheresweep.jl` sweeps the ACA tolerance on a fixed discretization, over the 2×2
grid of cluster tree (k-means, octree) and `ηhf ∈ {1.0, 5.0}`. All four runs share
one file, `spheresweep.csv`; the `tree` and `etahf` columns say which row belongs
to which. Each run assembles its own dense reference matrix, because the near/far
split -- and with it the far field the error is measured on -- changes with the
tree and with `ηhf`.

`spheremie.jl` solves the EFIE on the sphere with the same Calderón-preconditioned
GMRES as the rafale and compares the radar cross section against the analytic Mie
series. It writes `rcs_xy_mie.csv` (the cut the paper plots) and a `.vtu` of the
surface current density. Neither script writes the solution vector: it is only an
intermediate on the way to those files.

**Refined rafale**

```julia
julia --project=. --threads=16 simulations/rafalefuel/rafalefuel.jl   # results/rafalefuel.csv
```

`rafalefuel.jl` runs the same three approaches as the sphere on the seven fuel-tank
meshes (`ηhf = 2.0`), so the multiscale case is directly comparable to the
uniformly refined cube and sphere.

**Solved rafale**

```julia
julia --project=. --threads=16 simulations/rafaleopen/rafaleopenfull.jl # results/openrafale_<N>_640h.vtu
julia --project=. --threads=16 simulations/rafaleopen/rafaledetails.jl  # results/rafaledetails.csv
```

The solve script assembles the EFIE with the wideband DH² compression, solves it
with a Calderón-preconditioned GMRES and writes the surface current density as a
`.vtu` for ParaView. It computes no radar cross section -- there is no analytic
reference for this geometry. `.vtu` is excluded from version control by
`.gitignore`, so the file is not committed.

## 7. Reproducibility statement

The compression algorithms live in two separate packages, both openly available and
also archived on Zenodo:

| Package | Branch |
| --- | --- |
| [NestedCrossApproximation.jl](https://github.com/JoshuaTetzner/NestedCrossApproximation.jl) | `paper/WidebandDH2CompressionforEFIE` |
| [AdaptiveCrossApproximation.jl](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl) | `paper/WidebandDH2CompressionforEFIE` |

Both are pinned in `Manifest.toml` by `git-tree-sha1`, not merely by branch name, so
`Pkg.instantiate()` reproduces the exact source used for these results even if the
branch or the packages' main branches move on afterwards.
