#!/usr/bin/env bash
# Runs every simulation of the paper sequentially with 16 threads.
#
# The reported assembly and matrix-vector times are wall-clock times at this
# thread count; keep it fixed across a series of runs so they stay comparable.
set -e

JULIA="julia --project=. --threads=16"

# --- Cube ---------------------------------------------------------------------
$JULIA simulations/cube/cube.jl

# --- Sphere -------------------------------------------------------------------
$JULIA simulations/sphere/sphere.jl
$JULIA simulations/sphere/spheresweep.jl
$JULIA simulations/sphere/spheremie.jl

# --- Refined rafale (fuel tanks, locally refined nozzle) -----------------------
$JULIA simulations/rafalefuel/rafalefuel.jl

# --- Solved rafale -----------------------------------------------------------
$JULIA simulations/rafaleopen/rafaleopenfull.jl
$JULIA simulations/rafaleopen/rafaledetails.jl
