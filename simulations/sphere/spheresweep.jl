# Accuracy/storage sweep of the DH², NCA-EFIE compression over the ACA tolerance,
# on a fixed discretization, for both cluster trees and both admissibility
# parameters. All of it lands in results/spheresweep.csv, one row per
# (tree, ηhf, tol).

using CompScienceMeshes
using BEAST
using H2Trees
using ParallelKMeans
using NestedCrossApproximation
using CSV, DataFrames
using LinearAlgebra
using Random

BLAS.set_num_threads(4)
include(joinpath(@__DIR__, "..", "..", "src", "simsweep.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geo.jl"))

γ = 1.0
tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

filename = joinpath(@__DIR__, "..", "..", "results", "spheresweep.csv")
CSV.write(filename, sweepframe())

# Note this sweep meshes with `meshsphere`, while sphere/sphere.jl uses
# `meshicosphere`; the two are not meant to be compared row by row.
Γ = meshsphere(1.0, 0.025)
space = raviartthomas(Γ)
println("Size RT ", length(space))
h = edgeinfo(Γ)[3]
λ = 10h
println("Wavelength: ", λ)
k = 2 * pi / λ

op = Maxwell3D.singlelayer(; wavenumber=k)
##

for treelabel in ("kmeans", "octree"), ηhf in (1.0, 5.0)
    # Both sides get the same seed, so the Petrov-Galerkin run on one space gets
    # two identical clusterings. The octree is deterministic and ignores the seed.
    tree = if treelabel == "octree"
        octreeblocktree(space, space; minvalues=400)
    else
        kmeansblocktree(space, space; minvalues=200, seed=2)
    end

    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=ηhf, γ=γ)
    simsweep(
        filename,
        op,
        space,
        space,
        tree,
        isnear,
        tols;
        treelabel=treelabel,
        ηhf=ηhf,
        scheduler=DynamicScheduler(),
    )
    GC.gc()
end
