using CompScienceMeshes
using BEAST
using H2Trees
using ParallelKMeans
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CSV, DataFrames
using LinearAlgebra
using Random

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "..", "src", "simcompare.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geometry.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geo.jl"))

# Rafale with fuel tanks and a locally refined nozzle: a multiscale geometry, in
# contrast to the uniformly refined cube and sphere. Same three error-controlled
# compressions as the sphere, wavelength again at ten edge lengths.

γ = 1.0
ηhf = 2.0
tol = 1e-3
methods = ("hmatrs", "ncaefie", "ncaefieoct")

filename = joinpath(@__DIR__, "..", "..", "results", "rafalefuel.csv")
CSV.write(filename, compareframe(methods))
##

for reff in ["0.13", "0.1", "0.07", "0.05", "0.035", "0.025", "0.0175"]
    meshpath = geometrypath(
        joinpath(@__DIR__, "geometry"), "rafale10fuel_gmsh" * reff * ".msh"
    )
    Γ = CompScienceMeshes.read_gmsh_mesh(meshpath)
    space = raviartthomas(Γ)
    println("Size RT ", length(space))
    h = edgeinfo(Γ)[3]
    λ = 10h
    println("Wavelength: ", λ)
    k = 2 * pi / λ

    op = Maxwell3D.singlelayer(; wavenumber=k)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=ηhf, γ=γ)
    simcompare(
        filename, op, space, space, isnear; methods=methods, tol=tol, ηhf=ηhf, γ=γ
    )
end
