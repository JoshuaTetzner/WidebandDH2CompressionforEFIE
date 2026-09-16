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
include(joinpath(@__DIR__, "..", "..", "src", "geo.jl"))

# Unit cube, refined so the wavelength stays at ten edge lengths. All five
# compressions per discretization, including the two that are not
# error-controlled -- which is what the cube is here to show.

γ = 1.0
ηhf = 5.0
tol = 1e-3
methods = COMPARE_METHODS

filename = joinpath(@__DIR__, "..", "..", "results", "cube.csv")
CSV.write(filename, compareframe(methods))
##

for reff in [0.02, 0.0135, 0.01, 0.00675, 0.005]
    Γ = meshcuboid(1.0, 1.0, 1.0, reff)
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
