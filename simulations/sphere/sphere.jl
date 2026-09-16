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

# Icosphere refined so the wavelength stays at ten edge lengths. Only the three
# error-controlled compressions; the other two are shown on the cube.

γ = 1.0
ηhf = 5.0
tol = 1e-3
methods = ("hmatrs", "ncaefie", "ncaefieoct")

filename = joinpath(@__DIR__, "..", "..", "results", "sphere.csv")
CSV.write(filename, compareframe(methods))
##

for reff in [28, 40, 57, 80, 114, 160]
    Γ = meshicosphere(reff, 1.0)
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
