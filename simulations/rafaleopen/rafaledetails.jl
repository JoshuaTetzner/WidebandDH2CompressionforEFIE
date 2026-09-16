# Directional-tree statistics for the open-nose rafale: sweeps the wavelength over
# a fixed discretization and reports how the wideband admissibility splits the
# cluster tree into low- and high-frequency nodes, the largest direction count on
# any node, and the storage and error of the compression.

using CompScienceMeshes
using BEAST
using H2Trees
using ParallelKMeans
using NestedCrossApproximation
using CSV, DataFrames
using LinearAlgebra
using Random

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "..", "src", "simdetails.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geo.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geometry.jl"))

γ = 1.0
ηhf = 1.0
tol = 1e-3
meshname = "rafale_opennose_0.025"

df = DataFrame(;
    k=Float64[],
    gamma=Float64[],
    etahf=Float64[],
    tol=Float64[],
    N=Int[],
    hfnodes=Int[],
    lfnodes=Int[],
    dirmax=Int[],
    storh2mat=Float64[],
    farerrh2mat=Float64[],
    errh2mat=Float64[],
)

filename = joinpath(@__DIR__, "..", "..", "results", "rafaledetails.csv")
CSV.write(filename, df)
##

# Mesh, space and tree are the same for every wavelength -- only the wavenumber
# and with it the admissibility change -- so build them once.
meshpath = geometrypath(joinpath(@__DIR__, "geometry"), "$(meshname).msh")
Γ = CompScienceMeshes.read_gmsh_mesh(meshpath)
space = raviartthomas(Γ)
println("Size RT ", length(space))
h = edgeinfo(Γ)[3]

tree = kmeansblocktree(space, space; minvalues=100, seed=1)
##

for mult in [10, 40, 160]
    λ = mult * h
    println("Wavelength: ", λ)
    k = 2 * pi / λ

    op = Maxwell3D.singlelayer(; wavenumber=k)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=ηhf, γ=γ)

    # Reseed per row so no row depends on the RNG state the previous one left
    # behind; the reference matrix's RandomSampling criterion draws from it.
    Random.seed!(1)
    simdetails(filename, op, space, space, tree, isnear; tol=tol, ηhf=ηhf, γ=γ)
end
