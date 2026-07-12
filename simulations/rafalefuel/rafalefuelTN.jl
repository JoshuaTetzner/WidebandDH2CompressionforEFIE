using CompScienceMeshes
using BEAST
using H2Trees
using ParallelKMeans
using NestedCrossApproximation
using CSV, DataFrames
using LinearAlgebra
using Random

BLAS.set_num_threads(1)

include(pwd() * "/src/simh2.jl")
include(pwd() * "/src/geo.jl")

γ = 1.0
ηhf = 2.0
tol = 1e-3

df = DataFrame(;
    k=Float64[],
    gamma=Float64[],
    etahf=Float64[],
    tol=Float64[],
    N=Int[],
    th2mat=Float64[],
    tmvh2mat=Float64[],
    storh2mat=Float64[],
    farerrh2mat=Float64[],
)

filename = pwd() * "/results/rafalefuelTN.csv"
#CSV.write(filename, df)
##

for reff in ["0.13", "0.0175"]#["0.13", "0.1", "0.07", "0.05", "0.035", "0.025", "0.0175"]
    meshpath =
        "/home/jt286/Documents/Geometries/rafale_fuel/rafale10fuel_gmsh" * reff * ".msh"
    Γ = CompScienceMeshes.read_gmsh_mesh(meshpath)
    space = raviartthomas(Γ)
    println("Size RT ", length(space))
    h = edgeinfo(Γ)[3]
    λ = 10h
    println("Wavelength: ", λ)
    k = 2 * pi / λ
    gamma = im * k
    alpha = -gamma
    beta = -1 / gamma

    op = Maxwell3D.singlelayer(; wavenumber=k)
    Random.seed!(1)

    #testtree = KMeansTree(
    #    space.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    #)
    testtree = TwoNTree(space, 0.0; minvalues=200)
    Random.seed!(1)
    #trialtree = KMeansTree(
    #    space.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    #)
    trialtree = TwoNTree(space, 0.0; minvalues=200)

    tree = H2Trees.BlockTree(testtree, trialtree)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=ηhf, γ=γ)
    simh2(filename, op, space, space, tree, isnear; tol=tol, ηhf=ηhf, γ=γ)
end
