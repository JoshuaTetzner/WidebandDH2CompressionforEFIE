using CompScienceMeshes
using BEAST
using H2Trees
using ParallelKMeans
using NestedCrossApproximation
using CSV, DataFrames
using LinearAlgebra
using Random

BLAS.set_num_threads(1)

include(pwd() * "/src/simfull.jl")
include(pwd() * "/src/geo.jl")

γ = 1.0
ηhf = 5.0
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
    errh2mat=Float64[],
)

filename = pwd() * "/results/typhooniaca.csv"
CSV.write(filename, df)
##

for h in [0.25, 0.1735, 0.122, 0.0875, 0.06, 0.0425, 0.03, 0.021, 0.015]
    run(`gmsh simulations/typhoon/typhoon.geo -2 -clmax $h  -format
    msh2 -o simulations/typhoon/typhoon.msh`)
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd() * "/simulations/typhoon/typhoon.msh")
    space = raviartthomas(Γ)
    println("Size RT ", length(space))
    h = edgeinfo(Γ)[3]
    λ = 10h
    println("Wavelength: ", λ)
    k = 2 * pi / λ

    op = Maxwell3D.singlelayer(; wavenumber=k)
    Random.seed!(1)

    testtree = KMeansTree(
        space.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    )
    #testtree = TwoNTree(space, 2 / 2^10; minvalues=200)
    Random.seed!(1)
    trialtree = KMeansTree(
        space.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    )
    #trialtree = TwoNTree(space, 2 / 2^10; minvalues=200)

    tree = H2Trees.BlockTree(testtree, trialtree)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=ηhf, γ=γ)
    simfull(filename, op, space, space, tree, isnear; tol=tol, ηhf=ηhf, γ=γ)
end
