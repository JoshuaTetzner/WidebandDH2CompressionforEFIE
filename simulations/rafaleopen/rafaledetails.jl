using CompScienceMeshes
using BEAST
using H2Trees
using ParallelKMeans
using NestedCrossApproximation
using CSV, DataFrames
using LinearAlgebra
using Random

BLAS.set_num_threads(1)

include(pwd() * "/src/simdetails.jl")
include(pwd() * "/src/geo.jl")

γ = 1.0
ηhf = 1.0
tol = 1e-3

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

filename = pwd() * "/results/rafaledetails.csv"
CSV.write(filename, df)
##

for mult in [10, 40, 160, 640]
    ffilename = "rafale_opennose_0.025"
    meshpath = "/home/jt286/Documents/Geometries/rafaleopen/$(ffilename).msh"
    Γ = CompScienceMeshes.read_gmsh_mesh(meshpath)
    space = raviartthomas(Γ)
    println("Size RT ", length(space))
    h = edgeinfo(Γ)[3]
    λ = mult*h
    println("Wavelength: ", λ)
    k = 2 * pi / λ
    gamma = im * k
    alpha = -gamma
    beta = -1 / gamma

    op = Maxwell3D.singlelayer(; wavenumber=k)
    Random.seed!(1)

    testtree = KMeansTree(
        space.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    )
    Random.seed!(1)
    trialtree = KMeansTree(
        space.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    )

    tree = H2Trees.BlockTree(testtree, trialtree)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=ηhf, γ=γ)
    simdetails(filename, op, space, space, tree, isnear; tol=tol, ηhf=ηhf, γ=γ)
end
