using CompScienceMeshes
using BEAST
using H2Trees
using ParallelKMeans
using NestedCrossApproximation
using CSV, DataFrames
using LinearAlgebra
using Random

BLAS.set_num_threads(4)
include(pwd() * "/src/simsweep.jl")
include(pwd() * "/src/geo.jl")
γ = 1.0
ηhf = 1.0

df = DataFrame(; N=Int[], tol=Float64[], err=Float64[], storage=Float64[])

filename = pwd() * "/results/spheresweep1.0.csv"
#CSV.write(filename, df)

Γ = meshsphere(1.0, 0.025)
space = raviartthomas(Γ)
println("Size RT ", length(space))
h = edgeinfo(Γ)[3]
λ = 10h
println("Wavelength: ", λ)
k = 2 * pi / λ
##
op = Maxwell3D.singlelayer(; wavenumber=k)
Random.seed!(2)
testtree = KMeansTree(
    space.pos, 2; minvalues=200, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
testtree = TwoNTree(space, 0.0; minvalues=400)
Random.seed!(2)
trialtree = KMeansTree(
    space.pos, 2; minvalues=200, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
trialtree = TwoNTree(space, 0.0; minvalues=400)
tree = H2Trees.BlockTree(testtree, trialtree)
isnear = NestedCrossApproximation.isnearwideband(k; ηhf=ηhf, γ=γ)
simsweep(
    filename,
    op,
    space,
    space,
    tree,
    isnear,
    [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10];
    scheduler=DynamicScheduler(),
)
