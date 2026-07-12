using CompScienceMeshes
using BEAST
using H2Trees
using ParallelKMeans
using NestedCrossApproximation
using CSV, DataFrames
using LinearAlgebra
using Random

BLAS.set_num_threads(1)

include(pwd() * "/src/simtime.jl")
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
    asstime=Float64[],
    tmvmat=Float64[],
)
method = "thiswork"
filename = pwd() * "/results/spheretime" * method * ".csv"
#CSV.write(filename, df)
##

for reff in [80, 80, 114, 114, 114, 160, 160, 160]#[28, 28, 40, 40, 57, 57, 80, 80, 114, 114, 160, 160]
    Γ = meshicosphere(reff, 1.0)
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
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=ηhf, γ=γ)
    #isnear = AdaptiveCrossApproximation.isnear(),
    simtime(
        filename,
        op,
        space,
        space,
        isnear;
        tol=tol,
        method=method,
        ηhf=ηhf,
        γ=γ,
        scheduler=SerialScheduler(),
    )
end
