# Mie validation: EFIE on a PEC sphere, solved with a Calderón-preconditioned
# GMRES over the wideband DH² compression, radar cross section compared against
# the analytic Mie series. Writes results/rcs_xy_mie.csv and a .vtu of the surface
# current density.

using OhMyThreads
using BEAST, CompScienceMeshes
using ParallelKMeans
using H2Trees, AdaptiveCrossApproximation, NestedCrossApproximation, Random
using StaticArrays
using DelimitedFiles
using LinearAlgebra
using SphericalScattering
using WriteVTK
using Krylov

BLAS.set_num_threads(1)

const c = 2.99792458e8          # speed of light
const μ = 4π * 1e-7             # permeability
const ε = 8.8541878176e-12      # permittivity

include(joinpath(@__DIR__, "..", "..", "src", "loopsstars.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bcmap.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "graminverse.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "geo.jl"))

## -----------------------------------------------------------------------------
Γ = meshsphere(1.0, 0.017)

@assert length(skeleton(Γ, 0)) - length(skeleton(Γ, 1)) + length(skeleton(Γ, 2)) == 2
## -----------------------------------------------------------------------------

_, _, h = edgeinfo(Γ)
λ = 10h     # Wavelength
f = c / λ
k = 2 * π / λ   # Wavenumber
ω = c * k     # Angular frequency

## -----------------------------------------------------------------------------

𝓣A = Maxwell3D.singlelayer(; wavenumber=k, alpha=-im * k, beta=zero(λ) * im)
𝓣Φ = Maxwell3D.singlelayer(; wavenumber=k, alpha=zero(λ) * im, beta=-1 / (im * k))
##

edges = skeleton(Γ, 1)
X = raviartthomas(Γ, edges)

σ = lagrangecxd0(Γ)
Σ = getstars(Γ) * Diagonal([1 / volume(chart(Γ, i)) for i in 1:numcells(Γ)])
M, Y, Xr = rttobc(Γ; edges=edges)
mr = geometry(Xr)
σr = lagrangecxd0(mr)
Σr = getstars(mr) * Diagonal([1 / volume(chart(mr, i)) for i in 1:numcells(mr)])

println("\n
Functions of RT space: $(numfunctions(X))
Functions of refined RT space: $(numfunctions(Xr))
Functions of star space σ: $(numfunctions(σ))")

# Construct Gram matrix and its inverse
## -----------------------------------------------------------------------------

Nyx = assemble(BEAST.NCross(), Y, X) # = -Nxyᵀ
iNxy = GramInverse(permutedims(-Nyx); rtol=typeof(λ)(1e-8))
iTNxy = GramInverse(-Nyx; rtol=typeof(λ)(1e-8))
x = randn(eltype(Nyx), size(Nyx, 2))
@assert maximum(abs, -Nyx * (iTNxy * x) - x) < 1e-5

# Primal basis
## -----------------------------------------------------------------------------
println("Assembling primal basis...")
TAxx = assembleh2X(𝓣A, X; k=k)

x = randn(typeof(λ), numfunctions(X))
@assert !any(isnan, TAxx * x)

println("\tConstructed vector potential")
TΦσσ = assembleh2(Helmholtz3D.singlelayer(; wavenumber=k, alpha=𝓣Φ.β), σ, σ; k=k)#; threading=BEAST.Threading{:cellcoloring})

Txx = TAxx + Σ * TΦσσ * Σ'
println("\tConstructed scalar potential")

# Dual basis
## -----------------------------------------------------------------------------

println("Assembling dual basis...")
TAxrxr = assembleh2X(𝓣A, Xr; k=k)#; threading=BEAST.Threading{:cellcoloring})

println("\tConstructed vector potential")
TΦσrσr = assembleh2(Helmholtz3D.singlelayer(; wavenumber=k, alpha=𝓣Φ.β), σr, σr; k=k)#; threading=BEAST.Threading{:cellcoloring})
Tyy = M * (TAxrxr + Σr * TΦσrσr * Σr') * M'

println("\tConstructed scalar potential")
e = assemble(
    (n × Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=k)) × n, X
)

# Solve system
## -----------------------------------------------------------------------------

u, stats = Krylov.gmres(
    iTNxy * Tyy * iNxy * Txx,
    -iTNxy * Tyy * (iNxy * e);
    memory=50,
    history=true,
    rtol=typeof(λ)(1e-4),
    verbose=1,
    itmax=1000,
)
##
fcr, geo = facecurrents(u, X)

# Visualize solution
## -----------------------------------------------------------------------------

cellV = typeof(MeshCell(VTKCellTypes.VTK_TRIANGLE, [2, 4, 3]))[]
for (ind, face) in enumerate(Γ.faces)
    push!(cellV, MeshCell(VTKCellTypes.VTK_TRIANGLE, [face[1], face[2], face[3]]))
end

vtk_grid(
    joinpath(@__DIR__, "..", "..", "results", "solve_sphere_$(numfunctions(X)).vtu"),
    vertexarray(Γ)',
    cellV,
) do vtk
    vtk["amplitude"] = norm.(fcr)
    vtk["real"] = norm.(real.(fcr))
    return vtk["imag"] = norm.(imag.(fcr))
end

# Radar cross section against the analytic Mie series
## -----------------------------------------------------------------------------

E0 = 1.0

ntheta = 721
thetas = 2π .* (0:(ntheta - 1)) ./ (ntheta - 1)
rhat = [SVector{3,Float64}(sin(θ - π), 0.0, cos(θ - π)) for θ in thetas]

##
ex_anal = planeWave(; frequency=f)
Γ_anal = PECSphere(; radius=1.0)
Efar_anal = scatteredfield(Γ_anal, ex_anal, FarField(rhat))

ff = Maxwell3D.farfield(; wavenumber=k)
Efar = -im * f / (2 * c) * potential(ff, rhat, u, X; threading=BEAST.Threading{:multi})
##

sigma = 4π .* [sum(abs2, E) for E in Efar] ./ abs2(E0)
sigma_anal = 4π .* [sum(abs2, E) for E in Efar_anal] ./ abs2(E0)
sigma_dBsm = 10 .* log10.(max.(sigma, eps(Float64)))
sigma_dBsm_anal = 10 .* log10.(max.(sigma_anal, eps(Float64)))
relative_error = abs.(sigma .- sigma_anal) ./ max.(abs.(sigma_anal), eps(Float64))

theta_deg = rad2deg.(thetas)
data = hcat(theta_deg, sigma_dBsm, sigma_dBsm_anal, relative_error)

path = joinpath(@__DIR__, "..", "..", "results", "rcs_xy_mie.csv")
open(path, "w") do io
    println(io, "theta_deg,sigma_dBsm,sigma_dBsm_anal,relative_error")
    return writedlm(io, data, ',')
end
