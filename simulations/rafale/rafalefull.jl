using OhMyThreads
using BEAST, CompScienceMeshes
using ParallelKMeans
using H2Trees, AdaptiveCrossApproximation, NestedCrossApproximation, Random
using JLD2
using StaticArrays
using DelimitedFiles
using LinearAlgebra

BLAS.set_num_threads(1)

const c = 2.99792458e8          # speed of lightx
const μ = 4π * 1e-7             # permeability
const ε = 8.8541878176e-12      # permittivity

include(joinpath(@__DIR__, "../..", "src", "loopsstars.jl"))
include(joinpath(@__DIR__, "../..", "src", "bcmap.jl"))
include(joinpath(@__DIR__, "../..", "src", "graminverse.jl"))
include(joinpath(@__DIR__, "../..", "src", "utils.jl"))

## -----------------------------------------------------------------------------
filename = "rafale_gm_248448"
meshpath = "/home/jt286/Documents/Geometries/rafale/$(filename).msh"
Γ = CompScienceMeshes.read_gmsh_mesh(meshpath)

# Works for both open and closed surfaces. A closed surface has no free boundary
# (Euler characteristic 2); an open one does. Just sanity-check the mesh loaded
# and report which kind it is.
@assert numcells(Γ) > 0
let nb = numcells(boundary(Γ))
    println(nb == 0 ? "Closed surface" : "Open surface ($nb boundary edges)")
end
## -----------------------------------------------------------------------------

_, _, h = edgeinfo(Γ)
λ = 40h     # Wavelength
f = c / λ
k = 2 * π / λ   # Wavenumber
ω = c * k     # Angular frequency
#include(joinpath(@__DIR__, "../..", "src", "manufacturedsolution.jl"))

## -----------------------------------------------------------------------------

𝓣A = Maxwell3D.singlelayer(; wavenumber=k, alpha=(-im * k), beta=zero(λ) * im)
𝓣Φ = Maxwell3D.singlelayer(; wavenumber=k, alpha=zero(λ) * im, beta=-1 / (im * k))
println("assembled operators")
##

# Use interior edges only. On an open surface no current flows off the free rim,
# so dropping boundary edges keeps the primal RT space dual to the
# Buffa-Christiansen space (which lives on interior edges), so the mixed Gram
# matrix Nyx stays square and invertible. On a closed surface there are no
# boundary edges, so this is a no-op (all edges are interior).
edges = submesh(!CompScienceMeshes.in(boundary(Γ)), skeleton(Γ, 1))
X = raviartthomas(Γ, edges)

σ = lagrangecxd0(Γ)
Σ = getstars(Γ) * Diagonal([1 / volume(chart(Γ, i)) for i in 1:numcells(Γ)])
@time M, Y, Xr = rttobc(Γ; edges=edges);
mr = geometry(Xr)
σr = lagrangecxd0(mr)

# Dual star (divergence) operator for the refined RT space Xr. Built directly
# from BEAST's divergence(Xr) so its sign and normalization are exactly
# consistent with Xr's function ordering, including the half-RWGs on the
# refinement's free boundary. A connectivity/getstars-based star is
# sign-inconsistent on the all-edge (half-RWG) case; that corrupts the dual
# scalar potential (~48% error in the hypersingular term) and badly degrades the
# Calderón preconditioner. Σr[k,c] = div(Xr_k) on cell c, expanded in the σr basis.
@time Σr = rt_divergence_matrix(Xr)

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

# Dual basis (complex-wavenumber Calderón preconditioner)
## -----------------------------------------------------------------------------
# The preconditioner operator Tyy is assembled at a complex (lossy) wavenumber
# k̃ = k - iαk instead of the physical k. The static principal symbol of T is
# wavenumber-independent, so T(k̃)·T(k) is still a Calderón regularizer (≈ -¼I +
# compact), but the imaginary part damps the cavity quasi-resonances of the open
# MIG so the preconditioner has no near-null space → far fewer GMRES iterations.
# Txx (the physical operator) stays at real k, so the computed solution is exact.
#
# Sign of Im(k̃): with BEAST's exp(-im*k̃*R) kernel, k̃ = k - iαk makes the kernel
# decay so the NCA far-field stays low-rank. If assembly blows up, flip to +.
# Magnitude α: tune in ~0.02–0.1 (too small: no resonance damping; too large:
# poor Calderón quality off-resonance).
α = 0.05
k̃ = k - im * α * k

𝓣Ã = Maxwell3D.singlelayer(; wavenumber=k̃, alpha=(-im * k̃), beta=zero(k̃))
𝓣Φ̃ = Maxwell3D.singlelayer(; wavenumber=k̃, alpha=zero(k̃), beta=-1 / (im * k̃))

println("Assembling dual basis...")
TAxrxr = assembleh2X(𝓣Ã, Xr; k=abs(k̃))#; threading=BEAST.Threading{:cellcoloring})

println("\tConstructed vector potential")
TΦσrσr = assembleh2(
    Helmholtz3D.singlelayer(; wavenumber=k̃, alpha=𝓣Φ̃.β), σr, σr; k=abs(k̃)
)#; threading=BEAST.Threading{:cellcoloring})
Tyy = M * (TAxrxr + Σr * TΦσrσr * Σr') * M'
println("\tConstructed scalar potential")
e = assemble(
    (n × Maxwell3D.planewave(; direction=(-ẑ), polarization=ŷ, wavenumber=k)) × n, X
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
jldsave(joinpath(@__DIR__, "../..", "results", "rafale_$(numfunctions(X))_40h.jld2"); u)
##

fcr, geo = facecurrents(u, X)

# Visualize solution
## -----------------------------------------------------------------------------

using WriteVTK
cellV = typeof(MeshCell(VTKCellTypes.VTK_TRIANGLE, [2, 4, 3]))[]
for (ind, face) in enumerate(Γ.faces)
    push!(cellV, MeshCell(VTKCellTypes.VTK_TRIANGLE, [face[1], face[2], face[3]]))
end

vtk_grid(
    joinpath(@__DIR__, "../..", "results", "rafale_$(numfunctions(X))_40h"),
    vertexarray(Γ)',
    cellV,
) do vtk
    vtk["amplitude"] = norm.(fcr)
    vtk["real"] = norm.(real.(fcr))
    return vtk["imag"] = norm.(imag.(fcr))
end

#=
##
using JLD2
using DelimitedFiles
using StaticArrays
using SphericalScattering

u = load("/home/jt286/Documents/WidebandNCA/sphere$h.jld2")["u"]

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

path = joinpath(@__DIR__, "../..", "results", "rcs_xy_mie.csv")
open(path, "w") do io
    println(io, "theta_deg,sigma_dBsm,sigma_dBsm_anal,relative_error")
    return writedlm(io, data, ',')
end
=#
