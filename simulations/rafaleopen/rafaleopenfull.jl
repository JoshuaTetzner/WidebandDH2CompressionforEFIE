# Calderón-preconditioned EFIE solve of the open-nose rafale, assembled with the
# wideband DH² compression. Writes the surface current density as
# results/openrafale_<N>_<mult>h.vtu. The preconditioner is assembled at a complex
# wavenumber to damp the cavity quasi-resonances; see the dual-basis section.

using OhMyThreads
using BEAST, CompScienceMeshes
using ParallelKMeans
using H2Trees, AdaptiveCrossApproximation, NestedCrossApproximation, Random
using StaticArrays
using DelimitedFiles
using LinearAlgebra
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
include(joinpath(@__DIR__, "..", "..", "src", "geometry.jl"))

## -----------------------------------------------------------------------------
meshname = "rafale_opennose_0.025"
meshpath = geometrypath(joinpath(@__DIR__, "geometry"), "$(meshname).msh")
Γ = CompScienceMeshes.read_gmsh_mesh(meshpath)

@assert numcells(Γ) > 0
let nb = numcells(boundary(Γ))
    println(nb == 0 ? "Closed surface" : "Open surface ($nb boundary edges)")
end
## -----------------------------------------------------------------------------
_, _, h = edgeinfo(Γ)
const mult = 640     # Wavelength in edge lengths; also names the output file
λ = mult * h # Wavelength
f = c / λ
k = 2 * π / λ   # Wavenumber
ω = c * k     # Angular frequency

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
    rtol=typeof(λ)(1e-6),
    verbose=1,
    itmax=1500,
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
    joinpath(
        @__DIR__, "..", "..", "results", "openrafale_$(numfunctions(X))_$(mult)h"
    ),
    vertexarray(Γ)',
    cellV,
) do vtk
    vtk["amplitude"] = norm.(fcr)
    vtk["real"] = norm.(real.(fcr))
    return vtk["imag"] = norm.(imag.(fcr))
end
