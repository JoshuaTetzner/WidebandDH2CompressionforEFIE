using LinearAlgebra
using LinearMaps
using Random

"""
    estimate_norm(mat; tol=1e-4, itmax=1000, rng=Random.MersenneTwister(1))

Power-iteration estimate of the spectral norm of `mat`.

`rng` seeds the start vector. It defaults to a fixed stream so that repeated runs
return the same estimate; the iteration only converges to within `tol`, so an
unseeded start vector would move the last significant digits around.
"""
function estimate_norm(mat; tol=1e-4, itmax=1000, rng=Random.MersenneTwister(1))
    v = rand(rng, size(mat, 2))

    v = v / norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    @info "Estimate norm"
    while (norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold)) > tol || i < itermin) &&
        i < itmax
        @info i, norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold))
        σold = σnew
        w = Vector(mat * v)
        x = Vector(adjoint(mat) * w)
        σnew = norm(x)
        v = x / norm(x)
        i += 1
    end
    return sqrt(σnew)
end

"""
    estimate_reldifference(hmat, refmat; tol=1e-4, refnorm=nothing,
        rng=Random.MersenneTwister(1))

Power-iteration estimate of `norm(hmat - refmat) / norm(refmat)`.

Pass `refnorm` (from [`estimate_norm`](@ref)) when several matrices are compared
against the same reference, so that the norm of `refmat` is not re-estimated for
each of them. `rng` seeds the start vector, see [`estimate_norm`](@ref).
"""
function estimate_reldifference(
    hmat::H, refmat; tol=1e-4, refnorm=nothing, rng=Random.MersenneTwister(1)
) where {F,H<:LinearMaps.LinearMap{F}}
    v = rand(rng, F, size(hmat, 2))

    v = v / norm(v)
    itermin = 3
    itermax = 1000
    i = 1
    σold = 1
    σnew = 1
    @info "Estimate difference"
    while (norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold)) > tol || i < itermin) &&
        i < itermax
        @info i, norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold))
        σold = σnew
        w = Vector(hmat * v) - Vector(refmat * v)
        x = Vector(adjoint(hmat) * w) - Vector(adjoint(refmat) * w)
        σnew = norm(x)
        v = x / σnew
        i += 1
    end

    if refnorm === nothing
        @info "Estimate norm of reference matrix"
        refnorm = estimate_norm(refmat; tol=tol, rng=rng)
    end

    return sqrt(σnew) / refnorm
end
