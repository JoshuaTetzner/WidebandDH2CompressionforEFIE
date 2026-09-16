# Assembly helpers shared by the two EFIE solve scripts, sphere/spheremie.jl and
# rafaleopen/rafaleopenfull.jl.
#
# The orientation extraction these used to carry (edge directions, face normals
# and the direction-group ids per basis function) now lives in the packages: the
# `EFIEDirectionalFilter` derives all of it from the space, so the preamble these
# helpers used to run is gone. Tree and pivoting construction live in trees.jl,
# mesh metrics in geo.jl.

using BEAST, CompScienceMeshes
using LinearAlgebra
using StaticArrays

include("trees.jl")

"""
    fullfarmat(A, h2mat) -> A

Zero the near-field blocks of the dense matrix `A` in place, so that what remains
is the far field under `h2mat`'s near/far split. Used to compare a compressed far
field against a dense reference on the same tree.
"""
function fullfarmat(A, h2mat)
    nears = h2mat.nearinteractions
    for i in eachindex(nears.blocks)
        A[nears.rowindices[i], nears.colindices[i]] .= 0.0
    end
    return A
end

"""
    assembleh2X(op, X; k=0.0, minvalues=100, tol=1e-3, maxrank=80, seed=1)

Directional H² compression of `op` on a single Raviart-Thomas space `X`, with the
EFIE directional filter. Used for the vector-potential blocks of the solves.
"""
function assembleh2X(op, X; k=0.0, minvalues=100, tol=1e-3, maxrank=80, seed=1)
    tree = kmeansblocktree(X, X; minvalues=minvalues, seed=seed)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=1.0, γ=1.0)

    testpivoting, trialpivoting, convergence = ncapivoting(
        X, X, tree; filtered=true, tol=tol
    )

    return NestedCrossApproximation.PetrovGalerkinNCA(
        op,
        X,
        X,
        tree;
        maxrank=maxrank,
        isnear=isnear,
        testcompressor=NestedCrossApproximation.BottomUp(;
            factorization=IACA(MaximumValue(), testpivoting, convergence)
        ),
        trialcompressor=NestedCrossApproximation.BottomUp(;
            factorization=IACA(trialpivoting, MaximumValue(), convergence)
        ),
        scheduler=DynamicScheduler(),
        matrixdata=BEAST.DoubleNumWiltonSauterQStrat(3, 4, 5, 6, 5, 6, 5, 6),
        farmatrixdata=BEAST.DoubleNumQStrat(3, 4),
    )
end

"""
    assembleh2(op, X, Y; k=0.0, minvalues=100, tol=1e-3, maxrank=80, seed=1)

Directional H² compression of `op` between two different spaces, with plain tree
mimicry pivoting. Used for the scalar-potential blocks, where the trial and test
spaces are the scalar (Lagrange) spaces rather than RT, so the EFIE directional
filter does not apply.
"""
function assembleh2(op, X, Y; k=0.0, minvalues=100, tol=1e-3, maxrank=80, seed=1)
    tree = kmeansblocktree(X, Y; minvalues=minvalues, seed=seed)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=1.0, γ=1.0)

    testpivoting, trialpivoting, convergence = ncapivoting(
        X, Y, tree; filtered=false, tol=tol
    )

    return NestedCrossApproximation.PetrovGalerkinNCA(
        op,
        X,
        Y,
        tree;
        isnear=isnear,
        maxrank=maxrank,
        testcompressor=NestedCrossApproximation.BottomUp(;
            factorization=IACA(MaximumValue(), testpivoting, convergence)
        ),
        trialcompressor=NestedCrossApproximation.BottomUp(;
            factorization=IACA(trialpivoting, MaximumValue(), convergence)
        ),
        scheduler=DynamicScheduler(),
        matrixdata=BEAST.DoubleNumWiltonSauterQStrat(3, 4, 5, 6, 5, 6, 5, 6),
        farmatrixdata=BEAST.DoubleNumQStrat(3, 4),
    )
end
