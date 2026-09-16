using CompScienceMeshes
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using H2Trees
using OhMyThreads
using CSV, DataFrames
using StaticArrays
include("poweriteration.jl")
include("trees.jl")
include("utils.jl")

"""
    sweepframe() -> DataFrame

Empty `DataFrame` with the schema written by [`simsweep`](@ref): the cluster tree
and the admissibility parameter that identify the run, followed by N, the ACA
tolerance, the far-field error and the storage in GB.

Write it once per experiment to create the CSV header; `simsweep` then appends one
row per tolerance, so several `(tree, ηhf)` combinations share one file.
"""
function sweepframe()
    return DataFrame(;
        tree=String[],
        etahf=Float64[],
        N=Int[],
        tol=Float64[],
        err=Float64[],
        storage=Float64[],
    )
end

"""
    simsweep(filename, op, tspace, sspace, tree, isnear, tols; treelabel, ηhf, scheduler)

Compress `op` on `tree` once per entry of `tols` and append one row per tolerance
to `filename`. Create the file with `CSV.write(filename, sweepframe())` first.

`treelabel` and `ηhf` only label the rows; they must describe the `tree` and the
`isnear` actually passed in.

The dense reference matrix is assembled per call and then zeroed on the near-field
blocks of this tree by `fullfarmat`, so it cannot be shared across calls that use a
different tree or a different admissibility -- the near/far splits differ.
"""
function simsweep(
    filename,
    op,
    tspace,
    sspace,
    tree,
    isnear,
    tols;
    treelabel,
    ηhf,
    scheduler=DynamicScheduler(),
)
    println("\n######## tree = $treelabel, ηhf = $ηhf ########")
    println("Assemble reference matrix: ")

    quadstrat = AdaptiveCrossApproximation.defaultfarmatrixdata(op, tspace, sspace)
    refmat = assemble(op, tspace, sspace; threading=:cellcoloring, quadstrat=quadstrat)
    for tol in tols
        println("tol: ", tol)
        testpivoting, trialpivoting, convergence = ncapivoting(
            tspace, sspace, tree; filtered=true, tol=tol
        )

        h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
            op,
            tspace,
            sspace,
            tree;
            testcompressor=NestedCrossApproximation.BottomUp(;
                factorization=IACA(MaximumValue(), testpivoting, convergence)
            ),
            trialcompressor=NestedCrossApproximation.BottomUp(;
                factorization=IACA(trialpivoting, MaximumValue(), convergence)
            ),
            maxrank=200,
            isnear=isnear,
            scheduler=scheduler,
            farmatrixdata=quadstrat,
        )

        farh2mat = NestedCrossApproximation.farmatrix(h2mat)
        farfullmat = fullfarmat(refmat, h2mat)

        err = estimate_reldifference(farh2mat, farfullmat; tol=tol * 1e-1)
        storh2mat = NestedCrossApproximation.storage(h2mat)
        df = DataFrame(;
            tree=treelabel, etahf=ηhf, N=length(tspace), tol=tol, err=err, storage=storh2mat
        )

        CSV.write(filename, df; append=true, header=false)
    end
end
