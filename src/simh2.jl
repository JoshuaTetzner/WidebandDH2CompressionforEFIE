using CompScienceMeshes
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using H2Trees
using OhMyThreads
using CSV, DataFrames
using StaticArrays
include("poweriteration.jl")
include("orientations.jl")

function simh2(
    filename,
    op,
    tspace,
    sspace,
    tree,
    isnear;
    tol=1e-3,
    ηhf=1.0,
    γ=1.0,
    scheduler=DynamicScheduler(),
)
    tedges, tnormals = rwg_orientations(tspace)
    sedges, snormals = rwg_orientations(sspace)
    tedgeids, tnormalids, tnedgeids, tnnormalids = NestedCrossApproximation.basisfunction_orientation_ids(
        tedges, tnormals
    )
    sedgeids, snormalids, snedgeids, snnormalids = NestedCrossApproximation.basisfunction_orientation_ids(
        sedges, snormals
    )
    trial_node_normal_sets, trialnormalids, ntrialnormalids = NestedCrossApproximation.node_normal_orientation_sets(
        tnormals, tree.trialcluster
    )
    test_node_normal_sets, testnormalids, ntestnormalids = NestedCrossApproximation.node_normal_orientation_sets(
        snormals, tree.testcluster
    )

    th2mat = @elapsed h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
        op,
        tspace,
        sspace,
        tree;
        testcompressor=NestedCrossApproximation.BottomUp(;
            factorization=iACA(
                MaximumValue(),
                AdaptiveCrossApproximation.TreeMimicryPivoting2(
                    tspace.pos,
                    sspace.pos,
                    sedgeids,
                    trialnormalids,
                    trial_node_normal_sets,
                    tree.trialcluster,
                ),
                OversampIFNormEst(tol),
            ),
        ),
        trialcompressor=NestedCrossApproximation.BottomUp(;
            factorization=iACA(
                AdaptiveCrossApproximation.TreeMimicryPivoting2(
                    sspace.pos,
                    tspace.pos,
                    tedgeids,
                    testnormalids,
                    test_node_normal_sets,
                    tree.testcluster,
                ),
                MaximumValue(),
                OversampIFNormEst(tol),
            ),
        ),
        maxrank=50,
        isnear=isnear,
        scheduler=scheduler,
    )

    @time refmat = AdaptiveCrossApproximation.HMatrix(
        op,
        tspace,
        sspace,
        tree;
        isnear=isnear,
        maxrank=100,
        spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
        compressor=ACA(;
            convergence=AdaptiveCrossApproximation.CombinedConvCrit([
                FNormEstimator(tol * 1e-2),
                AdaptiveCrossApproximation.RandomSampling(; tol=tol * 1e-2, factor=1.0),
            ]),
        ),
        scheduler=scheduler,
    )

    farh2mat = NestedCrossApproximation.farmatrix(h2mat)
    refmat = AdaptiveCrossApproximation.farmatrix(refmat)

    x = rand(eltype(h2mat), length(sspace))
    y = h2mat * x
    tmvh2mat = Float64[]
    for _ in 1:15
        tt = @elapsed y = h2mat * x
        push!(tmvh2mat, tt)
    end
    tmvh2mat = minimum(tmvh2mat)

    farerrh2mat = estimate_reldifference(farh2mat, refmat; tol=tol * 1e-1)
    storh2mat = NestedCrossApproximation.storage(h2mat)

    df = DataFrame(;
        k=isnear.k,
        gamma=γ,
        etahf=ηhf,
        tol=tol,
        N=length(tspace),
        th2mat=th2mat,
        tmvh2mat=tmvh2mat,
        storh2mat=storh2mat,
        farerrh2mat=farerrh2mat,
    )

    return CSV.write(filename, df; append=true, header=false)
end
