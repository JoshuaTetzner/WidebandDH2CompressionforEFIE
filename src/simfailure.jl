using CompScienceMeshes
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using H2Trees
using OhMyThreads
using CSV, DataFrames
using StaticArrays
include("poweriteration.jl")

function simfailure(
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
    th2mat = @elapsed h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
        op,
        tspace,
        sspace,
        tree;
        testcompressor=NestedCrossApproximation.BottomUp(;
            factorization=iACA(
                MaximumValue(),
                AdaptiveCrossApproximation.TreeMimicryPivoting(
                    tspace.pos, sspace.pos, tree.trialcluster
                ),
                FNormExtrapolator(iFNormEstimator(tol)),
            ),
        ),
        trialcompressor=NestedCrossApproximation.BottomUp(;
            factorization=iACA(
                AdaptiveCrossApproximation.TreeMimicryPivoting(
                    sspace.pos, tspace.pos, tree.testcluster
                ),
                MaximumValue(),
                FNormExtrapolator(iFNormEstimator(tol)),
            ),
        ),
        maxrank=50,
        isnear=isnear,
        scheduler=scheduler,
    )

    @time thmat = @elapsed hmat = AdaptiveCrossApproximation.HMatrix(
        op,
        tspace,
        sspace,
        tree;
        isnear=isnear,
        maxrank=50,
        spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
        compressor=ACA(; tol=tol),
        scheduler=StaticScheduler(),
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
    farhmat = AdaptiveCrossApproximation.farmatrix(hmat)
    refmat = AdaptiveCrossApproximation.farmatrix(refmat)

    x = rand(eltype(hmat), length(sspace))
    println(norm(h2mat * x - hmat * x) / norm(hmat * x))
    println(norm(farh2mat * x - farhmat * x) / norm(farhmat * x))
    y = h2mat * x
    tmvh2mat = Float64[]
    tmvhmat = Float64[]
    for _ in 1:15
        tt = @elapsed y = h2mat * x
        th = @elapsed y = hmat * x
        push!(tmvhmat, th)
        push!(tmvh2mat, tt)
    end
    tmvh2mat = minimum(tmvh2mat)
    tmvhmat = minimum(tmvhmat)

    farerrh2mat = estimate_reldifference(farh2mat, refmat; tol=tol * 1e-1)
    farerrhmat = estimate_reldifference(farhmat, refmat; tol=tol * 1e-1)
    storh2mat = NestedCrossApproximation.storage(h2mat)
    storhmat = AdaptiveCrossApproximation.storage(hmat)

    df = DataFrame(;
        k=isnear.k,
        gamma=γ,
        etahf=ηhf,
        tol=tol,
        N=length(tspace),
        th2mat=th2mat,
        thmat=thmat,
        tmvh2mat=tmvh2mat,
        tmvhmat=tmvhmat,
        storh2mat=storh2mat,
        storhmat=storhmat,
        farerrh2mat=farerrh2mat,
        farerrhmat=farerrhmat,
    )

    return CSV.write(filename, df; append=true, header=false)
end
