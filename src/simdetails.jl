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

# `nodetodirsptr` is the node -> direction-range pointer of DirectionalData; it
# was called `dirptr` before the directional-subdivision rewrite.
function hflfnodes(fardata)
    Nlf = 0
    Nhf = 0
    maxdirs = 0
    for node in 1:(length(fardata.nodetodirsptr) - 1)
        ndirs = NestedCrossApproximation.ndirections(fardata, node)
        ndirs == 0 && continue
        maxdirs = max(maxdirs, ndirs)
        if first(NestedCrossApproximation.dirs(fardata, node)) == 0
            Nlf += 1
        else
            Nhf += 1
        end
    end
    return Nlf, Nhf, maxdirs
end

function simdetails(
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
    testfardata, trialfardata = NestedCrossApproximation.fardata(tree, isnear)
    Nlf, Nhf, maxdirs = hflfnodes(testfardata)

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
        maxrank=80,
        isnear=isnear,
        scheduler=scheduler,
        matrixdata=BEAST.DoubleNumWiltonSauterQStrat(3, 4, 5, 6, 5, 6, 5, 6),
        farmatrixdata=BEAST.DoubleNumQStrat(3, 4),
    )

    @time refmat = AdaptiveCrossApproximation.HMatrix(
        op,
        tspace,
        sspace,
        tree; 
        isnear=isnear,
        maxrank=60,
        spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
        compressor=ACA(;
            convergence=AdaptiveCrossApproximation.CombinedConvCrit([
                FNormEstimator(tol * 1e-1),
                AdaptiveCrossApproximation.RandomSampling(; tol=tol * 1e-1, factor=1.0),
            ]),
        ),
        scheduler=scheduler,
        nearmatrixdata=BEAST.DoubleNumWiltonSauterQStrat(3, 4, 5, 6, 5, 6, 5, 6),
        farmatrixdata=BEAST.DoubleNumQStrat(3, 4),
    )

    farh2mat = NestedCrossApproximation.farmatrix(h2mat)
    farrefmat = AdaptiveCrossApproximation.farmatrix(refmat)
    errh2mat = estimate_reldifference(h2mat, refmat; tol=tol * 1e-1)
    farerrh2mat = estimate_reldifference(farh2mat, farrefmat; tol=tol * 1e-1)
    storh2mat = NestedCrossApproximation.storage(h2mat)
    df = DataFrame(;
        k=isnear.k,
        gamma=γ,
        etahf=ηhf,
        tol=tol,
        N=length(tspace),
        hfnodes=Nhf,
        lfnodes=Nlf,
        dirmax=maxdirs,
        storh2mat=storh2mat,
        farerrh2mat=farerrh2mat,
        errh2mat=errh2mat,
    )

    return CSV.write(filename, df; append=true, header=false)
end
