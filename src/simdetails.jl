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

function hflfnodes(fardata)
    Nlf = 0
    Nhf = 0
    maxdirs = 0
    for node in 1:(length(fardata.dirptr) - 1)
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

    testfardata, trialfardata = NestedCrossApproximation.fardata(tree, isnear)
    Nlf, Nhf, maxdirs = hflfnodes(testfardata)

    h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
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
