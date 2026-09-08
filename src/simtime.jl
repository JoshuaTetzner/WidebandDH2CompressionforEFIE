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

function simtime(
    filename,
    op,
    tspace,
    sspace,
    isnear;
    tol=1e-3,
    ηhf=1.0,
    γ=1.0,
    method="thiswork",
    scheduler=SerialScheduler(),
)
    if method == "thiswork"
        testtree = KMeansTree(
            tspace.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
        )
        Random.seed!(1)
        trialtree = KMeansTree(
            sspace.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
        )
        tree=H2Trees.BlockTree(testtree, trialtree)
        println("Running this work's method")
        tedges, tnormals = rwg_orientations(tspace)
        sedges, snormals = rwg_orientations(sspace)
        tedgeids, tnormalids, tnedgeids, tnnormalids = NestedCrossApproximation.basisfunction_orientation_ids(
            tedges, tnormals
        )
        sedgeids, snormalids, snedgeids, snnormalids = NestedCrossApproximation.basisfunction_orientation_ids(
            sedges, snormals
        )
        trial_node_normal_sets, trialnormalids, ntrialnormalids = NestedCrossApproximation.node_normal_orientation_sets(
            snormals, tree.trialcluster
        )
        test_node_normal_sets, testnormalids, ntestnormalids = NestedCrossApproximation.node_normal_orientation_sets(
            tnormals, tree.testcluster
        )

        @time asstime = @elapsed mat = NestedCrossApproximation.PetrovGalerkinNCA(
            op,
            tspace,
            sspace,
            tree;
            testcompressor=NestedCrossApproximation.BottomUp(;
                factorization=IACA(
                    MaximumValue(),
                    AdaptiveCrossApproximation.TreeMimicryPivoting2(
                        tspace.pos,
                        sspace.pos,
                        sedgeids,
                        trialnormalids,
                        trial_node_normal_sets,
                        tree.trialcluster,
                    ),
                    AdaptiveCrossApproximation.PhaseExtrapolator(tol),
                ),
            ),
            trialcompressor=NestedCrossApproximation.BottomUp(;
                factorization=IACA(
                    AdaptiveCrossApproximation.TreeMimicryPivoting2(
                        sspace.pos,
                        tspace.pos,
                        tedgeids,
                        testnormalids,
                        test_node_normal_sets,
                        tree.testcluster,
                    ),
                    MaximumValue(),
                    AdaptiveCrossApproximation.PhaseExtrapolator(tol),
                ),
            ),
            maxrank=50,
            isnear=isnear,
            scheduler=scheduler,
        )
    elseif method=="octreethiswork"
        testtree = TwoNTree(tspace, 0.0; minvalues=200)
        trialtree = TwoNTree(sspace, 0.0; minvalues=200)
        tree=H2Trees.BlockTree(testtree, trialtree)
        println("Running octree")
        tedges, tnormals = rwg_orientations(tspace)
        sedges, snormals = rwg_orientations(sspace)
        tedgeids, tnormalids, tnedgeids, tnnormalids = NestedCrossApproximation.basisfunction_orientation_ids(
            tedges, tnormals
        )
        sedgeids, snormalids, snedgeids, snnormalids = NestedCrossApproximation.basisfunction_orientation_ids(
            sedges, snormals
        )
        trial_node_normal_sets, trialnormalids, ntrialnormalids = NestedCrossApproximation.node_normal_orientation_sets(
            snormals, tree.trialcluster
        )
        test_node_normal_sets, testnormalids, ntestnormalids = NestedCrossApproximation.node_normal_orientation_sets(
            tnormals, tree.testcluster
        )

        @time asstime = @elapsed mat = NestedCrossApproximation.PetrovGalerkinNCA(
            op,
            tspace,
            sspace,
            tree;
            testcompressor=NestedCrossApproximation.BottomUp(;
                factorization=IACA(
                    MaximumValue(),
                    AdaptiveCrossApproximation.TreeMimicryPivoting2(
                        tspace.pos,
                        sspace.pos,
                        sedgeids,
                        trialnormalids,
                        trial_node_normal_sets,
                        tree.trialcluster,
                    ),
                    AdaptiveCrossApproximation.PhaseExtrapolator(tol),
                ),
            ),
            trialcompressor=NestedCrossApproximation.BottomUp(;
                factorization=IACA(
                    AdaptiveCrossApproximation.TreeMimicryPivoting2(
                        sspace.pos,
                        tspace.pos,
                        tedgeids,
                        testnormalids,
                        test_node_normal_sets,
                        tree.testcluster,
                    ),
                    MaximumValue(),
                    AdaptiveCrossApproximation.PhaseExtrapolator(tol),
                ),
            ),
            maxrank=50,
            isnear=isnear,
            scheduler=scheduler,
        )
    elseif method=="hmat"
        println("Running HMatrix with KMeans trees")
        testtree = KMeansTree(
            tspace.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
        )
        Random.seed!(1)
        trialtree = KMeansTree(
            sspace.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
        )
        tree=H2Trees.BlockTree(testtree, trialtree)
        @time asstime = @elapsed mat = AdaptiveCrossApproximation.HMatrix(
            op,
            tspace,
            sspace,
            tree;
            isnear=isnear,
            maxrank=50,
            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
            compressor=ACA(;
                convergence=AdaptiveCrossApproximation.CombinedConvCrit([
                    FNormEstimator(1e-2),
                    AdaptiveCrossApproximation.RandomSampling(; tol=8e-3, factor=1.0),
                ]),
            ),
            scheduler=scheduler,
        )
    end
    x = rand(eltype(mat), length(sspace))
    y = mat * x
    tmvmat = Float64[]
    for _ in 1:15
        tt = @elapsed y = mat * x
        push!(tmvmat, tt)
    end
    tmvmat = minimum(tmvmat)

    df = DataFrame(;
        k=isnear.k,
        gamma=γ,
        etahf=ηhf,
        tol=tol,
        N=length(tspace),
        asstime=asstime,
        tmvmat=tmvmat,
    )

    return CSV.write(filename, df; append=true, header=false)
end
