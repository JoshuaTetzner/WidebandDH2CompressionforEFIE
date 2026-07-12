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
include("utils.jl")

function simsweep(
    filename, op, tspace, sspace, tree, isnear, tols; scheduler=DynamicScheduler()
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
    println("Assemble reference matrix: ")

    quadstrat=AdaptiveCrossApproximation.defaultfarmatrixdata(op, tspace, sspace)
    refmat = assemble(op, tspace, sspace; threading=:cellcoloring, quadstrat=quadstrat)
    for tol in tols
        println("tol: ", tol)
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
            maxrank=200,
            isnear=isnear,
            scheduler=scheduler,
            farmatrixdata=quadstrat,
        )

        farh2mat = NestedCrossApproximation.farmatrix(h2mat)
        farfullmat = fullfarmat(refmat, h2mat)

        err = estimate_reldifference(farh2mat, farfullmat; tol=tol * 1e-1)
        storh2mat = NestedCrossApproximation.storage(h2mat)
        df = DataFrame(; N=length(tspace), tol=tol, err=err, storage=storh2mat)

        CSV.write(filename, df; append=true, header=false)
    end
end
