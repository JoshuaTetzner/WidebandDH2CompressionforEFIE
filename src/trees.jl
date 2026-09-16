# Construction of the cluster trees and of the NCA pivoting/convergence pair,
# shared by every driver so they all build them the same way.
#
# Both H2Trees and the ACA pivoting moved to a new interface: trees are built
# through a builder object rather than positional keyword arguments, and the
# EFIE direction filter is now a `PivotingFilter` handed to `TreeMimicryPivoting`
# instead of a separate `TreeMimicryPivoting2` that took precomputed orientation
# ids. The filter derives the edge/normal groups from the space itself, so the
# orientation preamble the old drivers carried is no longer needed.

using H2Trees
using ParallelKMeans
using AdaptiveCrossApproximation
using NestedCrossApproximation
using Random

# A fresh builder per tree: the k-means splitter consumes the rng stored in the
# builder, so reusing one builder for both trees would cluster them differently.
function _kmeansbuilder(minvalues, seed)
    return H2Trees.KMeansTreeBuilder(;
        numberofclusters=2,
        minvalues=minvalues,
        updateradii=H2Trees.unsafemaxradiusboundingsphere,
        splitterkwargs=(; rng=Random.MersenneTwister(seed)),
    )
end

"""
    kmeansblocktree(tspace, sspace; minvalues=100, seed=1)

k-means `BlockTree` over the basis-function positions. Both sides get the same
seed, so a Petrov-Galerkin run on `tspace == sspace` gets two identical
clusterings.
"""
function kmeansblocktree(tspace, sspace; minvalues=100, seed=1)
    testtree = KMeansTree(tspace.pos; builder=_kmeansbuilder(minvalues, seed))
    trialtree = KMeansTree(sspace.pos; builder=_kmeansbuilder(minvalues, seed))
    return H2Trees.BlockTree(testtree, trialtree)
end

"""
    octreeblocktree(tspace, sspace; minvalues=200)

Octree (`TwoNTree`) `BlockTree` over the same spaces.
"""
function octreeblocktree(tspace, sspace; minvalues=200)
    builder = H2Trees.TwoNTreeBuilder(; minhalfsize=0.0, minvalues=minvalues)
    return H2Trees.BlockTree(
        TwoNTree(tspace; builder=builder), TwoNTree(sspace; builder=builder)
    )
end

"""
    ncapivoting(tspace, sspace, tree; filtered=true, tol=1e-3)

`(testpivoting, trialpivoting, convergence)` for the nested cross approximation.

`filtered == false` uses the plain `TreeMimicryPivoting` together with the
`FNormExtrapolator`. `filtered == true` restricts descent and pivot selection to
one orientation group at a time via the `EFIEDirectionalFilter`, which needs the
phase-aware `PhaseExtrapolator` as its stopping criterion.

The filter is always built from the *candidate* side: the test compressor pivots
over trial functions, so it gets the trial space and the trial cluster tree, and
vice versa.
"""
function ncapivoting(tspace, sspace, tree; filtered=true, tol=1e-3)
    if filtered
        testpivoting = TreeMimicryPivoting(
            tspace.pos,
            sspace.pos,
            tree.trialcluster,
            EFIEDirectionalFilter(sspace, tree.trialcluster),
        )
        trialpivoting = TreeMimicryPivoting(
            sspace.pos,
            tspace.pos,
            tree.testcluster,
            EFIEDirectionalFilter(tspace, tree.testcluster),
        )
        return testpivoting, trialpivoting, PhaseExtrapolator(tol)
    end

    testpivoting = TreeMimicryPivoting(tspace.pos, sspace.pos, tree.trialcluster)
    trialpivoting = TreeMimicryPivoting(sspace.pos, tspace.pos, tree.testcluster)
    return testpivoting, trialpivoting, FNormExtrapolator(tol)
end
