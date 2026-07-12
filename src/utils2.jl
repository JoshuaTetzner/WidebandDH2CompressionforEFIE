using JLD2
using BEAST, CompScienceMeshes
using LinearAlgebra
using StaticArrays

# NOTE: This is the adapted copy of `utils.jl` for the *new* ACA/NCA interface
# (TreeMimicryPivoting2 + DirectionFilter). All the per-basis-function
# orientation extraction (edge directions / face normals) and the
# orientation-id / node-normal-set helpers now live in the packages:
#   * `AdaptiveCrossApproximation.rwgorientations(space)` — BEAST extension
#     (ACABEAST) extracts edge directions and face normals from an RT space.
#   * `AdaptiveCrossApproximation.basisfunction_orientation_ids`
#   * `AdaptiveCrossApproximation.node_normal_orientation_sets`
# so the local `rwg_orientations`/`dirkey`/`orientation_keys`/… helpers from the
# old `utils.jl` have been dropped here.
#
# Renames vs. the old interface:
#   iACA(...)                  -> IACA(...)
#   OversampIFNormEst(tol)     -> PhaseExtrapolator(tol)   (orientation-filtered path)
#   FNormExtrapolator(iFNormEstimator(tol)) -> FNormExtrapolator(tol) (scalar path)

const ACA = AdaptiveCrossApproximation

function fullfarmat(A, h2mat)
    nears = h2mat.nearinteractions
    for i in eachindex(nears.blocks)
        A[nears.rowindices[i], nears.colindices[i]] .= 0.0
    end
    return A
end

function edgeinfo(m)
    edges = skeleton(m, 1)

    _edgelength = Float64[]
    for (_, e) in enumerate(edges)
        push!(_edgelength, norm(diff(vertices(chart(edges, e)))))
    end
    _edgelength
    return minimum(_edgelength),
    sum(_edgelength) / length(_edgelength),
    maximum(_edgelength)
end

function assembleh2X(op, X; minvalues=100, tol=1e-3, maxrank=80)
    Random.seed!(1)
    testtree = KMeansTree(
        X.pos, 2; minvalues=minvalues, updateradii=H2Trees.unsafemaxradiusboundingsphere
    )
    tree = H2Trees.BlockTree(testtree, testtree)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=3.0, γ=1.0)

    # Orientation data: edge directions + face normals per basis function, from
    # the BEAST extension. Test and trial tree are the same object here, so a
    # single node-normal-set computation is reused for both compressors.
    edges, normals = ACA.rwgorientations(X)
    edgeids, normalids, nedgeids, nnormalids = ACA.basisfunction_orientation_ids(
        edges, normals
    )
    node_normal_sets, nodenormalids, nnodenormalids = ACA.node_normal_orientation_sets(
        normals, testtree
    )

    testcompressor = NestedCrossApproximation.BottomUp(;
        factorization=IACA(
            MaximumValue(),
            ACA.TreeMimicryPivoting2(
                X.pos, X.pos, edgeids, normalids, node_normal_sets, testtree
            ),
            PhaseExtrapolator(tol),
        ),
    )
    trialcompressor = NestedCrossApproximation.BottomUp(;
        factorization=IACA(
            ACA.TreeMimicryPivoting2(
                X.pos, X.pos, edgeids, normalids, node_normal_sets, testtree
            ),
            MaximumValue(),
            PhaseExtrapolator(tol),
        ),
    )
    h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
        op,
        X,
        X,
        tree;
        maxrank         = maxrank,
        isnear          = isnear,
        testcompressor  = testcompressor,
        trialcompressor = trialcompressor,
        scheduler       = DynamicScheduler(),
        verbose         = true,
    )
    return h2mat
end

function assembleh2(op, X, Y; k=0.0, minvalues=100, tol=1e-3, maxrank=80)
    Random.seed!(1)
    testtree = KMeansTree(
        X.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    )
    trialtree = KMeansTree(
        Y.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    )
    tree = H2Trees.BlockTree(testtree, trialtree)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=3.0, γ=1.0)
    testcompressor = NestedCrossApproximation.BottomUp(;
        factorization=IACA(
            MaximumValue(),
            TreeMimicryPivoting(X.pos, Y.pos, H2Trees.trialtree(tree)),
            FNormExtrapolator(tol),
        ),
    )
    trialcompressor = NestedCrossApproximation.BottomUp(;
        factorization=IACA(
            TreeMimicryPivoting(Y.pos, X.pos, H2Trees.testtree(tree)),
            MaximumValue(),
            FNormExtrapolator(tol),
        ),
    )
    h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
        op,
        X,
        Y,
        tree;
        isnear=isnear,
        maxrank=maxrank,
        testcompressor=testcompressor,
        trialcompressor=trialcompressor,
        scheduler=DynamicScheduler(),
        verbose=true,
    )
    return h2mat
end

function assembleh(op, space; maxrank=40, tol=1e-3, minvalues=100)
    Random.seed!(1)
    testtree = KMeansTree(space.pos, 2; minvalues=minvalues)
    trialtree = testtree
    tree = BlockTree(testtree, trialtree)
    mat = HM.PetrovGalerkinHMatrix(
        op,
        space,
        space,
        tree;
        compressor=ACA(; convergence=FNormEstimator(tol)),
        maxrank=maxrank,
    )
    return mat
end
