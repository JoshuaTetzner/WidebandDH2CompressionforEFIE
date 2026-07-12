using JLD2
using BEAST, CompScienceMeshes
using LinearAlgebra
using StaticArrays

@inline unit(v) = v / norm(v)
@inline cz(x) = iszero(x) ? zero(x) : x
@inline clean(v) = typeof(v)(cz(v[1]), cz(v[2]), cz(v[3]))
@inline roundvec(v; digits=1) = clean(
    typeof(v)(
        round(v[1]; digits=digits),
        round(v[2]; digits=digits),
        round(v[3]; digits=digits),
    ),
)

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

@inline function canon(v)
    v = clean(v)
    return if v[1] < 0 || (v[1] == 0 && v[2] < 0) || (v[1] == 0 && v[2] == 0 && v[3] < 0)
        clean(-v)
    else
        v
    end
end

@inline function edgeverts(cell, refid)
    refid == 1 && return cell[2], cell[3]
    refid == 2 && return cell[3], cell[1]
    return cell[1], cell[2]
end

@inline function cellnormal(verts, cell)
    return cross(verts[cell[2]] - verts[cell[1]], verts[cell[3]] - verts[cell[1]])
end

function rwg_orientations(rwg)
    mesh  = rwg.geo
    verts = vertices(mesh)
    cells = collect(CompScienceMeshes.cells(mesh))
    nc    = length(cells)

    ℓ  = similar(rwg.pos)
    n  = similar(rwg.pos)
    cn = similar(rwg.pos, nc)

    @inbounds for c in 1:nc
        cn[c] = clean(unit(cellnormal(verts, cells[c])))
    end

    @inbounds for i in eachindex(rwg.fns)
        fn = rwg.fns[i]
        sh = fn[1]

        cell = cells[sh.cellid]
        a, b = edgeverts(cell, sh.refid)

        ℓ[i] = roundvec(canon(unit(verts[b] - verts[a])))

        ni = cn[fn[1].cellid]
        if length(fn) > 1
            nj = cn[fn[2].cellid]
            ni += dot(ni, nj) < 0 ? -nj : nj
        end
        n[i] = roundvec(unit(ni))
    end

    return ℓ, n
end

@inline function dirkey(v; digits=1)
    s = 10.0^digits
    x = round(Int16, v[1] * s)
    y = round(Int16, v[2] * s)
    z = round(Int16, v[3] * s)

    return if x < 0 || (x == 0 && y < 0) || (x == 0 && y == 0 && z < 0)
        (-x, -y, -z)
    else
        (x, y, z)
    end
end

@inline orientation_key(ℓ, n; dell=1, dn=1) = (dirkey(ℓ; digits=dell), dirkey(n; digits=dn))

function orientation_keys(ℓ, n; dell=1, dn=1)
    K = typeof(orientation_key(ℓ[1], n[1]; dell=dell, dn=dn))
    q = Vector{K}(undef, length(ℓ))

    @inbounds for i in eachindex(ℓ)
        q[i] = orientation_key(ℓ[i], n[i]; dell=dell, dn=dn)
    end

    return q
end

function assembleh2X(op, X; k=0.0, minvalues=100, tol=1e-3, maxrank=80)
    Random.seed!(1)
    testtree = KMeansTree(
        X.pos, 2; minvalues=minvalues, updateradii=H2Trees.unsafemaxradiusboundingsphere
    )
    tree = H2Trees.BlockTree(testtree, testtree)
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=1.0, γ=1.0)
    edges, normals = rwg_orientations(X)
    edgeids, normalids, nedgeids, nnormalids = NestedCrossApproximation.basisfunction_orientation_ids(
        edges, normals
    )
    trial_node_normal_sets, trialnormalids, ntrialnormalids = NestedCrossApproximation.node_normal_orientation_sets(
        normals, testtree
    )
    test_node_normal_sets, testnormalids, ntestnormalids = NestedCrossApproximation.node_normal_orientation_sets(
        normals, testtree
    )
    testcompressor = NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            AdaptiveCrossApproximation.TreeMimicryPivoting2(
                X.pos, X.pos, edgeids, trialnormalids, trial_node_normal_sets, testtree
            ),
            OversampIFNormEst(tol),
        ),
    )
    trialcompressor = NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            AdaptiveCrossApproximation.TreeMimicryPivoting2(
                X.pos, X.pos, edgeids, testnormalids, test_node_normal_sets, testtree
            ),
            MaximumValue(),
            OversampIFNormEst(tol),
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
        matrixdata      = BEAST.DoubleNumWiltonSauterQStrat(3, 4, 5, 6, 5, 6, 5, 6),
        farmatrixdata   = BEAST.DoubleNumQStrat(3, 4),
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
    isnear = NestedCrossApproximation.isnearwideband(k; ηhf=1.0, γ=1.0)
    testcompressor = NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            TreeMimicryPivoting(X.pos, Y.pos, H2Trees.trialtree(tree)),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    )
    trialcompressor = NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            TreeMimicryPivoting(Y.pos, X.pos, H2Trees.testtree(tree)),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
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
        matrixdata      = BEAST.DoubleNumWiltonSauterQStrat(3, 4, 5, 6, 5, 6, 5, 6),
        farmatrixdata   = BEAST.DoubleNumQStrat(3, 4),
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
