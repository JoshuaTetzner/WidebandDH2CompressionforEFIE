using CompScienceMeshes
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using H2Trees
using OhMyThreads
using CSV, DataFrames
using StaticArrays
include("poweriteration.jl")

@inline unit(v) = v / norm(v)
@inline cz(x) = iszero(x) ? zero(x) : x
@inline clean(v) = typeof(v)(cz(v[1]), cz(v[2]), cz(v[3]))
@inline roundvec(v; digits=2) = clean(
    typeof(v)(
        round(v[1]; digits=digits),
        round(v[2]; digits=digits),
        round(v[3]; digits=digits),
    ),
)

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

function rwg_orientations(rwg)
    mesh  = rwg.geo
    verts = vertices(mesh)
    cells = collect(CompScienceMeshes.cells(mesh))
    nc    = length(cells)

    ℓ  = similar(rwg.pos)
    n  = similar(rwg.pos)
    cn = similar(rwg.pos, nc)

    @inbounds for c in 1:nc
        cn[c] = clean(unit(normal(chart(mesh, cells[c]))))
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
function simfull(
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
    trial_node_normal_sets = NestedCrossApproximation.node_orientation_sets(
        tnormalids, tree.trialcluster, tnnormalids
    )
    test_node_normal_sets = NestedCrossApproximation.node_orientation_sets(
        snormalids, tree.testcluster, snnormalids
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
                    snormalids,
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
                    tnormalids,
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

    @time hmat = AdaptiveCrossApproximation.HMatrix(
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

    x = rand(eltype(hmat), length(sspace))
    println(norm(h2mat * x - hmat * x) / norm(hmat * x))
    println(norm(farh2mat * x - farhmat * x) / norm(farhmat * x))
    y = h2mat * x
    tmvh2mat = 0.0
    for _ in 1:10
        tmvh2mat += @elapsed y = h2mat * x
    end
    tmvh2mat = tmvh2mat / 10

    farerrh2mat = estimate_reldifference(farh2mat, farhmat; tol=tol * 1e-1)
    errh2mat = estimate_reldifference(h2mat, hmat; tol=tol * 1e-1)
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
        errh2mat=errh2mat,
    )

    return CSV.write(filename, df; append=true, header=false)
end
