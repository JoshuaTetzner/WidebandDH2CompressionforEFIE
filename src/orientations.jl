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

        ℓ[i] = canon(unit(verts[b] - verts[a]))

        ni = cn[fn[1].cellid]
        if length(fn) > 1
            nj = cn[fn[2].cellid]
            if abs(dot(ni, nj)) < 0.9
                n[i] = zero(ni)
                continue
            end
            ni += dot(ni, nj) < 0 ? -nj : nj
        end
        n[i] = clean(unit(ni))
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
