using BEAST
using CompScienceMeshes
using SparseArrays

function rttobc(Γ; edges=:all, kwargs...)
    # Buffa-Christiansen functions are only defined on interior (and junction)
    # edges. On an open mesh the free-boundary edges have no associated BC
    # function: BEAST's `buildhalfbc` cannot construct one for a port that sits
    # on the open boundary and errors out. Drop those edges here so that open
    # geometries are handled gracefully. (`buffachristiansen` does the same
    # filtering itself when `edges == :all`.)
    if edges !== :all
        bnd = boundary(Γ)
        in_bnd = CompScienceMeshes.in(bnd)
        edges = submesh(!in_bnd, edges)
    end

    @time bc = buffachristiansen(Γ; edges=edges, kwargs...)

    Γfine = geometry(bc).mesh

    # Create the full Raviart-Thomas space on the refined mesh (including half RTs)
    rt = raviartthomas(Γfine, skeleton(Γfine, 1))

    _, rtad = assemblydata(rt; onlyactives=false)

    I = Int[]
    J = Int[]
    V = coordtype(Γ)[]

    for (bcid, bcfn) in enumerate(bc.fns)
        for shape in bcfn
            cellid = shape.cellid
            refid = shape.refid
            coeff = shape.coeff

            iszero(coeff) && continue

            for (rtid, rt_coeff) in rtad[cellid, refid]
                push!(I, bcid)
                push!(J, rtid)
                if length(rt.fns[rtid]) == 2
                    # Internal edge, distribute coeff equally
                    push!(V, coeff * rt_coeff / 2)
                else
                    # Boundary edge, take full coeff
                    push!(V, coeff * rt_coeff)
                end
            end
        end
    end

    return sparse(I, J, V, numfunctions(bc), numfunctions(rt)), bc, rt
end
