
using BEAST
using CompScienceMeshes
using SparseArrays

function getstars(mesh::CompScienceMeshes.AbstractMesh)
    meshboundary = boundary(mesh)
    edges = setminus(skeleton(mesh, 1), meshboundary)
    return CompScienceMeshes.connectivity(mesh, edges, sign)
end

function rt_divergence_matrix(space)
    geo = geometry(space)

    first_shape = first(first(space.fns))
    T = typeof(first_shape.coeff / volume(chart(geo, first_shape.cellid)))

    I = Int[]
    J = Int[]
    V = T[]

    nnz_est = sum(length, space.fns)
    sizehint!(I, nnz_est)
    sizehint!(J, nnz_est)
    sizehint!(V, nnz_est)

    for (fnid, fn) in enumerate(space.fns)
        for sh in fn
            push!(I, fnid)
            push!(J, sh.cellid)
            push!(V, sh.coeff / volume(chart(geo, sh.cellid)))
        end
    end

    return sparse(I, J, V, numfunctions(space), numcells(geo))
end
