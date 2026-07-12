using CompScienceMeshes
using WriteVTK
using DelimitedFiles
using LinearAlgebra
using Statistics

function face_indices(face)
    return (face[1], face[2], face[3])
end

function tri_points(Γ, face)
    p1 = collect(vertices(Γ, face[1]))
    p2 = collect(vertices(Γ, face[2]))
    p3 = collect(vertices(Γ, face[3]))
    return p1, p2, p3
end

function tri_area(p1, p2, p3)
    return 0.5 * norm(cross(p2 - p1, p3 - p1))
end

function tri_center(p1, p2, p3)
    return (p1 + p2 + p3) / 3
end

function triangle_center_area(Γ, face)
    p1, p2, p3 = tri_points(Γ, face)
    return tri_center(p1, p2, p3), tri_area(p1, p2, p3)
end

function refined_region_center(
    Γ;
    fraction=0.10,
    front_axis=3,
    front_quantile=0.20,
    vertical_axis=2,
    vertical_quantile=0.65,
)
    centers = Vector{Vector{Float64}}()
    areas = Float64[]

    for face in cells(Γ)
        c, a = triangle_center_area(Γ, face)
        push!(centers, c)
        push!(areas, a)
    end

    center_matrix = reduce(hcat, centers)
    front_cut = quantile(center_matrix[front_axis, :], front_quantile)
    vertical_cut = quantile(center_matrix[vertical_axis, :], vertical_quantile)
    candidate_ids = findall(eachindex(areas)) do i
        return center_matrix[front_axis, i] <= front_cut &&
               center_matrix[vertical_axis, i] >= vertical_cut
    end

    isempty(candidate_ids) && error("No nozzle candidates found; relax quantile settings.")

    n = max(1, round(Int, fraction * length(candidate_ids)))
    ids = candidate_ids[sortperm(areas[candidate_ids])[1:n]]

    pts = reduce(hcat, centers[ids])
    return vec(mean(pts; dims=2))
end

function writevtk_mesh(Γ, filename; nozzle_center=nothing, nozzle_radius=Inf)
    points = hcat([collect(v) for v in vertices(Γ)]...)

    vtk_cells = [
        MeshCell(VTKCellTypes.VTK_TRIANGLE, collect(face_indices(face))) for
        face in cells(Γ)
    ]

    area = Float64[]
    refinement = Float64[]
    nozzle = Int[]

    for face in cells(Γ)
        p1, p2, p3 = tri_points(Γ, face)

        a = tri_area(p1, p2, p3)
        c = tri_center(p1, p2, p3)

        push!(area, a)
        push!(refinement, -log10(a))

        if nozzle_center === nothing
            push!(nozzle, 0)
        else
            push!(nozzle, norm(c - nozzle_center) <= nozzle_radius ? 1 : 0)
        end
    end

    vtk_grid(filename, points, vtk_cells) do vtk
        vtk["area", VTKCellData()] = area
        vtk["refinement", VTKCellData()] = refinement
        return vtk["nozzle", VTKCellData()] = nozzle
    end
end

function writePgfplots(Γ, filename; crop_center=nothing, crop_radius=Inf)
    rows = Float64[]

    for face in cells(Γ)
        p1, p2, p3 = tri_points(Γ, face)
        center = tri_center(p1, p2, p3)

        if crop_center !== nothing && norm(center - crop_center) > crop_radius
            continue
        end

        area = tri_area(p1, p2, p3)
        c = -log10(area)

        append!(rows, [p1[1], p1[2], p1[3], c])
        append!(rows, [p2[1], p2[2], p2[3], c])
        append!(rows, [p3[1], p3[2], p3[3], c])
    end

    rows = reshape(rows, 4, :)'

    open(filename, "w") do io
        println(io, "x y z c")
        return writedlm(io, rows)
    end
end

function triangle_edges(face)
    i, j, k = face_indices(face)
    return ((i, j), (j, k), (k, i))
end

function ordered_edge(i, j)
    return i < j ? (i, j) : (j, i)
end

function mesh_edges(Γ; crop_center=nothing, crop_radius=Inf)
    edge_set = Set{Tuple{Int,Int}}()

    for face in cells(Γ)
        p1, p2, p3 = tri_points(Γ, face)
        center = tri_center(p1, p2, p3)

        if crop_center !== nothing && norm(center - crop_center) > crop_radius
            continue
        end

        for (i, j) in triangle_edges(face)
            push!(edge_set, ordered_edge(i, j))
        end
    end

    return sort!(collect(edge_set))
end

function write_tikz_edge_table(Γ, filename; crop_center=nothing, crop_radius=Inf)
    edges = mesh_edges(Γ; crop_center=crop_center, crop_radius=crop_radius)

    open(filename, "w") do io
        println(io, "x y z")
        for (i, j) in edges
            p1 = vertices(Γ, i)
            p2 = vertices(Γ, j)
            println(io, "$(p1[1]) $(p1[2]) $(p1[3])")
            println(io, "$(p2[1]) $(p2[2]) $(p2[3])")
            println(io, "NaN NaN NaN")
        end
    end

    return length(edges)
end

# ------------------------
# Main script
# ------------------------


meshpath = "/home/jt286/Documents/Geometries/rafale_fuel/rafale10fuel_coarseplot.msh"
m = CompScienceMeshes.read_gmsh_mesh(meshpath)

nozzle_center = refined_region_center(m)
println("Estimated nozzle center: ", nozzle_center)

# Adjust this after checking in ParaView.
nozzle_radius = 0.45

writevtk_mesh(
    m, "rafale_refined_debug"; nozzle_center=nozzle_center, nozzle_radius=nozzle_radius
)

writePgfplots(m, "rafale_full.txt")

writePgfplots(m, "rafale_nozzle.txt"; crop_center=nozzle_center, crop_radius=nozzle_radius)

full_edges = write_tikz_edge_table(m, "rafale_edges.dat")
nozzle_edges = write_tikz_edge_table(
    m,
    "rafale_nozzle_edges.dat";
    crop_center=nozzle_center,
    crop_radius=nozzle_radius,
)
println(
    "Wrote TikZ edge tables (",
    full_edges,
    " full edges, ",
    nozzle_edges,
    " nozzle edges).",
)
