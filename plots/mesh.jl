using CompScienceMeshes
using NestedCrossApproximation
using ParallelKMeans
using H2Trees
using PlotlyJS
using BEAST
using Random

λ = 0.2
k = 2 * pi / λ
m = meshcuboid(1.0, 1.0, 1.0, 0.02)
rwg = raviartthomas(m)
Random.seed!(1234)
tree = KMeansTree(rwg.pos, 2; minvalues=100)
blktree = BlockTree(tree, tree)
##
isnear = NestedCrossApproximation.isnearwideband(k; ηhf=5.0, γ=1.0);
@time fardata, trialfardata = NestedCrossApproximation.fardata(blktree, isnear)
edges, normals = rwg_orientations(rwg)
unique(normals)
edgeids, normalids, nedgeids, nnormalids = NestedCrossApproximation.basisfunction_orientation_ids(
    edges, normals
)
normalset, _, _ = NestedCrossApproximation.node_normal_orientation_sets(normals, tree)
##
level = H2Trees.levels(tree)[end]
t = collect(H2Trees.LevelIterator(tree, level))[1]
for (idx, dir) in enumerate(NestedCrossApproximation.dirrange(fardata, t))
    Ft = NestedCrossApproximation.dirfarfield(tree, fardata, t, fardata.dirs[dir])
    println(idx, "; ", length(Ft))
end

dir = NestedCrossApproximation.dirrange(fardata, t)[2]
Ft = NestedCrossApproximation.dirfarfield(tree, fardata, t, fardata.dirs[dir])
##
using Plots
plotlyjs()
tidcs = H2Trees.values(tree, t)
sidcs = H2Trees.values(tree, Ft)

spos = rwg.pos[sidcs]
tpos = rwg.pos[tidcs]

p = Plots.scatter(
    getindex.(spos, 1),
    getindex.(spos, 2),
    getindex.(spos, 3);
    aspect_ratio=:equal,
    xlims=(0, 1),
    ylims=(0, 1),
    zlims=(0, 1),
);
Plots.scatter!(getindex.(tpos, 1), getindex.(tpos, 2), getindex.(tpos, 3));

Plots.display(p)
##
normalids
Ftweight = Int32[]
Ftfaces = Int[]
for i in H2Trees.values(tree, Ft)
    !(rwg.fns[i][1].cellid in Ftfaces) &&
        (push!(Ftfaces, rwg.fns[i][1].cellid); push!(Ftweight, normalids[i]))
    !(rwg.fns[i][2].cellid in Ftfaces) &&
        (push!(Ftfaces, rwg.fns[i][2].cellid); push!(Ftweight, normalids[i]))
end
tfaces = Int[]
for i in H2Trees.values(tree, t)
    !(rwg.fns[i][1].cellid in tfaces) && push!(tfaces, rwg.fns[i][1].cellid)
    !(rwg.fns[i][2].cellid in tfaces) && push!(tfaces, rwg.fns[i][2].cellid)
end
##
writePgfplots(m, "cube.txt")
writePgfplots(m, Ftfaces, Float64.(Ftweight), "Ft.txt")
writePgfplots(m, tfaces, ones(Float64, length(tfaces)), "t.txt")
