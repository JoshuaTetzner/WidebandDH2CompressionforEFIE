using CompScienceMeshes
using BEAST
using AdaptiveCrossApproximation
using NestedCrossApproximation
using H2Trees
using ParallelKMeans
using LinearAlgebra
using OhMyThreads
using CSV, DataFrames
using Random
using StaticArrays

include("poweriteration.jl")
include("trees.jl")

# One driver for every matrix-compression experiment of the paper. It builds all
# five approximations of the same operator on the same discretization and reports
# storage, far-field error, assembly time and matrix-vector time for each:
#
#   key          paper label              method
#   hmataca      H, ACA                   H-matrix, standard ACA (Frobenius-norm
#                                         stopping criterion)
#   hmatrs       H, ACA-RS                H-matrix, ACA with the additional
#                                         random-sampling criterion
#   nca          DH², NCA                 directional H², wideband admissibility,
#                                         plain tree mimicry pivoting
#   ncaefie      DH², NCA-EFIE            the same, but with the EFIE directional
#                                         filter
#   ncaefieoct   DH², NCA-EFIE (octree)   ncaefie on an octree instead of the
#                                         k-means tree
#
# hmatrs, ncaefie and ncaefieoct are the three error-controlled compressions:
# their stopping criterion tracks the true far-field error, so they attain the
# requested tolerance. hmataca and nca do not.
#
# The first four share the k-means BlockTree; `ncaefieoct` needs its own octree,
# and therefore its own reference matrix, because the error of a compression is
# only meaningful against a reference that uses the same near/far split.
#
# Only one approximation is alive at a time (plus the reference), so the peak
# memory is that of the largest single matrix rather than the sum of all five.

# --- output schema -----------------------------------------------------------

# The four methods sharing the k-means BlockTree, and the one on the octree.
const KMEANS_METHODS = ("hmataca", "hmatrs", "nca", "ncaefie")
const OCTREE_METHODS = ("ncaefieoct",)
const COMPARE_METHODS = (KMEANS_METHODS..., OCTREE_METHODS...)

"""
    compareframe(methods=COMPARE_METHODS) -> DataFrame

Empty `DataFrame` with the schema written by [`simcompare`](@ref): the run
parameters, followed by storage (GB), far-field error, assembly time (s) and
matrix-vector time (s) for each of `methods`.

Write it once per experiment to create the CSV header; `simcompare` then appends
one row per discretization. Pass the same `methods` to both, so that the header
and the rows agree.
"""
function compareframe(methods=COMPARE_METHODS)
    methods = canonicalmethods(methods)
    df = DataFrame(;
        k=Float64[], gamma=Float64[], etahf=Float64[], tol=Float64[], N=Int[]
    )
    for method in methods, quantity in ("stor", "err", "tass", "tmv")
        df[!, quantity * method] = Float64[]
    end
    return df
end

"""
    canonicalmethods(methods) -> Vector{String}

Validate `methods` and return them in the fixed order of `COMPARE_METHODS`, so the
column order of an experiment never depends on how the methods were listed.
"""
function canonicalmethods(methods)
    requested = string.(collect(methods))
    for method in requested
        method in COMPARE_METHODS || error(
            "unknown method \"$method\"; choose from " * join(COMPARE_METHODS, ", ")
        )
    end
    return [method for method in COMPARE_METHODS if method in requested]
end

# --- the five compressed matrices --------------------------------------------

"""
    hmatrix(op, tspace, sspace, tree, isnear; convergence, maxrank=50, scheduler)

H-matrix on `tree`, compressed with `ACA` using the given `convergence` criterion.
"""
function hmatrix(
    op,
    tspace,
    sspace,
    tree,
    isnear;
    convergence,
    maxrank=50,
    scheduler=DynamicScheduler(),
)
    return HMatrix(
        op,
        tspace,
        sspace,
        tree;
        isnear=isnear,
        maxrank=maxrank,
        spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
        compressor=ACA(; convergence=convergence),
        scheduler=scheduler,
    )
end

"""
    acaconvergence(tol)

Standard ACA stopping criterion: the Frobenius-norm estimator alone.
"""
acaconvergence(tol) = FNormEstimator(tol)

"""
    randomsamplingconvergence(tol; factor=1.0)

Frobenius-norm estimator combined with the random-sampling criterion, which
catches the blocks whose error the norm estimator alone underestimates.
"""
function randomsamplingconvergence(tol; factor=1.0)
    return AdaptiveCrossApproximation.CombinedConvCrit([
        FNormEstimator(tol),
        AdaptiveCrossApproximation.RandomSampling(; tol=tol, factor=factor),
    ])
end

"""
    ncamatrix(op, tspace, sspace, tree, isnear; tol, filtered, maxrank=50, scheduler)

Directional H²-matrix built by the nested cross approximation. `filtered`
selects the EFIE directional filter; see [`ncapivoting`](@ref).
"""
function ncamatrix(
    op,
    tspace,
    sspace,
    tree,
    isnear;
    tol=1e-3,
    filtered=true,
    maxrank=50,
    scheduler=DynamicScheduler(),
)
    testpivoting, trialpivoting, convergence = ncapivoting(
        tspace, sspace, tree; filtered=filtered, tol=tol
    )

    return NestedCrossApproximation.PetrovGalerkinNCA(
        op,
        tspace,
        sspace,
        tree;
        testcompressor=NestedCrossApproximation.BottomUp(;
            factorization=IACA(MaximumValue(), testpivoting, convergence)
        ),
        trialcompressor=NestedCrossApproximation.BottomUp(;
            factorization=IACA(trialpivoting, MaximumValue(), convergence)
        ),
        maxrank=maxrank,
        isnear=isnear,
        scheduler=scheduler,
    )
end

"""
    referencematrix(op, tspace, sspace, tree, isnear; tol, maxrank=100, scheduler)

Far-field reference for the error measurement: an H-matrix on the same `tree` and
with the same admissibility as the matrices under test, but compressed two orders
of magnitude tighter. Sharing the tree is what makes the comparison meaningful --
the near-field blocks then cancel exactly in the difference, so the reported error
is purely the far-field compression error.
"""
function referencematrix(
    op, tspace, sspace, tree, isnear; tol=1e-3, maxrank=100, scheduler=DynamicScheduler()
)
    return hmatrix(
        op,
        tspace,
        sspace,
        tree,
        isnear;
        convergence=randomsamplingconvergence(tol * 1e-2),
        maxrank=maxrank,
        scheduler=scheduler,
    )
end

"""
    matrixbuilder(method, op, tspace, sspace, tree, isnear; tol, maxrank, scheduler)

Zero-argument closure that builds the approximation named `method` on `tree`.
"""
function matrixbuilder(
    method, op, tspace, sspace, tree, isnear; tol=1e-3, maxrank=50, scheduler
)
    if method == "hmataca"
        return () -> hmatrix(
            op,
            tspace,
            sspace,
            tree,
            isnear;
            convergence=acaconvergence(tol),
            maxrank=maxrank,
            scheduler=scheduler,
        )
    elseif method == "hmatrs"
        return () -> hmatrix(
            op,
            tspace,
            sspace,
            tree,
            isnear;
            convergence=randomsamplingconvergence(tol),
            maxrank=maxrank,
            scheduler=scheduler,
        )
    elseif method == "nca"
        return () -> ncamatrix(
            op,
            tspace,
            sspace,
            tree,
            isnear;
            tol=tol,
            filtered=false,
            maxrank=maxrank,
            scheduler=scheduler,
        )
    elseif method in ("ncaefie", "ncaefieoct")
        return () -> ncamatrix(
            op,
            tspace,
            sspace,
            tree,
            isnear;
            tol=tol,
            filtered=true,
            maxrank=maxrank,
            scheduler=scheduler,
        )
    end
    return error("unknown method \"$method\"")
end

# --- measurement -------------------------------------------------------------

# `storage` and `farmatrix` exist in both packages but are separate generic
# functions, so pick the right one by the matrix type.
_storage(mat::HMatrix) = AdaptiveCrossApproximation.storage(mat)
_storage(mat::NestedCrossApproximation.PetrovGalerkinNCA) =
    NestedCrossApproximation.storage(mat)

_farmatrix(mat::HMatrix) = AdaptiveCrossApproximation.farmatrix(mat)
_farmatrix(mat::NestedCrossApproximation.PetrovGalerkinNCA) =
    NestedCrossApproximation.farmatrix(mat)

"""
    measure(name, build, refmat, refnorm; tol=1e-3, nmv=20, seed=1)

Build one approximation, measure it, and let it go out of scope again.

Returns `(stor, err, tass, tmv)`: storage in GB, relative far-field error against
`refmat`, assembly time and the fastest of `nmv` matrix-vector products. `refnorm`
is the precomputed spectral norm of `refmat`, so that comparing several matrices
against the same reference does not repeat that power iteration.

`seed` reseeds the root task's RNG immediately before the assembly. The
`RandomSampling` convergence criterion draws its sample positions from the
task-local RNG, which Julia derives from the root RNG when the assembly spawns
its tasks -- without this, the sampled entries, and with them the block ranks,
differ from run to run. See [`simcompare`](@ref) for the thread-count caveat.
"""
function measure(name, build, refmat, refnorm; tol=1e-3, nmv=20, seed=1)
    println("\n===== $name =====")

    Random.seed!(seed)
    tass = @elapsed mat = build()
    println("assembly: ", tass, " s")

    x = rand(Random.MersenneTwister(seed), eltype(mat), size(mat, 2))
    mat * x  # discard the first product, it pays for compilation and page faults
    tmv = minimum(@elapsed(mat * x) for _ in 1:nmv)
    println("mat-vec: ", tmv, " s")

    stor = _storage(mat)
    if iszero(refnorm)
        # No admissible block pair on this tree: everything landed in the near
        # field, so there is no far-field error to measure. Happens when the
        # discretization is too coarse for the tree to have enough levels.
        @warn "$name: reference far field is empty, reporting the error as NaN"
        err = NaN
    else
        err = estimate_reldifference(
            _farmatrix(mat), refmat; tol=tol * 1e-1, refnorm=refnorm
        )
    end
    println("far-field error: ", err)

    return stor, err, tass, tmv
end

# --- the experiment ----------------------------------------------------------

"""
    simcompare(filename, op, tspace, sspace, isnear; tol, ηhf, γ, kwargs...)

Run the selected compressions of `op` on one discretization and append a row to
`filename`. Create the file with `CSV.write(filename, compareframe(methods))`
first, passing the same `methods`.

`methods` selects any subset of `COMPARE_METHODS`; the default runs all five. The
row carries storage, far-field error, assembly time and matrix-vector time for
each selected method, always in the canonical column order of
[`compareframe`](@ref). Whole groups are skipped when nothing in them is asked
for: selecting only `"ncaefieoct"` builds no k-means tree and no k-means reference
matrix, and selecting only k-means methods builds no octree.

The reported times are wall-clock times at the thread count Julia was started
with; keep that count fixed across a series of runs so they stay comparable.

# Reproducibility

`seed` fixes every random draw in the pipeline: the k-means clustering, the sample
positions of the `RandomSampling` convergence criterion, and the start vectors of
the power iterations that estimate the errors. Two runs at the same thread count
therefore produce bit-identical `stor…` and `err…` values; the `tass…`/`tmv…`
columns are wall-clock times and vary from run to run.

Across *different* thread counts the `hmatrs` column and the reference matrix
still move slightly: the assembly spawns one task per chunk, Julia derives each
task's RNG from the root RNG at spawn time, and the number of chunks follows
`Threads.nthreads()`. The sampled entries therefore differ, which shifts a few
block ranks. Use the same thread count across a series of runs that has to be
comparable -- any fixed count will do.
"""
function simcompare(
    filename,
    op,
    tspace,
    sspace,
    isnear;
    methods=COMPARE_METHODS,
    tol=1e-3,
    ηhf=1.0,
    γ=1.0,
    maxrank=50,
    refmaxrank=100,
    kmeansminvalues=100,
    octreeminvalues=200,
    seed=1,
    nmv=20,
    scheduler=DynamicScheduler(),
)
    methods = canonicalmethods(methods)
    N = length(tspace)
    println("\n######## N = $N, tol = $tol, methods = $(join(methods, ", ")) ########")
    results = Dict{String,NTuple{4,Float64}}()

    # --- k-means tree, shared by up to four methods --------------------------
    kmeansselection = [method for method in KMEANS_METHODS if method in methods]
    if !isempty(kmeansselection)
        tree = kmeansblocktree(tspace, sspace; minvalues=kmeansminvalues, seed=seed)

        println("\n===== reference (k-means tree) =====")
        Random.seed!(seed)
        refmat = _farmatrix(
            referencematrix(
                op,
                tspace,
                sspace,
                tree,
                isnear;
                tol=tol,
                maxrank=refmaxrank,
                scheduler=scheduler,
            ),
        )
        refnorm = estimate_norm(refmat; tol=tol * 1e-1)

        for method in kmeansselection
            results[method] = measure(
                method,
                matrixbuilder(
                    method,
                    op,
                    tspace,
                    sspace,
                    tree,
                    isnear;
                    tol=tol,
                    maxrank=maxrank,
                    scheduler=scheduler,
                ),
                refmat,
                refnorm;
                tol=tol,
                nmv=nmv,
                seed=seed,
            )
            GC.gc()
        end

        refmat = nothing
        tree = nothing
        GC.gc()
    end

    # --- octree, with its own reference matrix -------------------------------
    octreeselection = [method for method in OCTREE_METHODS if method in methods]
    if !isempty(octreeselection)
        octree = octreeblocktree(tspace, sspace; minvalues=octreeminvalues)

        println("\n===== reference (octree) =====")
        Random.seed!(seed)
        octrefmat = _farmatrix(
            referencematrix(
                op,
                tspace,
                sspace,
                octree,
                isnear;
                tol=tol,
                maxrank=refmaxrank,
                scheduler=scheduler,
            ),
        )
        octrefnorm = estimate_norm(octrefmat; tol=tol * 1e-1)

        for method in octreeselection
            results[method] = measure(
                method,
                matrixbuilder(
                    method,
                    op,
                    tspace,
                    sspace,
                    octree,
                    isnear;
                    tol=tol,
                    maxrank=maxrank,
                    scheduler=scheduler,
                ),
                octrefmat,
                octrefnorm;
                tol=tol,
                nmv=nmv,
                seed=seed,
            )
            GC.gc()
        end

        octrefmat = nothing
        octree = nothing
        GC.gc()
    end

    # --- one row -------------------------------------------------------------
    row = Dict{Symbol,Any}(
        :k => isnear.k, :gamma => γ, :etahf => ηhf, :tol => tol, :N => N
    )
    for method in methods
        stor, err, tass, tmv = results[method]
        row[Symbol("stor" * method)] = stor
        row[Symbol("err" * method)] = err
        row[Symbol("tass" * method)] = tass
        row[Symbol("tmv" * method)] = tmv
    end

    df = compareframe(methods)
    push!(df, row)

    return CSV.write(filename, df; append=true, header=false)
end
