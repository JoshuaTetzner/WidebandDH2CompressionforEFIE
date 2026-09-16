using CodecZlib

"""
    geometrypath(dir, name) -> String

Return the path of the mesh `name` inside the geometry directory `dir`.

The rafale meshes are too large to be version-controlled uncompressed, so they ship
as `<name>.gz`. The first call expands the archive next to it and every later call
reuses the expanded file. Expanded `*.msh` files are excluded by `.gitignore`, so
they can be deleted at any time to reclaim disk space.
"""
function geometrypath(dir, name)
    path = joinpath(dir, name)
    isfile(path) && return path

    archive = path * ".gz"
    isfile(archive) || error(
        "Neither $(path) nor $(archive) exists. The meshes ship gzip-compressed " *
        "in the geometry directory of each simulation.",
    )

    @info "Expanding $(basename(archive)) (one-time, the result is gitignored)"

    # Expand into a temporary file first so that an interrupted run cannot leave a
    # truncated mesh behind that later calls would happily read.
    tmp = path * ".tmp"
    try
        open(archive) do input
            open(tmp, "w") do output
                return write(output, GzipDecompressorStream(input))
            end
        end
        mv(tmp, path; force=true)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end

    return path
end
