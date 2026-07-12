using DelimitedFiles
using StaticArrays

function writePgfplots(Γ::Mesh, filename)
    x = ones(Float64, 3 * length(Γ.faces), 4) # initialize array to store information
    ind = 1
    for (i, face) in enumerate(Γ.faces) # loop over all triangles
        x[ind, 1:3]     = Γ.vertices[face[1]] # write point 1 of triangle into array
        x[ind + 1, 1:3] = Γ.vertices[face[2]] #
        x[ind + 2, 1:3] = Γ.vertices[face[3]] #

        ind += 3
    end

    # write header
    writedlm(filename, ["x y z c"])

    # append data
    open(filename, "a") do io
        writedlm(io, x)
    end
end

function writePgfplots(Γ::Vector{SVector{3,Float64}}, filename)
    x = ones(Float64, length(Γ), 4) # initialize array to store information
    ind = 1
    for (i, face) in enumerate(Γ) # loop over all triangles
        x[i, 1] = face[1] # write point 1 of triangle into array
        x[i, 2] = face[2] #
        x[i, 3] = face[3] #

        ind += 3
    end

    # write header
    writedlm(filename, ["x y z c"])

    # append data
    open(filename, "a") do io
        writedlm(io, x)
    end
end

function writePgfplots(Γ::Mesh, c::Vector{Float64}, filename)
    x = ones(Float64, 3 * length(Γ.faces), 4) # initialize array to store information
    ind = 1
    for (i, face) in enumerate(Γ.faces) # loop over all triangles
        x[ind, 1:3] = Γ.vertices[face[1]] # write point 1 of triangle into array
        x[ind + 1, 1:3] = Γ.vertices[face[2]] #
        x[ind + 2, 1:3] = Γ.vertices[face[3]] #
        x[ind:(ind + 2), 4] .= c[i]
        ind += 3
    end

    # write header
    writedlm(filename, ["x y z c"])

    # append data
    open(filename, "a") do io
        writedlm(io, x)
    end
end

function writePgfplots(Γ::Mesh, faces::Vector{Int}, c::Vector{Float64}, filename)
    x = ones(Float64, 3 * length(faces), 4) # initialize array to store information
    ind = 1
    for (i, face) in enumerate(Γ.faces[faces]) # loop over all triangles
        x[ind, 1:3] = Γ.vertices[face[1]] # write point 1 of triangle into array
        x[ind + 1, 1:3] = Γ.vertices[face[2]] #
        x[ind + 2, 1:3] = Γ.vertices[face[3]] #
        x[ind:(ind + 2), 4] .= c[i]
        ind += 3
    end

    # write header
    writedlm(filename, ["x y z c"])

    # append data
    open(filename, "a") do io
        writedlm(io, x)
    end
end
