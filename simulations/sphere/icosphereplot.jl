"""
    Write mesh information to textfile for plotting in LaTex with pgfplots.


    Generated file format for pgfplots:

        x y z c
        * * * *   |
        * * * *    > three rows form a triangle
        * * * *   |
        * * * *
        ...

    Example for LaTex:

        \addplot3[patch,patch type=triangle,
        	style={line width=0.05pt},
        	shader=faceted,
        	faceted color=black,
        	axis equal,
        	color=copper,
        	opacity=0.9,
        	patch refines=0,
        	point meta=explicit	]
        table[x=x,y=y,z=z,
        	  meta =c] {mySphere.txt};
"""
function writePgfplots(Γ::Mesh, filename)
    x = ones(Float64, 3*length(Γ.faces), 4) # initialize array to store information
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
        return writedlm(io, x)
    end
end

m = meshicosphere(3, 1.0)

writePgfplots(m, "icosphere.txt")
