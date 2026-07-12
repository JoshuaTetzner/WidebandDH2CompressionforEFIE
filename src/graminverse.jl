using Krylov, KrylovPreconditioners, LinearMaps, LinearAlgebra

struct GramInverse{T,G,P,R,X} <: LinearMaps.LinearMap{T}
    gram::G
    precon::P
    rtol::R
    itmax::Int
    yreal::X
    yimag::X
    xreal::X
    ximag::X

    function GramInverse(G; rtol=eltype(G)(eps(eltype(G)) * 1e4), itmax=1000)
        precon = ilu(G)
        yreal = zeros(eltype(G), size(G, 1))
        yimag = zeros(eltype(G), size(G, 1))
        xreal = zeros(eltype(G), size(G, 2))
        ximag = zeros(eltype(G), size(G, 2))
        return new{eltype(G),typeof(G),typeof(precon),typeof(rtol),typeof(xreal)}(
            G, precon, rtol, itmax, yreal, yimag, xreal, ximag
        )
    end
end

function Base.size(G::GramInverse)
    return size(G.gram)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, G::GramInverse, x::AbstractVector)
    G.xreal .= real.(x)
    G.ximag .= imag.(x)

    G.yreal .= Krylov.gmres(
        G.gram,
        G.xreal;
        M=G.precon,
        ldiv=true,
        verbose=0,
        history=false,
        rtol=G.rtol,
        itmax=G.itmax,
    )[1]
    G.yimag .= Krylov.gmres(
        G.gram,
        G.ximag;
        M=G.precon,
        ldiv=true,
        verbose=0,
        history=false,
        rtol=G.rtol,
        itmax=G.itmax,
    )[1]

    map!(complex, y, G.yreal, G.yimag)
    return y
end
