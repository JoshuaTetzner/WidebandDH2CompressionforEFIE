using SphericalScattering
using LinearAlgebra

const c = 2.99792458e8          # speed of light
const μ = 4π * 1e-7             # permeability
const ε = 8.8541878176e-12      # permittivity

f = typeof(λ)(c / λ)           # Frequency
k = typeof(λ)(2 * π / λ)       # Wavenumber
ω = typeof(λ)(c * k)           # Angular frequency

function compareManufacturedWorstError(excitation::Excitation, bsf, u; NFdist=0.0, field=:E)
    f = excitation.frequency

    Rng = getIncident(excitation; NFdist=NFdist, type=field)
    MoM = getPotential(f, bsf, u; NFdist=NFdist, type=field)

    diff = norm.(MoM - Rng) ./ maximum(norm.(Rng))

    return maximum(abs.(diff))
end

function getIncident(excitation::Excitation; NFdist=0.0, type=:E)

    # ------------------- FF
    if NFdist == 0.0
        points_cart, points_sph = sphericalGridPoints()
        F_cart = field(excitation, SphericalScattering.FarField(points_cart))
        F_sph = SphericalScattering.convertCartesian2Spherical.(F_cart, points_sph)

        return reshape(F_sph, size(points_cart))

        # ------------------- NF
    else
        points_cart, points_sph = sphericalGridPoints(; r=NFdist)

        if type == :E
            F_cart = -field(excitation, ElectricField(points_cart))
        elseif type == :H
            F_cart = -field(excitation, MagneticField(points_cart))
        end

        F_sph = SphericalScattering.convertCartesian2Spherical.(F_cart, points_sph)

        return reshape(F_sph, size(points_cart))
    end
end

function getPotential(f, bsf, u; NFdist=0.0, type=:E)
    λ = c / f       # Wavelength
    k = 2 * π / λ   # Wavenumber

    # ------------------- FF
    if NFdist == 0.0

        # --- get points where the FF ist computed
        points_cart, points_sph = sphericalGridPoints()

        # --- Compute the MoM FF
        MoM_cart =
            -im * f / (2 * c) * potential(MWFarField3D(; gamma=im * k), points_cart, u, bsf)#, quadstrat=BEAST.SingleNumQStrat(3))          # MoM FF in cartesian vector components  !! rescale

        return SphericalScattering.convertCartesian2Spherical.(MoM_cart, points_sph)

        # ------------------- NF
    else
        # --- get points where the NF ist computed
        points_cart, points_sph = sphericalGridPoints(; r=NFdist)

        if type == :E
            MoM_cart = potential(MWSingleLayerField3D(; wavenumber=k), points_cart, u, bsf)

        elseif type == :H
            MoM_cart =
                1 / (c * μ) *
                potential(BEAST.MWDoubleLayerField3D(; wavenumber=k), points_cart, u, bsf)
        else
            error("field can only be :E or :H")
        end

        return SphericalScattering.convertCartesian2Spherical.(MoM_cart, points_sph)
    end
end
