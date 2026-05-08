# _interp.jl
#
# Helpers loaded by hmcode_py._bridge into a juliacall-owned module.
# Their job is to take numpy-shaped tabulated data and produce the
# Julia callables that HMcode.hmcode_power expects:
#   Pk_lin(k, z) -> Float64
#   sigma_R(R, z) -> Float64
#
# Conventions:
#   Python passes Pk_table with shape (nz, nk).  Julia receives it
#   transposed via the wrapper into shape (nk, nz) (column-major,
#   contiguous along k), but we accept either and reshape if needed.
#   Same story for sigma_table: Python (nz, nR) -> Julia (nR, nz).

using Interpolations

"""
Build a Pk_lin(k, z) callable from a tabulated (k, z, Pk_table) grid.

`Pk_table` must be indexable as `Pk_table[ik, iz]`, i.e. shape (nk, nz)
in Julia (the wrapper transposes once before calling this).  All values
must be strictly positive (we interpolate in log-log).
"""
function build_Pk_lin_interp(k_grid::AbstractVector{<:Real},
                             z_grid::AbstractVector{<:Real},
                             Pk_table::AbstractMatrix{<:Real})
    nk, nz = size(Pk_table)
    length(k_grid) == nk || throw(ArgumentError("Pk_table first axis must match length(k_grid)"))
    length(z_grid) == nz || throw(ArgumentError("Pk_table second axis must match length(z_grid)"))
    minimum(Pk_table) > 0 || throw(ArgumentError("Pk_table must be strictly positive (log-interpolated)"))

    logk = collect(Float64.(log.(k_grid)))
    z_vec = collect(Float64.(z_grid))
    logP = log.(Float64.(Pk_table))

    # One log-log interpolant per redshift slice (matches the test_power_spectrum.jl pattern).
    itps = Vector{Any}(undef, nz)
    @inbounds for iz in 1:nz
        itp = interpolate((logk,), view(logP, :, iz), Gridded(Linear()))
        itps[iz] = extrapolate(itp, Line())
    end

    # nearest-z dispatch: HMcode loops over the user-supplied zs, and
    # hmcode_power asks for Pk_lin at the *same* z values, so nearest
    # works exactly when z_grid == zs.
    nearest = function(zval::Real)
        ibest = 1
        dbest = abs(z_vec[1] - zval)
        @inbounds for i in 2:nz
            d = abs(z_vec[i] - zval)
            if d < dbest
                dbest = d
                ibest = i
            end
        end
        return ibest
    end

    Pk_lin = function (kval::Real, zval::Real)
        iz = nearest(zval)
        return exp(itps[iz](log(kval)))
    end
    return Pk_lin
end

"""
Build a sigma_R(R, z) callable from a tabulated (R, z, sigma_table) grid.

`sigma_table` must be indexable as `sigma_table[iR, iz]`, i.e. shape (nR, nz)
in Julia. All values must be strictly positive (log-log interpolation).
"""
function build_sigma_R_interp(R_grid::AbstractVector{<:Real},
                              z_grid::AbstractVector{<:Real},
                              sigma_table::AbstractMatrix{<:Real})
    nR, nz = size(sigma_table)
    length(R_grid) == nR || throw(ArgumentError("sigma_table first axis must match length(R_grid)"))
    length(z_grid) == nz || throw(ArgumentError("sigma_table second axis must match length(z_grid)"))
    minimum(sigma_table) > 0 || throw(ArgumentError("sigma_table must be strictly positive"))

    logR = collect(Float64.(log.(R_grid)))
    z_vec = collect(Float64.(z_grid))
    logS = log.(Float64.(sigma_table))

    itps = Vector{Any}(undef, nz)
    @inbounds for iz in 1:nz
        itp = interpolate((logR,), view(logS, :, iz), Gridded(Linear()))
        itps[iz] = extrapolate(itp, Line())
    end

    nearest = function(zval::Real)
        ibest = 1
        dbest = abs(z_vec[1] - zval)
        @inbounds for i in 2:nz
            d = abs(z_vec[i] - zval)
            if d < dbest
                dbest = d
                ibest = i
            end
        end
        return ibest
    end

    sigma_R = function (Rval::Real, zval::Real)
        iz = nearest(zval)
        return exp(itps[iz](log(Rval)))
    end
    return sigma_R
end
