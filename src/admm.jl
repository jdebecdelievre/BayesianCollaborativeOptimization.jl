"""
Alternating Direction Method of Multipliers, parallel version.
See ref [1] Proximal algorithms, N. Parikh and S. Boyd, 2014
https://stanford.edu/~boyd/papers/pdf/prox_algs.pdf
"""
struct ADMM{disciplines,ndisciplines} <: AbstractSolver
    ρ::Float64
    y::NamedTuple{disciplines,NTuple{ndisciplines,Vector{Float64}}} # Lagrange Multipliers
    yobj::Vector{Float64}
    function ADMM(idz::IndexMap{disciplines}; ρ=1.) where disciplines
        y = map(id->zeros(length(id)), idz)
        Nz = maximum(map(maximum,idz))
        yobj = zeros(Nz)
        return new{disciplines,length(disciplines)}(ρ, y, yobj)
    end
end

"""
At iteration k:
zd[k+1] = zs_d[k] + (zd[k] - z̄s[k])
where:
- zd is the input to discipline d
- zs_d is the projection of zd by discipline d 
- z̄s is the average of zs_d over disciplines
"""
function get_new_point(ite::Int64, solver::ADMM, objective,
                        data::NamedTuple{disciplines}, 
                        idz::IndexMap{disciplines}, 
                        savedir::String) where disciplines
    (; y, yobj, ρ) = solver
    Nz  = length(yobj)
    nid = zeros(Nz)

    # Find index of current iteration in data
    k = map(d->findprev(==(ite-1), d.ite, length(d.ite)), data)
    
    # Recover current global point
    z_o = zeros(Nz)
    for d=disciplines
        z_o[idz[d]] .= data[d].Z[k[d]]
    end
    
    # Solve prox problem for obj (all other prox are projections, done in main solve function)
    zs_o = prox(objective, z_o, yobj, ρ)

    # Perform z update (average)
    z̄s   = copy(zs_o)
    nid .= 1
    for d=disciplines
        z̄s[idz[d]]  .+= data[d].Zs[k[d]]
        nid[idz[d]] .+= 1.
    end
    @. z̄s = z̄s / nid

    # Perform y update
    yobj .+= ρ * (zs_o - z̄s)
    for d = disciplines
        y[d] .+= ρ * (data[d].Zs[k[d]] - z̄s[idz[d]])
    end
    return z̄s, 0.
end

function prox(objective, z0::Vector{Float64}, y::Vector{Float64}, ρ::Float64)
    # solve proximal problem for objective function min_z f(z) + ρ(z-z0)ᵀ(z-z0) /2
    function fun(z, grad)
        o = -objective(z,grad) + y⋅z
        grad .*= -1
        grad += ρ * (z-z0) + y
        return o + ρ/2 * ((z-z0) ⋅ (z-z0))
    end

    # Use SLSQP with NLOPT
    n = length(z0)
    opt = Opt(:LD_SLSQP, n)
    opt.lower_bounds = zeros(n)
    opt.upper_bounds = ones(n)
    opt.xtol_rel = 1e-7
    opt.maxeval = 2500
    opt.min_objective = fun
    (maxf,maxz,ret) = optimize(opt, copy(z0))

    return maxz
end