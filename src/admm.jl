"""
Alternating Direction Method of Multipliers for global consensus, parallel version.
See ref [1], in Section 7.1:

Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
Foundations and Trends in Machine Learning Vol. 3, No. 1 (2010) 1–122
2011 S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein
DOI: 10.1561/2200000016
https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf 

"""
struct ADMM{problem,disciplines,ndisciplines} <: AbstractSolver{problem}
    problem::problem
    ρ::Float64
    y::NamedTuple{disciplines,NTuple{ndisciplines,Vector{Float64}}} # Lagrange Multipliers
    yobj::Vector{Float64}
    to::TimerOutput
    function ADMM(problem::pb; ρ::Float64=1.) where {pb<:AbstractProblem}
        disc = discipline_names(problem)
        idz  = indexmap(problem)
        Nz   = number_shared_variables(problem)
        y    = map(id->zeros(length(id)), idz)
        yobj = zeros(Nz)
        to = TimerOutput()
        return new{pb, disc,length(disc)}(problem, ρ, y, yobj, to)
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
function get_new_point(ite::Int64, solver::ADMM,
                        data::NamedTuple{disciplines}, 
                        savedir::String) where disciplines
    (; y, yobj, ρ, problem) = solver
    idz = indexmap(problem)
    Nz  = length(yobj)
    nid = zeros(Nz)
    
    if ite==1
        map(iy->(iy .= 0.), y)
        yobj .= 0.
    end

    # Find index of current iteration in data
    k = map(d->findprev(==(ite-1), d.ite, length(d.ite)), data)
    
    # Recover current global point from other averages
    nid .= 0.
    z_o = zeros(Nz)
    for d=disciplines
        z_o[idz[d]] .+= data[d].Z[k[d]]
        nid[idz[d]] .+= 1.
    end
    @. z_o = z_o / nid - (nid+1)/nid * yobj / ρ
    
    # Solve prox problem for obj (all other prox are projections, done in main solve function)
    zs_o = prox(problem, z_o, ρ)

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

    # Evaluate target for each discipline
    Zd = map((id,iy)->z̄s[id] - iy/ρ, idz, y)
    @. z̄s -= yobj / ρ
    
    return z̄s, Zd, 0.
end

function prox(problem::AbstractProblem, z_o::Vector{Float64}, ρ::Float64)
    # solve proximal problem for objective function min_z f(z) + ρ(z-z0)ᵀ(z-z0) /2
    function fun(z, grad)
        dz = (z-z_o)
        o = -objective(problem, z, grad)
        grad .*= -1
        grad .+= ρ * dz
        return o + ρ/2 * (dz ⋅ dz)
    end

    z0 = max.(z_o,0.)
    @. z0 = min(z0,1.)

    # Use SLSQP with NLOPT
    n = length(z0)
    opt = Opt(:LD_SLSQP, n)
    opt.lower_bounds = zeros(n)
    opt.upper_bounds = ones(n)
    opt.xtol_rel = 1e-7
    opt.maxeval = 2500
    opt.min_objective = fun
    (maxf,maxz,ret) = optimize(opt, z0)
    
    # @show ret
    grad = zeros(n)
    fun(maxz, grad)
    # @show grad
    return maxz
end