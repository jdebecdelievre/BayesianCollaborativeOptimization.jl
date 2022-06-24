"""
Cache for EIC calculation:
    Zd: NamedTuple{disciplines} of input vectors
    h: NamedTuple{disciplines} of output vectors (of each net in ensemble)
    dhdx: NamedTuple{disciplines} of vector gradient vectors for each net in ensemble
    dp: NamedTuple{disciplines} of gradient of ensemble
    WZ: NamedTuple{disciplines} of vectors useful in backprop
"""
function eic_cache(ensembles::NamedTuple{dis,NTuple{nd,Vector{HouseholderNet{L,Sn,TF}}}} where {nd, dis,L,Sn,TF})
    cache = (;  Zd   = map(e->copy(e[1].W[1]),ensembles),
                dp   = map(e->copy(e[1].W[1]),ensembles), # gradient of each ensemble
                h    = map(e->zeros(length(e)), ensembles), # output of each network for each ensemble
                dhdx = map(e->[copy(net.W[1]) for net=e], ensembles), # gradient of each network for each ensemble
                WZ   = map(e->zeros(length(e[1].W)-1), ensembles) # helper buffer for backprop
            )
    return cache
end


"""
EIc with gradient calculation
z, grad, z0: vectors of size n
stepsize: scalar
objective: objective function
ensembles: NamedTuple{disciplines} of ensembles of HouseholderNets
best: best objective so far
idz: IndexMap to populate inputs to ensembles
cache: result of EIC cache
"""
function eic(z::V, grad::V, z0::V, 
            stepsize::Float64, objective, ensembles::NamedTuple{disciplines,NTuple{nd, Vector{HouseholderNet{L,Sn,TF}}}} where {nd,L,Sn,TF}, 
            best::Float64, idz::IndexMap{disciplines}, cache::NamedTuple=eic_cache(ensembles)) where {V<:AbstractVector,disciplines}
    # get preallocated buffers
    (; Zd, h, dhdx, dp, WZ) = cache

    # determine whether to compute the gradient
    g = length(grad) > 0
    
    # improvement (objective(z)-best) 
    obj = g ? objective(z, grad) : objective(z)
    ei     = (obj-best)
    val    = 1. #(ei > 0) dropping the max(0., ...) for now
    ei    *= -val
    g && (grad .*= -val)

    # probability
    for d=disciplines
        # Assign Zd
        Zd[d] .= view(z,idz[d])
        
        # Compute vector and gradient
        p = predict_grad!(dp[d],Zd[d], ensembles[d], WZ=WZ[d], dhdx=dhdx[d], h=h[d])
        if g
            # @. grad = grad*p + ei * dp[d]
            grad[idz[d]] .*= p
            grad[idz[d]] .+= ei * dp[d]
        end
        ei *= p
    end

    # Localize search
    loc = exp(-(z-z0)⋅(z-z0)/stepsize^2)
    if g
        @. grad = grad*loc - (ei * 1/stepsize^2 * 2 * loc) * (z-z0)
    end
    ei *= loc

    return ei
end

# EIc without gradient calculation
function eic(z,z0,stepsize, objective, ensembles::NamedTuple{disciplines}, best, idz, cache=eic_cache(ensembles)) where disciplines
    (; Zd, dp) = cache
    for d=disciplines
        Zd[d] .=  view(z,idz[d])
    end
    p = map((z,E,z_) -> HouseholderNets.predict(z,E,z_), Zd, ensembles, dp)
    return -(objective(z)-best)*prod(p)*exp(-1/stepsize^2*(z-z0)⋅(z-z0))
end

"""
objective function, NamedTuple of ensembles, initial values
"""
function maximize_ei(savedir::String, ensembles::NamedTuple{disciplines}, 
                    data::NamedTuple{disciplines}, idz::NamedTuple{disciplines}, 
                    objective, options::BCO) where disciplines
    m = length(data[disciplines[1]].Z)
    Nz = maximum(map(maximum,idz))
    
    ## Initial guesses and best point
    z = zeros(Nz)
    EIc = zeros(m)
    ite = zeros(Int64,m)
    maxZ = [copy(z) for _=1:m]
    iniZ = [copy(z) for _=1:m]
    best = 0.
    nid = copy(z)
    for i = 1:m
        # Set z to average zstar from disciplines
        nid .= 0.
        z .= 0.
        for d=disciplines
            z[idz[d]]  .+= data[d].Zs[i]
            nid[idz[d]] .+= 1.
        end
        @. iniZ[i] = z / nid
        
        # Find best point so far
        gfs = 1
        nid .= 0.
        z .= 0.
        for d=disciplines
            z[idz[d]]  .+= data[d].Z[i]
            nid[idz[d]] .+= 1.
            gfs *= data[d].fsb[i]
        end
        z ./= nid
        obj = objective(z)
        if (gfs == 1) && (obj>best)
            best = obj
        end
    end

    ## Constrained Expected Improvement EIc function
    # Preallocations
    cache = eic_cache(ensembles)

    ## Maximization function
    function maximize(z0, stepsize)
        n = length(z0)
        lower = zeros(n)
        upper =  ones(n)

        # Start with CMA if local gradient is too small
        if norm(z) < 1e-6
            results = Evolutionary.optimize(z -> eic(z,copy(z0),stepsize, ensembles, best, idz, cache), 
                        BoxConstraints(lower, upper), z0, 
                        CMAES(μ=50,sigma0=1.), Evolutionary.Options(iterations=5000))
            maxz, maxf, numevals = Evolutionary.minimizer(results), Evolutionary.minimum(results), Evolutionary.iterations(results)
            z0 .= maxz
        end
        
        # Cast to strict bounds before using SLSQP
        @. z0 = max(z0, eps())
        @. z0 = min(z0, 1-eps())

        # Use SLSQP
        opt = Opt(:LD_SLSQP, n)
        opt.lower_bounds = lower
        opt.upper_bounds = upper
        opt.xtol_rel = 1e-7
        opt.maxeval = 2500
        opt.min_objective = (z,grad) -> eic(z, grad, copy(z0), stepsize, objective, ensembles, best, idz, cache)
        (maxf,maxz,ret) = optimize(opt, z0)
        numevals = opt.numevals
        return -maxf, maxz, numevals, ret
    end
    
    ## Run maximization of EIc for each initial guess
    stepsize = ones(m) * options.stepsize
    msg = Vector{Symbol}(undef,m)
    for i = 1:m
        EIc[i], z, ite[i], msg[i] = maximize(iniZ[i],stepsize[i])
        if EIc[i] < 1e-4
            stepsize[i] *= 2
            EIc[i], z, ite[i], msg[i] = maximize(iniZ[i],stepsize[i])
        end
        maxZ[i] .= z
    end

    # Save EIc found
    path = "$savedir/eic.jld2"
    save(path, "EIc", EIc, "maxZ", maxZ, "iniZ", 
                iniZ, "ite", ite, "best", best, "stepsize", stepsize,
                "msg",msg)
    
    # Return best 
    max_eic, imax = findmax(EIc)
    z = maxZ[imax]

    return z, max_eic
end