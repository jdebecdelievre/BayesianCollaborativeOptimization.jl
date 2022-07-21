"""
Cache for EIC calculation:
    Zd: NamedTuple{disciplines} of input vectors
    h: NamedTuple{disciplines} of output vectors (of each net in ensemble)
    dhdx: NamedTuple{disciplines} of vector gradient vectors for each net in ensemble
    dp: NamedTuple{disciplines} of gradient of ensemble
    WZ: NamedTuple{disciplines} of vectors useful in backprop
"""
function eic_cache(ensembles::NamedTuple{dis}) where dis #,NTuple{nd,Vector{HouseholderNet{L,Sn,TF}}}} where {nd, dis,L,Sn,TF})
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
problem: problem to solve
stepsize: scalar
ensembles: NamedTuple{disciplines} of ensembles of HouseholderNets
best: best objective so far
idz: IndexMap to populate inputs to ensembles
cache: result of EIC cache
"""
function eic(z::V, grad::V, z0::V, problem::AbstractProblem,
            tol::Float64, stepsize::Float64, 
            ensembles::NamedTuple{disciplines},#,NTuple{nd, Vector{HouseholderNet{L,Sn,TF}}}} where {nd,L,Sn,TF}, 
            best::Float64, cache::NamedTuple=eic_cache(ensembles)) where {V<:AbstractVector,disciplines}
    # get preallocated buffers
    (; Zd, h, dhdx, dp, WZ) = cache
    idz = indexmap(problem)

    # determine whether to compute the gradient
    g = length(grad) > 0
    
    # improvement (objective(z)-best) 
    obj = g ? objective(problem, z, grad) : objective(problem, z)
    ei     = (obj-best)
    val    = 1. #(ei > 0) dropping the max(0., ...) for now
    ei    *= -val
    g && (grad .*= -val)

    # probability
    for d=disciplines
        # Assign Zd
        Zd[d] .= view(z,idz[d])
        
        # Compute vector and gradient
        p = predict_grad!(dp[d],Zd[d], ensembles[d], tol, WZ=WZ[d], dhdx=dhdx[d], h=h[d])
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

"""
objective function, NamedTuple of ensembles, initial values
"""
function maximize_ei(solver::BCO, data::NamedTuple{disciplines}, ensembles::NamedTuple{disciplines}, savedir::String) where disciplines
    m   = length(data[disciplines[1]].Z)
    Nz  = number_shared_variables(solver.problem)
    nd = length(disciplines)
    idz = indexmap(solver.problem)
    to  = solver.to
    tol = solver.tol
    
    ## Initial guesses and best point
    z = zeros(Nz)
    EIc = zeros(m*nd)
    ite = zeros(Int64,m*nd)
    maxZ = [copy(z) for _=1:m*nd]
    iniZ = [copy(z) for _=1:m*nd]
    best = objective_lowerbound(solver.problem)
    nid = copy(z)
    j = 1
    for i = 1:m
        # Compute average zstar from disciplines
        nid .= 0.
        z .= 0.
        for d=disciplines
            z[idz[d]]  .+= data[d].Zs[i]
            nid[idz[d]] .+= 1.
        end
        @. z = z / nid
        
        # Add one initial value per discipline
        for d=disciplines
            @. iniZ[j] = z
            @. iniZ[j][idz[d]] = data[d].Zs[i]
            j+=1
        end
        
        # Find best point so far
        gfs = 1
        for d=disciplines
            z[idz[d]] .= data[d].Z[i]
            gfs *= data[d].fsb[i]
        end
        obj = objective(solver.problem, z)
        if (gfs == 1) && (obj>best)
            best = obj
        end
    end

    ## Constrained Expected Improvement EIc function
    # Preallocations
    cache = eic_cache(ensembles)
    
    ## Initialize SLSQP
    tol = 0.
    slsqp = Opt(:LD_SLSQP, Nz)
    slsqp.lower_bounds = zeros(Nz)
    slsqp.upper_bounds = ones(Nz)
    slsqp.xtol_rel = 1e-7
    slsqp.maxeval = 2500
    
    ## Run maximization of EIc for each initial guess
    stepsize = ones(m*nd) * solver.stepsize
    msg = Vector{Symbol}(undef,m*nd)
    for i = 1:length(EIc)
        # Cast to strict bounds before using SLSQP
        @. iniZ[i] = max(iniZ[i], eps())
        @. iniZ[i] = min(iniZ[i], 1-eps())
        
        # Use SLSQP
        @timeit to "slsqp" begin
            slsqp.min_objective = (z,grad) -> eic(z, grad, iniZ[i], solver.problem, tol, stepsize[i], ensembles, best, cache)
            maxZ[i] .= iniZ[i]
            _, z, msg[i] = optimize(slsqp, maxZ[i])
            maxZ[i] .= z
            EIc[i] = -eic(maxZ[i], Float64[], iniZ[i], solver.problem, tol, 1000., ensembles, best, cache)
            ite[i] = slsqp.numevals
        end
    end

    # Save EIc found
    path = "$savedir/eic.jld2"
    save(path, "EIc", EIc, "maxZ", maxZ, "iniZ", iniZ, "ite", ite, 
                "best", best, "stepsize", stepsize, "msg",msg)
    
    # Return best 
    max_eic, imax = findmax(EIc)
    z = maxZ[imax]

    ## Also use MLSL
     @timeit to "mlsl" begin
        z0 = copy(z)
    
        gopt = Opt(:G_MLSL_LDS, Nz)
        local_optimizer!(gopt, slsqp)
        population!(gopt, 5000)
        gopt.lower_bounds = lower_bounds(slsqp)
        gopt.upper_bounds = upper_bounds(slsqp)
        gopt.xtol_rel = 1e-7
        gopt.maxeval = 2500
        gopt.min_objective = (z,grad) -> eic(z, grad, z0, solver.problem, tol, stepsize[imax], ensembles, best, cache)
        (maxf,maxz,ret) = optimize(gopt, z0)
        eic_mlsl = -eic(maxz, Float64[], z0, solver.problem, tol, 1000., ensembles, best, cache)
    end

    if eic_mlsl > max_eic
        z = maxz
        max_eic = eic_mlsl
    end

    return z, max_eic
end