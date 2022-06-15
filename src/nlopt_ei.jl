using NLopt
using FiniteDiff
"""
objective function, NamedTuple of ensembles, initial values
"""
function maximize_ei(savedir, ensembles, data, idz, idz_all, options; objective=default_obj)
    disciplines = keys(ensembles)
    m = length(data[disciplines[1]].Z)
    
    # Initial guesses and best point
    z = zeros(sum(length(idz_all)))
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
            for v=keys(idz[d])
                z[idz_all[v]] += data[d].Zs[i][idz[d][v]]
                nid[idz_all[v]] += 1.
            end
        end
        @. iniZ[i] = z / nid
        
        # Find best point so far
        gfs = 1
        nid .= 0.
        z .= 0.
        for d=disciplines
            for v=keys(idz[d])
                z[idz_all[v]] += data[d].Z[i][idz[d][v]]
                nid[idz_all[v]] += 1.
            end
            gfs *= data[d].fsb[i]
        end
        z ./= nid
        obj = objective(z)
        if (gfs == 1) && (obj>best)
            best = obj
        end
    end

    # Maximization function
    Z    = map(e->copy(e[1].W[1]),ensembles)
    dp   = deepcopy(Z)
    WZ   = map(e->zeros(length(e[1].W)-1), ensembles)
    dhdx = map(e->[copy(net.W[1]) for net=e], ensembles)
    h    = map(e->zeros(length(e)), ensembles)

    function eic(z, grad, z0, stepsize)
        g = length(grad) > 0
        # improvement
        obj = g ? objective(z, grad) : objective(z)
        ei     = (obj-best)
        val    = 1. #(ei > 0) dropping the max(0., ...) for now
        ei    *= -val
        g && (grad .*= -val)

        # probability
        for d=disciplines
            # Assign Z
            getvar!(Z[d],z,idz[d], idz_all)

            # Compute vector and gradient
            p = predict_grad!(dp[d],Z[d], ensembles[d], WZ=WZ[d], dhdx=dhdx[d], h=h[d])
            if g
                # @. grad = grad*p + ei * dp[d]
                grad .*= p
                for v=keys(idz[d])
                    grad[idz_all[v]] = ei * dp[d][idz[d][v]]
                end
            end 
            ei *= p
        end

        # Localize search(objective(z)-best)
        loc = exp(-(z-z0)⋅(z-z0)/stepsize^2)
        if g
            @. grad = grad*loc - (ei * 1/stepsize^2 * 2 * loc) * (z-z0)
        end
        ei *= loc

        return ei
    end

    function eic(z,z0,stepsize)
        for (zz, izz) = zip(Z,idz)
            getvar!(zz,z,izz, idz_all)
        end
        p = map((z,E,z_) -> HouseholderNets.predict(z,E,z_), Z, ensembles, dp)
        return -max(0., objective(z)-best)*prod(p)*exp(-1/stepsize^2*(z-z0)⋅(z-z0))
    end

    # @show eic(iniZ[1], iniZ[1],.1)
    # J = FiniteDiff.finite_difference_gradient(z->eic(z,[],iniZ[1],.1), iniZ[1])
    # @show J
    # @show eic(iniZ[1], J, iniZ[1],.1)
    # @show J
    # @show best

    function maximize(z0, stepsize)
        n = length(z0)
        lower = zeros(n)
        upper =  ones(n)

        # Start with CMA if local gradient is too small
        eic(z0, z, z0,stepsize) 
        if norm(z) < 1e-6
            results = Evolutionary.optimize(z -> eic(z,z0,stepsize), BoxConstraints(lower, upper), z0, CMAES(μ=50,sigma0=1.), Evolutionary.Options(iterations=5000))
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
        opt.min_objective = (z,grad) -> eic(z,grad,z0,stepsize)
        (maxf,maxz,ret) = optimize(opt, z0)
        numevals = opt.numevals
        return -maxf, maxz, numevals, ret
    end
    
    # Maximize EIc
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
    
    
    max_eic, imax = findmax(EIc)
    return maxZ[imax], max_eic
end