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
    nid = similar(z)
    for i = 1:m
        gfs = 1
        nid .*= 0.
        z .= 0.
        # Set z to average zstar from disciplines
        for d=disciplines
            for v=keys(idz[d])
                z[idz_all[v]] += data[d].Zs[end-i+1][idz[d][v]]
                nid[idz_all[v]] += 1.
            end
            gfs *= data[d].fsb[i]
        end
        z ./= nid
        
        # Find best point so far
        obj = objective(z)
        if (gfs == 1) && (obj>best)
            best = obj
        end

        iniZ[i] .= z
    end

    # Maximization function
    Z    = map(e->copy(e[1].W[1]),ensembles)
    dp   = deepcopy(Z)
    WZ   = map(e->zeros(length(e[1].W)-1), ensembles)
    dhdx = map(e->[copy(net.W[1]) for net=e], ensembles)
    h    = map(e->zeros(length(e)), ensembles)

    function eic(z, grad, z0)
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
                @. grad = grad*p + ei * dp[d]
            end 
            ei *= p
        end

        # Localize search(objective(z)-best)
        loc = exp(-options.local_search_multiplier*(z-z0)⋅(z-z0))
        if g
            @. grad = grad*loc - (ei * options.local_search_multiplier * 2 * loc) * (z-z0)
        end
        ei *= loc

        return ei
    end

    function eic(z,z0)
        for (zz, izz) = zip(Z,idz)
            getvar!(zz,z,izz, idz_all)
        end
        p = map((z,E,z_) -> HouseholderNets.predict(z,E,z_), Z, ensembles, dp)
        return -max(0., objective(z)-best)*prod(p)*exp(-options.local_search_multiplier*(z-z0)⋅(z-z0))
    end

    @show eic(iniZ[1], iniZ[1])
    J = FiniteDiff.finite_difference_gradient(z->eic(z,[],iniZ[1]), iniZ[1])
    @show J
    @show eic(iniZ[1], J, iniZ[1])
    @show J

    function maximize(z0)
        n = length(z0)
        lower = zeros(n)
        upper =  ones(n)

        opt = Opt(:LD_SLSQP, n)
        opt.lower_bounds = lower
        opt.upper_bounds = upper
        # opt.xtol_rel = 1e-4
        opt.min_objective = (z,grad) -> eic(z,grad,z0)
        (maxf,maxz,ret) = optimize(opt, z0)
        numevals = opt.numevals
        # results = Evolutionary.optimize(z -> eic(z,z0), BoxConstraints(lower, upper), z0, CMAES(μ=50,sigma0=1.), Evolutionary.Options(iterations=5000))
        # maxz, maxf, numevals = Evolutionary.minimizer(results), Evolutionary.minimum(results), Evolutionary.iterations(results)
        @show numevals
        @show ret
        return -maxf, maxz, numevals
    end
    
    # Maximize EIc
    for i = 1:m
        EIc[i], z, ite[i] = maximize(iniZ[i])
        maxZ[i] .= z
    end

    # Save EIc found
    path = "$savedir/eic.jld2"
    save(path, "EIc", EIc, "maxZ", maxZ, "iniZ", iniZ, "ite", ite, "best", best)
    
    
    max_eic, imax = findmax(EIc)
    return maxZ[imax], max_eic
end