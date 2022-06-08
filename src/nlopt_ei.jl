using NLopt
"""
objective function, NamedTuple of ensembles, initial values
"""
function maximize_ei(savedir, ensembles, data, idz, idz_all; objective=z->z[1])
    Z = map(E->copy(E[1].W[1]),ensembles)
    disciplines = keys(ensembles)
    m = length(data[disciplines[1]].Z)
    
    # Initial guesses and best point
    z = zeros(sum(length(idz_all)))
    EIc = zeros(m)
    ite = zeros(Int64,m)
    maxZ = [copy(z) for _=1:m]
    best = 0.
    nid = similar(z)
    for i = 1:m
        gfs = 1
        nid .*= 0.
        z .= 0.
        # Set z to average zstar from disciplines
        for d=disciplines
            for v=idz[d]
                z[idz_all[v]] += data[d].Zs[i][idz[d][v]]
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

        maxZ[i] .= z
    end

    # Maximization function
    function maximize(initial_z)
        function eic(z)
            for (zz, izz) = zip(Z,idz)
                getvar!(zz,z,izz, idz_all)
            end
            p = map((E,z_) -> HouseholderNets.predict(z,E,z_), ensembles, Z)
            return (objective(z)-best)*prod(p)#*exp(-10*(z-initial_z)⋅(z-initial_z))
        end
        n = length(initial_z)
        opt = Opt(:LD_MMA, 2)
        opt.lower_bounds = zeros(n)
        opt.upper_bounds = ones(n)
        opt.xtol_rel = 1e-4
        opt.max_objective = myfunc
        (maxf,maxz,ret) = optimize(opt, initial_z)
        return maxf, maxz, opt.numevals
    end

    # Maximize EIc
    for i = 1:m
        z, EIc[i], ite[i] = maximize(maxZ[i])
        maxZ[i] .= z
    end

    # Save EIc found
    path = "$savedir/eic.jld2"
    save(path, "EIc", EIc, "maxZ", maxZ, "ite", ite, "best", best)
    
    
    max_eic, imax = findmax(EIc)
    return maxZ[imax], max_eic
end