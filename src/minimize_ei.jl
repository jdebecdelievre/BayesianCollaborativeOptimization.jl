
"""
objective function, NamedTuple of ensembles, initial values
"""
function minimize_ei(savedir, ensembles, data, idz, idz_all)
    Z = map(E->copy(E[1].W[1]),ensembles)
    disciplines = keys(ensembles)
    m = length(data[disciplines[1]].Z)
    
    # Initial guesses and best point
    z = zeros(sum(length(idz_all)))
    EIc = zeros(m)
    ite = zeros(Int64,m)
    minZ = [copy(z) for _=1:m]
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
            # setvar!(z, data[d].Z[i], idz_all, idz[d])
            gfs *= data[d].fsb[i]
        end
        z ./= nid
        
        # Find best point so far
        if (gfs == 1) && (-z[1]<best)
            best = -z[1]
        end

        minZ[i] .= z
    end

    # Minimization function
    function minimize(initial_z)
        function eic(z)
            for (zz, izz) = zip(Z,idz)
                getvar!(zz,z,izz, idz_all)
            end
            p = map((E,z_) -> HouseholderNets.predict(z,E,z_), ensembles, Z)
            return (-best-z[1])*prod(p)*exp(-10*(z-initial_z)⋅(z-initial_z))
            # return (-best-z[1])*(prod(p))^(1/length(p))#*exp(-100*(z-initial_z)⋅(z-initial_z))
        end
        n = length(initial_z)
        lower = zeros(n)
        upper =  ones(n)
        results = Evolutionary.optimize(eic, BoxConstraints(lower, upper), initial_z, CMAES(μ=10,sigma0=.05), Evolutionary.Options(iterations=5000))
        return Evolutionary.minimizer(results), Evolutionary.minimum(results), Evolutionary.iterations(results)
    end

    # Minimize neg EIc
    for i = 1:m
        z, EIc[i], ite[i] = minimize(minZ[i])
        minZ[i] .= z
    end

    # Save EIc found
    path = "$savedir/eic.jld2"
    save(path, "EIc", EIc, "minZ", minZ, "ite", ite, "best", best)
    
    
    min_eic, imin = findmin(EIc)
    return minZ[imin], min_eic
end