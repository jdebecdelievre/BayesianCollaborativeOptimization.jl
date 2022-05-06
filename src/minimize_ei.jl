
"""
objective function, NamedTuple of ensembles, initial values
"""
function minimize_ei(savedir, ensembles, data, idz, idz_all)
    Z = map(E->copy(E[1].W[1]),ensembles)
    
    # Minimization function
    function minimize(initial_z)
        function eic(z)
            for (zz, izz) = zip(Z,idz)
                getvar!(zz,z,izz, idz_all)
            end
            p = map((E,z_) -> HouseholderNets.predict(z,E,z_), ensembles, Z)
            return -z[1]*prod(p)
        end
        n = length(initial_z)
        lower = zeros(n)
        upper =  ones(n)
        results = Evolutionary.optimize(eic, BoxConstraints(lower, upper), initial_z, CMAES(), Evolutionary.Options(iterations=5000))
        return Evolutionary.minimizer(results), Evolutionary.minimum(results), Evolutionary.iterations(results)
    end

    # Initial guesses
    z = zeros(sum(length(idz_all)))
    disciplines = keys(ensembles)
    m = length(data[disciplines[1]].Z)
    EIc = zeros(m)
    ite = zeros(Int64,m)
    minZ = [copy(z) for _=1:m]
    for i = 1:m
        # Set z to values from disciplines
        for d=disciplines
            setvar!(z, data[d].Z[i], idz_all, idz[d])
        end
        # Minimize neg EIc
        z, EIc[i], ite[i] = minimize(z)
        minZ[i] .= z
    end

    # Save EIc found
    path = "$savedir/eic.jld2"
    save(path, "EIc", EIc, "minZ", minZ, "ite", ite)
    
    
    min_eic, imin = findmin(EIc)
    return minZ[imin], min_eic
end