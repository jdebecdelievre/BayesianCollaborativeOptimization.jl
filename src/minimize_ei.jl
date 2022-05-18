using LinearAlgebra
"""
objective function, NamedTuple of ensembles, initial values
"""
function minimize_ei(savedir, ensembles, data, idz, idz_all)
    mkpath(savedir)
    Z = map(E->copy(E[1].W[1]),ensembles)
    Z_ = map(E->copy(E[1].W[1]),ensembles)
    disciplines = keys(ensembles)
    m = length(data[disciplines[1]].Z)
    
    # Initial guesses and best point
    zopt = [0.6854243695443947
    0.33607571942239045
    0.7451928938495387
    0.29273370333643656
    0.7280058625813217]
    m = min(5,m)
    z = zeros(len(idz_all))
    EIc = zeros(m+1)
    ite = zeros(Int64,m+1)
    minZ = [[copy(z) for _=1:m];[zopt]]
    iniZ = [[copy(z) for _=1:m];[zopt]]
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
            # setvar!(z, data[d].Z[i], idz_all, idz[d])
            gfs *= data[d].fsb[i]
        end
        z ./= nid
        
        # Find best point so far
        if (gfs == 1) && (-z[1]<best)
            best = -z[1]
        end

        iniZ[i] .= z
    end

    # Minimization function
    function minimize(initial_z)
        function eic(z)
            for (zz, izz) = zip(Z,idz)
                getvar!(zz,z,izz, idz_all)
            end
            p = map((z,E,z_) -> HouseholderNets.predict(z,E,z_), Z, ensembles, Z_)
            violation = map((d,z_)-> minimum(norm(d.Z[i] - z_)-d.sqJ[i] for i=eachindex(d.Z))>0,data,Z_)
            return (-best-z[1])*prod(p)*prod(violation)#*exp(-10*(z-initial_z)⋅(z-initial_z))
        end
        n = length(initial_z)
        lower = zeros(n)
        upper =  ones(n)
        results = Evolutionary.optimize(eic, BoxConstraints(lower, upper), initial_z, CMAES(μ=50,sigma0=.2), Evolutionary.Options(iterations=5000))
        return Evolutionary.minimizer(results), Evolutionary.minimum(results), Evolutionary.iterations(results)
    end

    # Minimize neg EIc
    for i = eachindex(iniZ)
        z, EIc[i], ite[i] = minimize(iniZ[i])
        minZ[i] .= z
    end

    # Save EIc found
    path = "$savedir/eic.jld2"
    @save path EIc minZ iniZ ite best
    
    min_eic, imin = findmin(EIc)
    return minZ[imin], min_eic
end