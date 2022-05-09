"""
Receives 1 folder name, and 2 NamedTuples.
"""
function bco(savedir, variables, subspace_fun)
    @assert keys(variables) == keys(subspace_fun) "$(keys(variables)) =/= $(keys(subspace_fun))"

    n_ite = 30 # number of iterations
    ini_samples = 2 # number of initial random samples. 0 to use provided z0
    iteration_restart = 0 # Iteration Number at which to restart.
    warm_start_sampler = 0
    tol = 5e-3

    # create xp directory
    mkpath(savedir)

    # Create list of global vars
    disciplines = keys(variables)
    variables_all = mergevar(values(variables)...)
    Nz = len(variables_all)
    idz = map(indexbyname, variables)
    idz_all = indexbyname(variables_all)

    # Create sampler
    Sz = SobolSeq(Nz-1)
    for _=1:warm_start_sampler
        next!(Sz)
    end

    # Initial samples and ensemble
    if iteration_restart < 1
        # Create data storage 
        data = NamedTuple{disciplines}(Tuple((; 
            Z   = [zeros(len(variables[name])) for _=1:(ini_samples)], 
            Zs  = [zeros(len(variables[name])) for _=1:(ini_samples)], 
            sqJ = zeros(ini_samples),
            fsb = BitVector(zeros(ini_samples)),
            ite = (zeros(Int64,ini_samples)))
        for name = disciplines))

        # get ini samples
        if ini_samples == 0
            Z = [ini_scaled(variables_all)]
        else
            Z = [[1.; next!(Sz)] for _=1:ini_samples]
        end

        # Evaluate subspaces
        for d=disciplines
            D = data[d]
            for (i,z)=enumerate(Z)
                getvar!(D.Z[i], z, idz[d], idz_all)
                D.Zs[i] .= subspace_fun[d](D.Z[i])
                D.sqJ[i] = norm(D.Z[i] .- D.Zs[i])
                D.fsb[i] = (D.sqJ[i]<tol)
            end
            save_data("$savedir/$d.jld2",D)
        end
    else
        # Load data
        data = NamedTuple{disciplines}(map(d->load_data("$savedir/$d.jld2"), disciplines))
        trim_data!(data, iteration_restart)
    end

    # Bayesian optimization
    @printf "%3s %9s %9s %9s %4s \n" "ite" "obj" "∑√J" "EIc" "Nfsb"
    ite = iteration_restart+1
    retrain = Dict((("n",0),("z",[zeros(Nz) for _=1:4],), ("eic",zeros(4))))
    while ite < n_ite+1 
        # A/ Fit ensemble
        trainingdir =  NamedTuple{disciplines}(map(d->"$savedir/training/$ite/$d", disciplines))
        ensemble = map(learn_feasible_set, data, trainingdir)

        # B/ Minimize EIc
        file = "$savedir/minimize/$ite"
        z, eic = minimize_ei(file, ensemble, data, idz, idz_all)

        # Retrain if eic is too small
        if (eic > -1e-6)
            if (retrain["n"]<4)
                retrain["n"] += 1
                println("Retraining after unpromising candidate point (EIC = $eic), $(retrain["n"])/4")
                retrain["z"][retrain["n"]] .= z
                retrain["eic"][retrain["n"]] = eic
                continue
            else
                i = argmin(retrain["eic"])
                z .= retrain["z"][i]
                eic = retrain["eic"][i]
            end
        end
        retrain["eic"] .*= 0
        for z_=retrain["z"]
            z_ .= 0.
        end
        retrain["n"] = 0

        # C/ Evaluate new sample
        for d=disciplines
            push!(data[d].Z, copy(data[d].Z[end]))
            getvar!(data[d].Z[end], z, idz[d], idz_all)
            push!(data[d].Zs, subspace_fun[d](data[d].Z[end]))
            push!(data[d].sqJ, norm(data[d].Z[end] .- data[d].Zs[end]))
            push!(data[d].fsb, data[d].sqJ[end]<tol)
            push!(data[d].ite, ite)

            save_data("$savedir/$d.jld2",data[d])
        end

        # D/ Show progress
        glb = sum(map(D->D.fsb[end],data))
        sqJ = sum(map(D->D.sqJ[end],data))
        @printf "%3i %.3e %.3e %.3e %2i \n" ite -z[1] sqJ eic glb
        ite += 1
    end
    return data
end

function load_data(filename)
    @assert (split(filename, ".")[end] == "jld2") "Filename extension must be jld2"
    dat = load(filename)
    return (; 
        Z  = dat["Z"]  ,
        Zs = dat["Zs"] ,
        sqJ= dat["sqJ"],
        fsb= dat["fsb"],
        ite= dat["ite"],
    )
end

function save_data(filename, D)
    @assert (split(filename, ".")[end] == "jld2") "Filename extension must be jld2"
    save(filename,
            "Z",D.Z,"Zs",D.Zs,
            "sqJ",D.sqJ,"fsb",D.fsb,
            "ite",D.ite)
end

function trim_data!(data, ite)
    function trim(D)
        keep = (D.ite .<= ite)
        map(v->v[keep],D)
    end
    map(trim, data)
end