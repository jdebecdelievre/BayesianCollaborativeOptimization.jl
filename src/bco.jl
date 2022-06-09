"""
Options structure for BCO
"""
@with_kw struct BCOoptions
    n_ite::Int64 = 30 # number of iterations
    ini_samples::Int64 = 2 # number of initial random samples. 0 to use provided z0
    iteration_restart::Int64 = 0 # Iteration Number at which to restart.
    warm_start_sampler::Int64 = 0
    tol::Float64 = 5e-3
    savedir::String = "bco_xp"

    # Training
    nparticles::Int64 = 12
    ntrials::Int64 = 1
    nlayers::Int64 = 20
    lr::Float64 = 0.01
    αlr::Float64 = .5
    N_epochs::Int64 = 100000
    logfreq::Int64 = 1000
    nresample::Int64 = 0
    dropprob::Float64 = 0.

    # Minimization
    stepsize::Float64=.3
end

"""
Default objective function to maximize.
"""
function default_obj(z,grad=nothing)
    if typeof(grad) <: AbstractArray
        grad .= 0.
        grad[1] = 1.
    end
    return z[1]
end

"""
Receives 2 NamedTuples, a BCOoptions structure, and optionally an objective function to maximize.
"""
function bco(variables, subspace_fun, options::BCOoptions; objective=default_obj)
    @assert keys(variables) == keys(subspace_fun) "$(keys(variables)) =/= $(keys(subspace_fun))"
    (; n_ite, ini_samples, iteration_restart, warm_start_sampler, tol, savedir) = options

    # create xp directory, save options
    mkpath(savedir)
    open("$savedir/inputs.txt","w") do io
        println(io,options)
        # k = @sprintf "%6s %9s %9s %9s" " " "ini" "lower" "upper"
        # println(io,k)
        # for disc = keys(variables)
        #     var = variables[disc]
        #     println(io, "\n $disc:")
        #     for v=keys(var)
        #         k = @sprintf "%6s %.3e %.3e %.3e" v var[v].ini var[v].lb var[v].ub
        #         println(io, k)
        #     end
        # end
    end
    JLD2.save("$savedir/inputs.jld2","options",options,"variables",variables)

    # Create list of global vars
    disciplines = keys(variables)
    variables_all = mergevar(values(variables)...)
    Nz = len(variables_all)
    idz = map(indexbyname, variables)
    idz_all = indexbyname(variables_all)

    # Create sampler
    Sz = SobolSeq(Nz)
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
            Z = [next!(Sz) for _=1:ini_samples]
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
        ensemble = map((d,t)->learn_feasible_set(d,t,options), data, trainingdir)

        # B/ Minimize EIc
        file = "$savedir/eic/$ite"
        z, eic = maximize_ei(file, ensemble, data, idz, idz_all, options, objective=objective)

        # Retrain if eic is too small
        if (eic < 1e-6)
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