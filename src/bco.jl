"""
Options structure for BCO
"""
@with_kw struct BCO <: AbstractSolver
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
Receives 2 NamedTuples, a BCOoptions structure, and optionally an objective function to maximize.
"""
function get_new_point(ite::Int64, solver::BCO, objective,
                        data::NamedTuple{disciplines}, idz::IndexMap{disciplines}, 
                        savedir::String) where disciplines
    mkpath(savedir)

    # A/ Fit ensemble
    trainingdir =  NamedTuple{disciplines}(map(d->"$savedir/training/$d", disciplines))
    ensemble = map((d,t)->learn_feasible_set(d,t,solver), data, trainingdir)

    # B/ Minimize EIc
    z, eic = maximize_ei("$savedir/eic", ensemble, data, idz, objective, solver)

    # # Retrain if eic is too small
    # if (eic < 1e-6)
    #     if (retrain["n"]<4)
    #         retrain["n"] += 1
    #         println("Retraining after unpromising candidate point (EIC = $eic), $(retrain["n"])/4")
    #         retrain["z"][retrain["n"]] .= z
    #         retrain["eic"][retrain["n"]] = eic
    #         continue
    #     else


    #         i = argmin(retrain["eic"])
    #         z .= retrain["z"][i]
    #         eic = retrain["eic"][i]
    #     end
    # end
    # retrain["eic"] .*= 0
    # for z_=retrain["z"]
    #     z_ .= 0.
    # end
    # retrain["n"] = 0

    return z, eic
end