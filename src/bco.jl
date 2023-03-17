"""
Options structure for BCO
"""
struct BCO{problem,DP<:Union{Nothing,Float64}} <: AbstractSolver{problem}
    problem::problem
    tol::Float64

    # Training
    training_tol::Float64
    nparticles::Int64 
    ntrials::Int64
    nlayers::Int64 
    nchannels::Int64
    lr::Float64
    αlr::Float64 
    N_epochs::Int64 
    logfreq::Int64 
    nresample::Int64
    dropprob::DP 

    # Minimization
    stepsize::Float64
    
    # Time tracking
    to::TimerOutput

    function BCO(problem::prb;
        tol::Float64 = 5e-3,

        # Training
        training_tol::Float64 = 1e-6,
        nparticles::Int64 = 6,
        ntrials::Int64 = 2,
        nlayers::Int64 = 20,
        nchannels::Int64 = 5,
        lr::Float64 = 0.01,
        αlr::Float64 = .95,
        N_epochs::Int64 = 500_000,
        logfreq::Int64 = 1000,
        nresample::Int64 = 0,
        dropprob::DP,

        # Minimization
        stepsize::Float64=1.

        ) where {prb<:AbstractProblem,DP<:Union{Nothing,Float64}}
        to = TimerOutput()
        new{prb,DP}(problem, tol, training_tol, nparticles, ntrials, nlayers, nchannels, lr, αlr, N_epochs, logfreq, nresample, dropprob, stepsize, to)
    end
end

function Base.show(io::IO, bco::BCO)
    print(io, 
"""BayesianCollaborativeOptimization:
    problem::: $(bco.problem)
    tol: $(bco.tol)
    
    # Training
    training_tol: $(bco.training_tol)
    nparticles: $(bco.nparticles)
    ntrials: $(bco.ntrials)
    nlayers: $(bco.nlayers)
    nchannels: $(bco.nchannels)
    lr: $(bco.lr)
    αlr: $(bco.αlr)
    N_epochs: $(bco.N_epochs)
    logfreq: $(bco.logfreq)
    nresample: $(bco.nresample)
    dropprob: $(bco.dropprob)
    
    # Minimization
    stepsize: $(bco.stepsize)
""")
end

"""
Receives 2 NamedTuples, a BCOoptions structure, and optionally an objective function to maximize.
"""
function get_new_point(ite::Int64, solver::BCO, data::NamedTuple{disciplines}, savedir::String) where disciplines
    mkpath(savedir)
    to = solver.to

    # A/ Fit ensemble
    @timeit to "learn" begin
        # trainingdir =  NamedTuple{disciplines}(map(d->"$savedir/training/$ite/$d", disciplines)
        ensemble = NamedTuple{disciplines}(map(d -> (@timeit to "learn_$d" learn_feasible_set(solver, data[d], "$savedir/training/$ite/$d")), disciplines))
    end
    
    # B/ Minimize EIc
    @timeit to "optimize" begin
        z, eic = maximize_ei(solver, data, ensemble, "$savedir/eic/$ite")
    end

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
    Zd = map(id->z[id], indexmap(solver.problem))
    return z, Zd, eic
end