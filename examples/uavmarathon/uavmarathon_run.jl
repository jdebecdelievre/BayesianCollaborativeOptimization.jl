using Pkg
Pkg.activate("$(@__DIR__)/../.")
##
using Revise
using LinearAlgebra
using JLD2
using BayesianCollaborativeOptimization
using Parameters
using Snopt
using SNOW
HOME = pwd()
include(joinpath(@__DIR__,"uavmarathon.jl"))

##
for i=1:20
    options = SolveOptions(n_ite=30, ini_samples=1, iteration_restart=0, warm_start_sampler=i, savedir="examples/uavmarathon/tmp/bco/xpu$i")
    solver = BCO(UAVmarathon(), training_tol=1e-3, N_epochs=300_000, stepsize=100., 
                            dropprob=nothing, αlr=0.97, nlayers=20, nchannels=5, tol=1e-3, nparticles=2, ntrials=1)
    solve(solver, options, terminal_print=false)
    
    ## 
    # options = SolveOptions(n_ite=150, ini_samples=1, warm_start_sampler=i, savedir="examples/uavmarathon/xp_feb14/xpu$i/sqp")
    # solver = SQP(UAVmarathon(), λ=1.,tol=1e-6)
    # solve(solver, options, terminal_print=false)
    
    # ##
    # options = SolveOptions(n_ite=150, ini_samples=1, warm_start_sampler=i, savedir="examples/uavmarathon/xp_feb14/xpu$i/admm")
    # solver = ADMM(UAVmarathon(), ρ=.1)
    # solve(solver, options, terminal_print=false)
end