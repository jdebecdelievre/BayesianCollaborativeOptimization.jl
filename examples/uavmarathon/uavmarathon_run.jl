using Pkg
Pkg.activate(".")
using LinearAlgebra
using JLD2
using BayesianCollaborativeOptimization
using Parameters
using Snopt
using SNOW
HOME = pwd()
include("$HOME/examples/uavmarathon/uavmarathon.jl")

##
for i=1:20
    options = SolveOptions(n_ite=15, ini_samples=1, warm_start_sampler=i, savedir="examples/uavmarathon/xpu$i/bco")
    solver = BCO(UAVmarathon(), training_tol=1e-3, N_epochs=300_000, stepsize=100., 
                            dropprob=0.02, αlr=0.97, nlayers=20, tol=1e-3)
    solve(solver, options, terminal_print=false)
    
    ## 
    options = SolveOptions(n_ite=150, ini_samples=1, warm_start_sampler=i, savedir="examples/uavmarathon/xpu$i/sqp")
    solver = SQP(UAVmarathon(), λ=1.,tol=1e-6)
    solve(solver, options, terminal_print=false)
    
    ##
    options = SolveOptions(n_ite=150, ini_samples=1, warm_start_sampler=i, savedir="examples/uavmarathon/xpu$i/admm")
    solver = ADMM(UAVmarathon(), ρ=.1)
    solve(solver, options, terminal_print=false)
end