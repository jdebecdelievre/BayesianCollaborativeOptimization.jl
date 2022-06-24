using BayesianCollaborativeOptimization

HOME = pwd()
include("$HOME/examples/tailless/tailless.jl")
##
ipoptions = Dict{Any,Any}("print_level"=>2, "tol"=>1e-8, "linear_solver"=>"ma97")
subspace = (;
    struc =(z,f)->struc_subspace(z,f, ipoptions),
    aero = (z,f)->aero_subspace(z,f,ipoptions))

savedir = "$HOME/examples/tailless/rxp"
options = BayesianCollaborativeOptimization.Options(
    n_ite = 5, # number of iterations
    ini_samples= 1, # number of initial random samples. 0 to use provided z0
    savedir=savedir, tol=1e-4)
solver = ADMM(tailless_idz)
data = solve(solver, subspace, tailless_idz, options)
