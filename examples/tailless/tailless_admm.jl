using BayesianCollaborativeOptimization

HOME = pwd()
include("$HOME/examples/tailless/tailless.jl")
##

solver = ADMM(Tailless(), ρ=.1)
options = SolveOptions()
data = solve(solver, options);

##
solve_aao()