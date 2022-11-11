using BayesianCollaborativeOptimization

include(joinpath(@__DIR__,"tailless.jl"))
##

solver = ADMM(Tailless(), ρ=.1)
options = SolveOptions()
data = solve(solver, options);

##
solve_aao()