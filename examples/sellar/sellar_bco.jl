using Distributed
using Pkg
Pkg.activate("/home/adgboost/.julia/dev/BayesianCollaborativeOptimization/")
addprocs(7, exeflags="--project=$(Base.active_project())")

@everywhere begin
    using BayesianCollaborativeOptimization
    HOME = pwd()
    include("$HOME/examples/sellar/sellar.jl")

    function run(i)
        solver = BCO(Sellar(), N_epochs=500_000, stepsize=10.,dropprob= 0.02, αlr=0.97, nlayers=40)
        options = SolveOptions(n_ite=15, ini_samples=1, warm_start_sampler=i, tol=1e-6, savedir="xpu$i")
        obj, sqJ, fsb, Z = solve(solver, options, terminal_print=false)
    end
end

metrics = pmap(i->run(i),1:20)
