using Distributed
using Pkg
Pkg.activate("/home/adgboost/.julia/dev/BayesianCollaborativeOptimization/")
addprocs(1, exeflags="--project=$(Base.active_project())")

@everywhere begin
    using BayesianCollaborativeOptimization
    HOME = pwd()
    include("$HOME/examples/twindisks/twindisks.jl")

    function run(i)
        # solver = BCO(TwinDisks(), training_tol=1e-5, N_epochs=500_000, 
        #                 stepsize=1000.,dropprob= 0.02, αlr=0.97, nlayers=40, tol=1e-3)
        solver = SQP(TwinDisks(), λ=1., tol=1e-6)
        # solver = ADMM(TwinDisks(), ρ=1., tol=1e-3)
        options = SolveOptions(n_ite=30, ini_samples=1, warm_start_sampler=i, 
                        savedir="$HOME/examples/twindisks/xp_jul20/sqp/xpu$i")
        obj, sqJ, fsb, Z = solve(solver, options, terminal_print=false)
    end
end

metrics = pmap(i->run(i),1:20)
