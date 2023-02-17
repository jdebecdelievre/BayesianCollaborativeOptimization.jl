using Distributed
using Pkg
Pkg.activate("/home/adgboost/.julia/dev/BayesianCollaborativeOptimization/")
addprocs(7, exeflags="--project=$(Base.active_project())")

@everywhere begin
    using BayesianCollaborativeOptimization
    prefix = joinpath(@__DIR__,"xp_feb16_23/bco")
    include(joinpath(@__DIR__,"tailless.jl"))

    Z0 =  [[0.65625, 0.46875, 0.09375, 0.46875, 0.28125],
            [0.03125, 0.34375, 0.71875, 0.59375, 0.65625],
            [0.6875, 0.5625, 0.4375, 0.1875, 0.8125],
            [0.1875, 0.0625, 0.9375, 0.6875, 0.3125],
            [0.75, 0.75, 0.25, 0.75, 0.25],
            [0.9375, 0.8125, 0.6875, 0.4375, 0.0625],
            [0.25, 0.5, 0.5, 0.5, 0.5],
            [0.25, 0.25, 0.75, 0.25, 0.75],
            [0.78125, 0.59375, 0.96875, 0.34375, 0.90625],
            [0.125, 0.625, 0.125, 0.375, 0.375],
            [0.875, 0.375, 0.375, 0.625, 0.125],
            [0.5625, 0.1875, 0.3125, 0.3125, 0.6875],
            [0.53125, 0.84375, 0.21875, 0.09375, 0.15625],
            [0.625, 0.125, 0.625, 0.875, 0.875],
            [0.0625, 0.6875, 0.8125, 0.8125, 0.1875],
            [0.28125, 0.09375, 0.46875, 0.84375, 0.40625],
            [0.375, 0.875, 0.875, 0.125, 0.625],
            [0.4375, 0.3125, 0.1875, 0.9375, 0.5625],
            [0.8125, 0.4375, 0.5625, 0.0625, 0.4375],
            [0.3125, 0.9375, 0.0625, 0.5625, 0.9375]]

    function run(i)
        solver = BCO(Tailless(), 
            N_epochs=500_000, stepsize=10.,
            dropprob= 0.02, 
            αlr=0.97, 
            nlayers=40, nparticles=6, ntrials=2, training_tol=1e-3, tol=1e-3)
        # solver = SQP(Tailless(), tol=1e-3)
        # solver = ADMM(Tailless())
        options = SolveOptions(n_ite=30, ini_samples=1, 
                                warm_start_sampler=i, 
                                savedir="$prefix/xpu$i")
        obj, sqJ, fsb, Z = solve(solver, options, terminal_print=false)
    end
end

metrics = pmap(i->run(i),1:20)



# using JLD2
# HOME = pwd()
# savedir = "$HOME/examples/tailless"
# save("$savedir/metrics.jld2", "metrics", metrics)   