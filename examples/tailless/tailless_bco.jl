using Distributed
using Pkg
Pkg.activate("/home/adgboost/.julia/dev/BayesianCollaborativeOptimization/")
addprocs(7, exeflags="--project=$(Base.active_project())")

@everywhere begin
    using HouseholderNets
    using BayesianCollaborativeOptimization
    const BCO = BayesianCollaborativeOptimization
    HOME = pwd()
    include("$HOME/examples/tailless/tailless.jl")


    ## subspaces
    ipoptions = Dict("print_level"=>2, "file_print_level"=>5, 
                    "tol"=>1e-8, "output_file"=>"tmp.txt", "linear_solver"=>"ma97")
    variables = (; 
        struc = struc_global,
        aero = aero_global)
    subspace = (;
        struc =(z,f)->struc_subspace(z,f, ipoptions),
        aero = (z,f)->aero_subspace(z,f,ipoptions))

    disciplines = keys(variables)

    variables_all = mergevar(values(variables)...)
    @assert keys(variables_all)[1] == :R
    idz = map(indexbyname, variables)
    idz_all = indexbyname(variables_all)

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
        # set initial value
        v = unscale_unpack(Z0[i], idz_all, variables_all)
        for k = keys(v)
            variables_all[k].ini .= v[k]
        end
        @assert norm(ini_scaled(variables_all) - Z0[i]) < 1e-10

        savedir = "$HOME/examples/tailless/xpu$i"
        options = BCOoptions(
            n_ite = 15, # number of iterations
            ini_samples= 0, # number of initial random samples. 0 to use provided z0
            savedir=savedir, ntrials=2, nparticles=8, nlayers=0, lr=0.01,
            warm_start_sampler=i-1, stepsize=1., tol=1e-4,
            αlr=.95, N_epochs=2_500_000, logfreq=5000, nresample=0,
            dropprob=0.02)

        obj, sqJ, fsb, Z = bco(variables, subspace, options)
        metric = abs.(obj .- zopt[1]) + sum(sqJ)
    end

end
##
metrics = pmap(i->run(i),1:20)

using JLD2
HOME = pwd()
savedir = "$HOME/examples/tailless"
save("$savedir/metrics.jld2", "metrics", metrics)   