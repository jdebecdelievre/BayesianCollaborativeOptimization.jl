using Distributed
using Pkg
Pkg.activate("/home/adgboost/.julia/dev/BayesianCollaborativeOptimization/")
addprocs(1, exeflags="--project=$(Base.active_project())")

@everywhere begin
    using HouseholderNets
    using BayesianCollaborativeOptimization
    const BCO = BayesianCollaborativeOptimization
    HOME = pwd()
    include("$HOME/examples/tailless/tailless.jl")


    ## subspaces
    ipoptions = Dict("print_level"=>2, "file_print_level"=>5, "tol"=>1e-8, "output_file"=>"tmp.txt")
    variables = (; 
        struc = struc_global,
        aero = aero_global)
    subspace = (;
        struc =(z,f)->struc_subspace(z,f, ipoptions),
        aero = (z,f)->aero_subspace(z,f,ipoptions))

    savedir = "$HOME/examples/tailless/xp"
    disciplines = keys(variables)

    variables_all = mergevar(values(variables)...)
    @assert keys(variables_all)[1] == :R
    idz = map(indexbyname, variables)
    idz_all = indexbyname(variables_all)

    function run(i)
        savedir = "$HOME/examples/tailless/rxp$i"
        options = BCOoptions(
            n_ite = 5, # number of iterations
            ini_samples= 2, # number of initial random samples. 0 to use provided z0
            savedir=savedir, ntrials=2, nparticles=8, nlayers=0, lr=0.01,
            warm_start_sampler=i-1, stepsize=.1, tol=1e-4,
            αlr=.95, N_epochs=2_500_000, logfreq=5000, nresample=0,
            dropprob=0.02)
        data = bco(variables, subspace, options)
    end

end
##
pmap(i->run(i),1:20)
