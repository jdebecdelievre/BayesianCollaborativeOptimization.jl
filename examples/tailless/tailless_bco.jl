using Distributed
using Pkg
Pkg.activate("/home/adgboost/.julia/dev/BayesianCollaborativeOptimization/")
addprocs(1, exeflags="--project=$(Base.active_project())")

@everywhere begin
    using HouseholderNets
    using BayesianCollaborativeOptimization
    const BCO = BayesianCollaborativeOptimization

    include("tailless.jl")


    ## subspaces
    ipoptions = Dict("print_level"=>2, "tol"=>1e-8)
    variables = (; 
        struc = struc_global,
        aero = aero_global)
    subspace = (;
        struc = z->struc_subspace(z,ipoptions),
        aero = z->aero_subspace(z,ipoptions))
    HOME = pwd()
    disciplines = keys(variables)

    variables_all = mergevar(values(variables)...)
    @assert keys(variables_all)[1] == :R
    idz = map(indexbyname, variables)
    idz_all = indexbyname(variables_all)

    function run(i)
        savedir = "$HOME/examples/tailless/rxp$i"
        data = bco(savedir, variables, subspace, warm_start_sampler=i-1);
    end

end
##
pmap(i->run(i),1:20)