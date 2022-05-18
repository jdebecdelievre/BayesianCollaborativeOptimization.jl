using Distributed
using Pkg
Pkg.activate("/home/adgboost/.julia/dev/BayesianCollaborativeOptimization/")
addprocs(7, exeflags="--project=$(Base.active_project())")

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
    savedir = "$HOME/examples/tailless/xp"
    disciplines = keys(variables)

    variables_all = mergevar(values(variables)...)
    @assert keys(variables_all)[1] == :R
    idz = map(indexbyname, variables)
    idz_all = indexbyname(variables_all)

    function run(i)
        savedir = "$HOME/examples/tailless/xp$i"
        data = bco(savedir, variables, subspace, warm_start_sampler=i-1);
    end

end
##
pmap(i->run(i),1:20)

# ##
# using HouseholderNets
# using BayesianCollaborativeOptimization
# const BCO = BayesianCollaborativeOptimization

# include("tailless.jl")


# ## subspaces
# ipoptions = Dict("print_level"=>2, "tol"=>1e-8)
# variables = (; 
#     struc = struc_global,
#     aero = aero_global)
# subspace = (;
#     struc = z->struc_subspace(z,ipoptions),
#     aero = z->aero_subspace(z,ipoptions))
# HOME = pwd()
# savedir = "$HOME/examples/tailless/xp"
# disciplines = keys(variables)

# variables_all = mergevar(values(variables)...)
# @assert keys(variables_all)[1] == :R
# idz = map(indexbyname, variables)
# idz_all = indexbyname(variables_all)

# function run(i)
#     savedir = "$HOME/examples/tailless/xp$i"
#     data = bco(savedir, variables, subspace, warm_start_sampler=i-1);
# end
# run(1)

# # z = copy(data.struc.Z);
# # f = data.aero.sqJ + data.struc.sqJ
# # o = map(zz->(zz[idz.struc.R] + Zopt.R[1])^2, z);

# # ##
# # Es = load_ensemble("$savedir/training/6/struc/ensemble.jld2");
# # data = load_data("$savedir/struc.jld2");
# # Es.(data.Z)
# # ##
# # Ea = load_ensemble("$savedir/training/1/aero/ensemble.jld2");
# # data = load_data("$savedir/aero.jld2");
# # Ea.(data.Z)