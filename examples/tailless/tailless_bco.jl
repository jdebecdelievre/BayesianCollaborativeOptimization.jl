using Pkg;
Pkg.activate(".")
using HouseholderNets
using BayesianCollaborativeOptimization
const BCO = BayesianCollaborativeOptimization

include("tailless.jl")
## subspaces
ipoptions = Dict("print_level"=>2, "tol"=>1e-8)
##

variables = (; 
    aero = global_variables,
    struc = global_variables)
subspace = (;
    aero = z->aero_subspace(z,ipoptions),
    struc = z->struc_subspace(z,ipoptions))
HOME = pwd()
savedir = "$HOME/examples/tailless/xp"
disciplines = keys(variables)

variables_all = mergevar(values(variables)...)
idz = map(indexbyname, variables)
idz_all = indexbyname(variables_all)
##
data = bco(savedir, variables, subspace);
##
# z = copy(data.struc.Z);
# f = data.aero.sqJ + data.struc.sqJ
# o = map(zz->(zz[idz.struc.R] + Zopt.R[1])^2, z);

# ##
# Es = load_ensemble("$savedir/training/6/struc/ensemble.jld2");
# data = load_data("$savedir/struc.jld2");
# Es.(data.Z)
# ##
# Ea = load_ensemble("$savedir/training/1/aero/ensemble.jld2");
# data = load_data("$savedir/aero.jld2");
# Ea.(data.Z)