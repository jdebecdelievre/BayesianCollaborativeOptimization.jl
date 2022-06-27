using Plots
using JLD2
using HouseholderNets
using BayesianCollaborativeOptimization
const BCO = BayesianCollaborativeOptimization
include("$(pwd())/examples/tailless/tailless.jl")

using Statistics

disciplines = (:struc,:aero)

variables = (; 
    struc = struc_global,
    aero = aero_global)
variables_all = mergevar(values(variables)...)
idz = map(indexbyname, variables)
idz_all = indexbyname(variables_all)

##
nruns = 7
svd = "examples/tailless"
metric = [get_metrics("examples/tailless/xpu$i",disciplines,zopt,default_obj)[1] for i=1:nruns]
plot(metric, yaxis=:log10)
lm = minimum(length.(metric))
plot!(mean([m[1:lm] for m=metric]), yaxis=:log10,label="mean")
met = [1.813618, 1.326392, 0.467193, 0.332232, 0.251307, 0.203285, 0.179433, 0.143485, 0.113937, 0.097779, 0.085133, 0.071687, 0.061396, 0.055545, 0.054146, 0.052986, 0.049264, 0.047197, 0.040634, 0.038498, 0.035806, 0.030987, 0.028716, 0.024090, 0.022092, 0.020837, 0.020427, 0.019728, 0.017669, 0.016691,]/2
plot!(met, yaxis=:log10,label="gsort")

## Ref result
savedir = "examples/tailless/xpu6"
metric, _, _ = get_metrics(savedir, disciplines, zopt, default_obj)
iteration = 16
@load "$savedir/data.jld2" Z sqJ fsb obj
data = load_data(savedir, disciplines);
datacheck(data)
data = trim_data!(data, iteration)

ensemble = load_ensemble("$savedir/training/$iteration", disciplines)

@load "$savedir/training/$iteration/aero/losses.jld2" loss
@load "$savedir/eic/$iteration/eic.jld2" EIc maxZ iniZ msg ite best stepsize

##
file = "tmp"
options = BCOoptions(
    n_ite = 10, # number of iterations
    ini_samples= 2, # number of initial random samples. 0 to use provided z0
    savedir=savedir, nparticles=12, nlayers=20, lr=0.01,
    αlr=.95, N_epochs=500_000, logfreq=2000, nresample=0, stepsize=10.,
    dropprob=0.02)

##
ipoptions = Dict("print_level"=>5, "file_print_level"=>5, "tol"=>1e-8, "output_file"=>"tmp.txt", 
                "linear_solver"=>"ma97")
##
mkpath("$savedir/eval/aero")
for i=1:length(data.aero.Z)
    aero_subspace(data.aero.Z[i],"$savedir/eval/aero/$i.txt", ipoptions)
end
##
include("tailless.jl")
mkpath("$savedir/eval/struc")
##
sqJs = copy(sqJ.struc)
for i=1:length(data.struc.Z)
    zs = struc_subspace(data.struc.Z[i],"$savedir/eval/struc/$i.txt", ipoptions)
    sqJs[i] = norm(zs - data.struc.Z[i])
    data.struc.sqJ[i] = norm(zs - data.struc.Z[i])
end

##
data = trim_data!(data, iteration-1)
z, eic = maximize_ei(file, ensemble, data, idz, idz_all, options)
                                                                                                                     