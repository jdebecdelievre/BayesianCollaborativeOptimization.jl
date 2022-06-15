using Plots
using JLD2
using HouseholderNets
using BayesianCollaborativeOptimization
const BCO = BayesianCollaborativeOptimization
include("tailless.jl")
using Statistics

disciplines = (:struc,:aero)
ite = 15

variables = (; 
    struc = struc_global,
    aero = aero_global)
variables_all = mergevar(values(variables)...)
idz = map(indexbyname, variables)
idz_all = indexbyname(variables_all)
##
svd = "examples/tailless"
function metrics(svd)
        sqJaero  = Vector{Float64}[]
        sqJstruc = Vector{Float64}[]
        obj = Vector{Float64}[]
        obj_sc = Vector{Float64}[]
        nxp = 14
        for i = 1:nxp
                @load "$svd/xp$i/aero.jld2" Z Zs sqJ
                push!(sqJaero, sqJ)
                @load "$svd/xp$i/struc.jld2" Z Zs sqJ
                push!(sqJstruc, sqJ)
                push!(obj_sc, [abs(z[1]-Zopt.R[1]) for z=Z])
                # push!(obj_sc, [abs(unscale_unpack(z, idz_all, variables_all).R-TL_optimum.R)/TL_optimum.R for z=Z])
                push!(obj, [z[1] for z=Z])
        end
        ##
        metric = (obj_sc + sqJstruc + sqJaero)
        for m=metric
        for i=2:length(m)
            m[i] = min(m[i],m[i-1])
        end
        end

        ##
        lm = minimum(length(m) for m=metric)
        μ = [mean(m[i] for m=metric) for i=1:lm]
        return μ, metric, obj, obj_sc, sqJaero, sqJstruc
end
met = [1.813618, 1.326392, 0.467193, 0.332232, 0.251307, 0.203285, 0.179433, 0.143485, 0.113937, 0.097779, 0.085133, 0.071687, 0.061396, 0.055545, 0.054146, 0.052986, 0.049264, 0.047197, 0.040634, 0.038498, 0.035806, 0.030987, 0.028716, 0.024090, 0.022092, 0.020837, 0.020427, 0.019728, 0.017669, 0.016691,]/2
μ, metric, obj, obj_sc, sqJaero, sqJstruc = metrics(svd);
p = plot(μ,yscale=:log10,label="us")
plot!(met,label="gpsort")

## Ref result
savedir = "examples/tailless/xp2"
ite = 5
datadir = NamedTuple{disciplines}(map(d->"$savedir/$d.jld2",disciplines))
data = map(load_data, datadir)
data = trim_data!(data, ite)

netdir = NamedTuple{disciplines}(map(d->"$savedir/training/$ite/$d/ensemble.jld2",disciplines))
ensemble = map(load_ensemble, netdir);
@load "$savedir/training/$ite/aero/losses.jld2" loss
@load "$savedir/eic/$ite/eic.jld2" EIc maxZ iniZ msg ite
##
options = BCOoptions(
    n_ite = 10, # number of iterations
    ini_samples= 2, # number of initial random samples. 0 to use provided z0
    savedir=savedir, nparticles=12, nlayers=20, lr=0.01,
    αlr=.5, N_epochs=500_000, logfreq=2000, nresample=0,
    dropprob=0.02)
file = "$savedir/eic/$ite"

##
ipoptions = Dict("print_level"=>2, "file_print_level"=>5, "tol"=>1e-8, "output_file"=>"tmp.txt")
mkpath("$savedir/eval/aero")
aero_subspace(data.aero.Z[1],"$savedir/eval/aero/1.txt", ipoptions)
mkpath("$savedir/eval/struc")
struc_subspace(data.struc.Z[1],"$savedir/eval/struc/1.txt", ipoptions)
##
z, eic = maximize_ei(file, ensemble, data, idz, idz_all, options)
    