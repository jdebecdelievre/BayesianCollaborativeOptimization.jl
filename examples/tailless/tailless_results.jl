using Plots
using JLD2
using HouseholderNets
using BayesianCollaborativeOptimization
const BCO = BayesianCollaborativeOptimization
include("tailless.jl")
using Statistics

##
savedir = "examples/tailless/xp"
disciplines = (:struc,:aero)
ite = 15
# data = NamedTuple{disciplines}(map(d->load_data("$savedir/$d.jld2"), disciplines))
# ensemble_path = map(d->"$savedir/training/$ite/$d/ensemble.jld2", disciplines)
# ensemble = map(load_ensemble, ensemble_path);
# data = trim_data!(data, ite)

variables = (; 
    struc = struc_global,
    aero = aero_global)
variables_all = mergevar(values(variables)...)
idz = map(indexbyname, variables)
idz_all = indexbyname(variables_all)
##

##
showpr(data.B.Z,
        ensembleB,
        data.B.fsb,[0,1])
##
showpr(data.A.Z,
        ensembleA,
        data.A.fsb,[0,1])
##
showfn(data.B.Z,
        ensembleB[2],   
        data.B.fsb,[0,1])
        
##
file = "$savedir/eic.jld2"
@load file EIc minZ best iniZ
eic(z) = -max(0.,(best+z[1])*(ensembleA(z)*ensembleB(z)))#*exp(-(z-data.A.Z[end-1])⋅(z-data.A.Z[end-1]))
# showpr(data.A.Z,
#         eic,
#         data.A.fsb,[0.,1])

##
svd = "examples/tailless"
# svd = "examples/tailless/run_10mai2022"
function metrics(svd)
        sqJaero  = Vector{Float64}[]
        sqJstruc = Vector{Float64}[]
        obj = Vector{Float64}[]
        obj_sc = Vector{Float64}[]
        nxp = 16
        for i = 1:nxp
                @load "$svd/xp$i/aero.jld2" Z Zs sqJ
                push!(sqJaero, sqJ)
                @load "$svd/xp$i/struc.jld2" Z Zs sqJ
                push!(sqJstruc, sqJ)
                push!(obj_sc, [abs(z[1]-Zopt.R[1])/Zopt.R[1] for z=Z])
                push!(obj, [z[1] for z=Z])
        end
        ##
        metric = obj_sc .+ sqJstruc*2 .+ sqJaero*2
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
met = [1.813618, 1.326392, 0.467193, 0.332232, 0.251307, 0.203285, 0.179433, 0.143485, 0.113937, 0.097779, 0.085133, 0.071687, 0.061396, 0.055545, 0.054146, 0.052986, 0.049264, 0.047197, 0.040634, 0.038498, 0.035806, 0.030987, 0.028716, 0.024090, 0.022092, 0.020837, 0.020427, 0.019728, 0.017669, 0.016691,]
# metricfs = metrics("examples/tailless/May13_feasiblefirst")[1];
# metricstep = metrics("examples/tailless/May13_sig2e-2")[1];
# metricopt = metrics("examples/tailless/May13_zopt")[1];
# metriclucky = metrics("examples/tailless/May13_luckystrike")[1];
# metrictol = metrics("examples/tailless/May13_tol5e-3")[1];
μ, metric, obj, obj_sc, sqJaero, sqJstruc = metrics(svd);

##
# using DataFrames
# using CSV
# metric = DataFrame(["run$i"=>metric[i] for i=1:length(metric)])
# CSV.write("examples/tailless/tailless_metric_aviation_rerun_Householder_may10th2022_2.csv", metric)
##
p = plot(μ,yscale=:log10,label="us")
plot!(met,label="gpsort")
# plot!(metricfs,label="fs")
plot!(metrictol,label="tol")
plot!(metricopt,label="zopt")
plot!(metriclucky,label="zlucky")
##
eic = []
for i=1:30
        @load "$svd/xp1/minimize/$i/eic.jld2" EIc minZ iniZ
        push!(eic, minimum(EIc))
end

## Ref result
savedir = NamedTuple{disciplines}(map(d->"examples/tailless/xp1/$d.jld2",disciplines))
data = map(load_data, savedir)
savedir = NamedTuple{disciplines}(map(d->"examples/tailless/xp1/training/1/$d/ensemble.jld2",disciplines))
ensemble = map(load_ensemble, savedir)
minimize_ei("tmp", ensemble, data, idz, idz_all)