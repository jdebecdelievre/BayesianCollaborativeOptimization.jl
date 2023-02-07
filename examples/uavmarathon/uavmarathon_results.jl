using BayesianCollaborativeOptimization
using LinearAlgebra
using Statistics
using LaTeXStrings
using JLD2
using Plots
HOME = pwd()
include("$HOME/examples/uavmarathon/uavmarathon.jl")
T = UAVmarathon()

##
nruns = 20
pad = true
method = ["bco","sqp","admm"]
metric = Dict{Any,Any}()
obj = Dict{Any,Any}()
dobj = Dict{Any,Any}()
sqJ = Dict{Any,Any}()
for m=method
    #
    M = map(i->get_metrics(T, "$HOME/examples/uavmarathon/xpu$(i)/$m"),1:nruns)
    metric[m] = first.(M)
    sqJ[m] = last.(M)
    obj[m] = getindex.(M,[2])
    dobj[m] = getindex.(M,[3])

    #
    if pad
        l         = maximum(length, metric[m])
        metric[m] = [[s;ones(l-length(s))*s[end]] for s=metric[m]]
        obj[m]    = [[s;ones(l-length(s))*s[end]] for s=obj[m]]
        dobj[m]   = [[s;ones(l-length(s))*s[end]] for s=dobj[m]]
        sqJ[m]    = [map(s->[s; ones(l-length(s))*s[end]],ss) for ss=sqJ[m]]
    end
end

## Objective Function
method = ["bco","sqp","admm"]
leg = ["Bayesian CO", "SQP","ADMM"]
using Plots.PlotMeasures
p = plot(legend=:bottomright)
ylabel!(p, "Airspeed V (m/s)")
lo = V.V.lb[1]
up = V.V.ub[1]
for (kc,m)=enumerate(method)
    l0  = (m[1:3] == "sqp") ? 2 : 1
    l = min(16,minimum(length, metric[m])) + l0 - 1
    O = zeros(l-l0+1)
    n = 0
    for k=1:20
        o = obj[m][k][l0:l] * (up-lo) .+ lo
        O += o
        n += 1
        plot!(p,o, alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, O./n, linewidth=2, label=leg[kc],color=kc)
end
hline!(p, [opt.V], linestyle=:dash, label="optimum")
xlabel!("Number of Subspace Evaluations")
title!("Objective Function For 20 Random Initial Guesses")
savefig("$HOME/examples/uavmarathon/comparative_obj_jul20_uavmarathon.pdf")


## Feasibility
p = plot(yaxis=:log10)
ylabel!(p, L"\sqrt{J_{wing}} + \sqrt{J_{prop}}")
for (kc,m)=enumerate(method)
    l0  = (m[1:3] == "sqp") ? 2 : 1
    l = min(16,minimum(length, metric[m])) + l0-1
    SQJ = zeros(l-l0+1)
    n = 0
    for k=1:20
        sqj = sum(sqJ[m][k])[l0:l]
        SQJ += sqj
        n += 1
        plot!(p,sqj, alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, SQJ./n, linewidth=2, label=leg[kc],color=kc)
end
ylims!(1e-4, 1.)
xlabel!("Number of Subspace Evaluations")
title!("Feasibility For 20 Random Initial Guesses")
savefig("$HOME/examples/uavmarathon/comparative_sqJ_jul20_uavmarathon.pdf")