using Pkg
Pkg.activate("$(@__DIR__)/../.")

using BayesianCollaborativeOptimization
using LinearAlgebra
using Statistics
using JLD2
using Plots
using CSV
HOME = pwd()
include("$HOME/examples/sellar/sellar.jl")
sl = Sellar()
disciplines = discipline_names(sl)

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
    M = map(i->get_metrics(sl, "$HOME/examples/sellar/xp_jul20/$m/xpu$i"),1:nruns)
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

##### PLOTS FOR DEFENSE PRESENTATION #####

## Objective Function
method = ["bco","sqp","admm"]
leg = ["Bayesian CO", "SQP","ADMM"]
using Plots.PlotMeasures
p = plot()#yaxis=:log10)
ylabel!(p, "Objective Function")
for (kc,m)=enumerate(method)
    l0  = (m[1:3] == "sqp") ? 2 : 1
    l = min(15,minimum(length, metric[m])) + l0 - 1
    O = zeros(l-l0+1)
    n = 0
    for k=1:20
        o = -obj[m][k][l0:l]
        O += o
        n += 1
        plot!(p,o, alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, O./n, linewidth=2, label=leg[kc],color=kc)
end
hline!(p, -objective_opt(Sellar()), linestyle=:dash, label="optimum")
xlabel!("Number of Subspace Evaluations")
title!("Objective Function For 20 Random Initial Guesses")
savefig("$HOME/examples/sellar/comparative_obj_jul20_sellar.pdf")


## Feasibility
p = plot(yaxis=:log10)
ylabel!("√J₁+√J₂")
for (kc,m)=enumerate(method)
    l0  = (m[1:3] == "sqp") ? 2 : 1
    l = min(15,minimum(length, metric[m])) + l0-1
    SQJ = zeros(l-l0+1)
    n = 0
    for k=1:20
        sqj = sum(sqJ[m][k])[l0:l]
        SQJ += sqj
        n += 1
        plot!(p,sqj,alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, SQJ./n, linewidth=2, label=leg[kc],color=kc)
end
ylims!(5e-5, 1.)
xlabel!("Number of Subspace Evaluations")
title!("Feasibility For 20 Random Initial Guesses")
savefig("$HOME/examples/sellar/comparative_sqJ_jul20_sellar.pdf")



## Objective function zoomed in, without ADMM
p = plot()#yaxis=:log10)
ylabel!(p, "Objective Function")
method = ["bco", "sqp"]
leg = ["Bayesian CO", "SQP"]
for (kc,m)=enumerate(method)
    l0  = (m[1:3] == "sqp") ? 2 : 1
    l = min(15,minimum(length, metric[m])) + l0 - 1
    O1 = zeros(l-l0+1)
    O2 = zeros(l-l0+1)
    n1 = 0
    n2 = 0
    for k=1:20
        o = -obj[m][k][l0:l]
        if abs(o[end]+objective_opt(sl)[1]) < abs(o[end]+objective_opt(sl)[2])
            O1 += o
            n1 += 1
        else
            O2 += o
            n2 += 1
        end
        plot!(p,o, alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, O1./n1, linewidth=2, label=leg[kc],color=kc)
    plot!(p, O2./n2, linewidth=2, label="",color=kc)
    @show n1 n2
end
hline!(p, -objective_opt(sl), linestyle=:dash, label="optimum")
ylims!(5.,30,)
xlabel!("Number of Subspace Evaluations")
title!("Objective Function For 20 Random Initial Guesses")
savefig("$HOME/examples/sellar/comparative_obj_zoomed_jul20_sellar.pdf")

## Feasibility zoomed in without ADMM
p = plot(yaxis=:log10)
ylabel!("√J₁+√J₂")
for (kc,m)=enumerate(method)
    l0  = (m[1:3] == "sqp") ? 2 : 1
    l = min(15,minimum(length, metric[m])) + l0-1
    SQJ = zeros(l-l0+1)
    n = 0
    for k=1:20
        sqj = sum(sqJ[m][k])[l0:l]
        SQJ += sqj
        n += 1
        plot!(p,sqj,alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, SQJ./n, linewidth=2, label=leg[kc],color=kc)
end
ylims!(5e-5, .1)
xlabel!("Number of Subspace Evaluations")
title!("Feasibility For 20 Random Initial Guesses")
savefig("$HOME/examples/sellar/comparative_sqJ_zoomed_jul20_sellar.pdf")
