using BayesianCollaborativeOptimization
using LinearAlgebra
using Statistics
using JLD2
using Plots
HOME = pwd()
include("$HOME/examples/sellar/sellar.jl")
sl = Sellar()
disciplines = discipline_names(sl)

##
nruns = 20
pad = true
method = ["sqp1", "sqp", "sqp100","bco","bco30"]#["admm","bco","sqp"]
metric = Dict{Any,Any}()
obj = Dict{Any,Any}()
dobj = Dict{Any,Any}()
sqJ = Dict{Any,Any}()
for m=method
    #
    M = map(i->get_metrics(sl, "$HOME/examples/sellar/xp_jun30/$m/xpu$i"),1:nruns)
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

##
p = plot(yaxis=:log10)
for m=method
    l = min(30,minimum(length, metric[m]))
    plot!(p,mean([m_[1:l] for m_=metric[m]]), yaxis=:log10,label=m)
end
xlabel!("Number of Subspace Evaluations")
ylabel!("Metric")
title!("Average Metric For 20 Random Initial Guesses")
# savefig("$HOME/examples/sellar/xp_jun30/comparative_plot_june28_sellar.pdf")

##
p = plot(yaxis=:log10)
for m=method
    l = min(30,minimum(length, dobj[m]))
    plot!(p,mean([abs.(m_[1:l]) for m_=dobj[m]]),label=m)
end
xlabel!("Number of Subspace Evaluations")
ylabel!("Objective")
title!("Average obj For 20 Random Initial Guesses")
# savefig("$HOME/examples/sellar/xp_jun30/comparative_plot_june28_sellar.pdf")
##
p = plot(yaxis=:log10)
l=30
for m=method
    l = min(30,minimum(d->length(sum(d)), sqJ[m]))
    plot!(p,mean([sum(m_)[1:l] for m_=sqJ[m]]),label=m)
end
xlabel!("Number of Subspace Evaluations")
ylabel!("√J₁+√J₂")
title!("Average Infeasibility For 20 Random Initial Guesses")
