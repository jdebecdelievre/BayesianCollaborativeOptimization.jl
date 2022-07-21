using BayesianCollaborativeOptimization
using LinearAlgebra
using Statistics
using JLD2
using Plots
using LaTeXStrings
HOME = pwd()
include("$(pwd())/examples/tailless/tailless.jl")
T = Tailless()

gmet = [ 0.9395783514629933, 0.6644157664688632, 0.24305338004907254, 0.17855792712535565, 0.13656728089934528, 0.11159422426723604, 0.09911113515752532, 0.08001423078816536, 0.06527614788311345, 0.05723934247593221, 0.05138013348777313, 0.04522013720308668, 0.04036728301607296, 0.03746641125711485, 0.03481941384248512, 0.03376239609724129, 0.031230444212835818, 0.029693000039475236, 0.02658597813187403, 0.025020945003832163, 0.02379023622151321, 0.021621746143340434, 0.020600042846018886, 0.018518185621759747, 0.013758299082697554, 0.013193317834839841, 0.012863455300141397, 0.012502872281513039, 0.011418526185243106, 0.010792017807775733, 0.010674422951337281]

##

method = ["bco22","sqp22"]
nruns = 20
pad = true
metric = Dict{Any,Any}()
obj = Dict{Any,Any}()
dobj = Dict{Any,Any}()
sqJ = Dict{Any,Any}()
for m=method
    #
    M = map(i->get_metrics(T, "$HOME/examples/tailless/xp_jun29/$m/xpu$i"),1:nruns)
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
    l = min(32,minimum(length, metric[m]))
    plot!(p,mean([m_[1:l] for m_=metric[m]]), yaxis=:log10,label=m)
end
plot!(gmet, yaxis=:log10,label="gpsort")
xlabel!("Number of Subspace Evaluations")
ylabel!("Metric")
title!("Average Metric For 20 Random Initial Guesses")
# savefig("$HOME/examples/tailless/xp_jun29/comparative_plot_june29_tailless.pdf")

##
using Plots.PlotMeasures
p = plot()#yaxis=:log10)
ylabel!(p, "Range (nautical miles)")
lb = global_variables.R.lb[1]
ub = global_variables.R.ub[1]
for (kc,m)=enumerate(method)
    l = min(32,minimum(length, metric[m]))
    O = zeros(l)
    n = 0
    for k=[1,3,4,7,8,9,10,11,12,13,14,15,16,17,19,20]
        o = obj[m][k][1:l] .* (ub-lb) .+ lb
        O += o
        n += 1
        plot!(p,o, alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, O./n, linewidth=2, label=m[1:3],color=kc)
end
hline!(p, [tailless_optimum.R], linestyle=:dash, label="optimum")
ylims!(lb,ub)
xlabel!("Number of Subspace Evaluations")
title!("Objective Function For 20 Random Initial Guesses")
savefig("$HOME/examples/tailless/xp_jun29/comparative_R_june29_tailless.pdf")

##
p = plot()
ylabel!(p, L"\sqrt{J_{aero}} + \sqrt{J_{struc}}")
for (kc,m)=enumerate(method)
    l = min(32,minimum(length, metric[m]))
    SQJ = zeros(l)
    n = 0
    for k=[1,3,4,7,8,9,10,11,12,13,14,15,16,17,19,20]
        sqj = sum(sqJ[m][k])[1:l]
        SQJ += sqj
        n += 1
        plot!(p,sqj,alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, SQJ./n, linewidth=2, label=m[1:3],color=kc)
end
xlabel!("Number of Subspace Evaluations")
title!("Feasibility For 20 Random Initial Guesses")
savefig("$HOME/examples/tailless/xp_jun29/comparative_sqJ_june29_tailless.pdf")