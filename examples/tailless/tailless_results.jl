using Pkg
Pkg.activate("$(@__DIR__)/../.")

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

method = ["bco"]
nruns = 20
pad = true
metric = Dict{Any,Any}()
obj = Dict{Any,Any}()
dobj = Dict{Any,Any}()
sqJ = Dict{Any,Any}()
for m=method
    #
    M = map(i->get_metrics(T, "$HOME/examples/tailless/xp2_feb14_23/$m/xpu$i"),1:nruns)
    metric[m] = first.(M)
    sqJ[m] = last.(M)
    obj[m] = getindex.(M,[2])
    dobj[m] = getindex.(M,[3])

    #
    if pad
        l = maximum(length, metric[m])
        metric[m] = [[s;ones(l-length(s))*s[end]] for s=metric[m]]
        obj[m]    = [[s;ones(l-length(s))*s[end]] for s=obj[m]]
        dobj[m]   = [[s;ones(l-length(s))*s[end]] for s=dobj[m]]
        sqJ[m]    = [map(s->[s; ones(l-length(s))*s[end]],ss) for ss=sqJ[m]]
    end
end

## Saving in format compatible with phd thesis
sqJaero = hcat((sqJ["bco"][i].aero for i=1:nruns)...)
sqJstruc = hcat((sqJ["bco"][i].struc for i=1:nruns)...)
JLD2.save("$HOME/examples/tailless/metric_hnet.jld2", 
    "dobj_srt", hcat(dobj["bco"]...),
    "sqJaero_srt", sqJaero,
    "sqJstruc_srt", sqJstruc,
    "metric",hcat(metric["bco"]...))
##
p = plot(yaxis=:log10)
for m=method
    l = min(32,minimum(length, metric[m]))
    plot!(p,mean([m_[1:l] for m_=metric[m]]),label=m)
    # plot!(p,([m_[1:l] for m_=metric[m]]),label=m)
end
plot!(gmet,label="gpsort")
xlabel!("Number of Subspace Evaluations")
ylabel!("Metric")
title!("Average Metric For 20 Random Initial Guesses")
# savefig("$HOME/examples/tailless/xp_jul21/comparative_plot_jul21_tailless.pdf")

##
method = ["bco","sqp"]
leg = ["Bayesian CO", "SQP"]
using Plots.PlotMeasures
p = plot()#yaxis=:log10)
ylabel!(p, "Range (nautical miles)")
lb = global_variables.R.lb[1]
ub = global_variables.R.ub[1]
for (kc,m)=enumerate(method)
    l0  = (m[1:3] == "sqp") ? 2 : 1
    l = min(32,minimum(length, metric[m])) + l0 - 1
    O = zeros(l-l0+1)
    n = 0
    for k=1:20
        o = obj[m][k][l0:l] .* (ub-lb) .+ lb
        O += o
        n += 1
        plot!(p,o, alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, O./n, linewidth=2, label=leg[kc],color=kc)
end
hline!(p, [tailless_optimum.R], linestyle=:dash, label="optimum")
ylims!(lb,ub)
xlabel!("Number of Subspace Evaluations")
title!("Objective Function For 20 Random Initial Guesses")
# savefig("$HOME/examples/tailless/comparative_R_jul21_tailless.pdf")

##
method = ["bco","sqp"]
leg = ["Bayesian CO", "SQP"]
using Plots.PlotMeasures
p = plot(yaxis=:log10)
ylabel!(p, L"|R-R^*|/(R^{ub}-R^{lb})")
lb = global_variables.R.lb[1]
ub = global_variables.R.ub[1]
for (kc,m)=enumerate(method)
    l0  = (m[1:3] == "sqp") ? 2 : 1
    l = min(32,minimum(length, metric[m])) + l0 - 1
    O = zeros(l-l0+1)
    n = 0
    for k=1:20
        o = dobj[m][k][l0:l]
        O += o
        n += 1
        plot!(p,o, alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, O./n, linewidth=2, label=leg[kc],color=kc)
end
# hline!(p, [tailless_optimum.R], linestyle=:dash, label="optimum")
ylims!(1e-3, 0.5)
xlabel!("Number of Subspace Evaluations")
title!("Objective Error For 20 Random Initial Guesses")
savefig("$HOME/examples/tailless/comparative_dR_jul21_tailless.pdf")

##
p = plot(yaxis=:log10)
ylabel!(p, L"\sqrt{J_{aero}} + \sqrt{J_{struc}}")
for (kc,m)=enumerate(method)
    l0  = (m[1:3] == "sqp") ? 2 : 1
    l = min(32,minimum(length, metric[m])) + l0-1
    SQJ = zeros(l-l0+1)
    n = 0
    for k=1:20
        sqj = sum(sqJ[m][k])[l0:l]
        SQJ += sqj
        n += 1
        plot!(p,sqj,alpha=.5, linewidth=.5,label="",color=kc)
    end
    plot!(p, SQJ./n, linewidth=2, label=m[1:3],color=kc)
end
ylims!(1e-4, 1.)
xlabel!("Number of Subspace Evaluations")
title!("Feasibility For 20 Random Initial Guesses")
# savefig("$HOME/examples/tailless/comparative_sqJ_jul21_tailless.pdf")
##
# @load "examples/tailless/xpu7tmp/solver/eic/29/eic.jld2"
@load "examples/tailless/xpu7tmp/obj.jld2"

solver = BCO(Tailless(), 
N_epochs=500_000, stepsize=10.,
dropprob= 0.02, 
αlr=0.97, 
nlayers=40, nparticles=6, ntrials=2, training_tol=1e-3, tol=1e-3)
# solver = SQP(Tailless(), tol=1e-3)
# solver = ADMM(Tailless())
savedir = "examples/tailless/xpu7tmp"
options = SolveOptions(n_ite=30, ini_samples=1, 
warm_start_sampler=1, 
savedir = savedir)

data = load_data(savedir, T);
z, Zd, eic_max = BayesianCollaborativeOptimization.get_new_point(29, solver, data, "$savedir/solver")