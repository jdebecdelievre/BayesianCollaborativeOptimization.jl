using BayesianCollaborativeOptimization
using LinearAlgebra
using Statistics
using JLD2
using Plots
HOME = pwd()
include("$(pwd())/examples/tailless/tailless.jl")
T = Tailless()

gmet = [ 0.9395783514629933, 0.6644157664688632, 0.24305338004907254, 0.17855792712535565, 0.13656728089934528, 0.11159422426723604, 0.09911113515752532, 0.08001423078816536, 0.06527614788311345, 0.05723934247593221, 0.05138013348777313, 0.04522013720308668, 0.04036728301607296, 0.03746641125711485, 0.03481941384248512, 0.03376239609724129, 0.031230444212835818, 0.029693000039475236, 0.02658597813187403, 0.025020945003832163, 0.02379023622151321, 0.021621746143340434, 0.020600042846018886, 0.018518185621759747, 0.013758299082697554, 0.013193317834839841, 0.012863455300141397, 0.012502872281513039, 0.011418526185243106, 0.010792017807775733, 0.010674422951337281]

##
disciplines = discipline_names(T)
method = ["bco22","sqp"]
metric = Dict{Any,Any}()
for m=method
    M = map(i->get_metrics(T, "$HOME/examples/tailless/xp_jun29/$m/xpu$i"),1:20)
    metric[m] = first.(M)
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
i = 8
data = load_data("xpu$i",T)
@load "xpu$i/obj.jld2" Z sqJ obj fsb
datacheck(data)