using BayesianCollaborativeOptimization
using LinearAlgebra
using Statistics
using JLD2
using Plots
HOME = pwd()
include("$HOME/examples/sellar/sellar.jl")

##
sl = Sellar()
disciplines = discipline_names(sl)

M = map(i->get_metrics(sl, "xpu$i", zopt),1:20)
metric = first.(M)
obj = getindex.(M, [2])
sqJ = last.(M)
plot(metric, yaxis=:log10)

##
solver = ADMM(Sellar(), ρ=.1)
for i=1:20
    options = SolveOptions(n_ite=25, ini_samples=1, warm_start_sampler=i, tol=1e-6, savedir="xpu$i")
    obj, sqJ, fsb, Z = solve(solver, options, terminal_print=false)
end

##
solver = SQP(Sellar())
for i=1:20
    options = SolveOptions(n_ite=35, ini_samples=1, warm_start_sampler=i, tol=1e-6, savedir="xpu$i")
    obj, sqJ, fsb, Z = solve(solver, options, terminal_print=false)
end

##
methods = ["bco", "admm", "sqp"]
p = plot(yaxis=:log10)
for tl = methods
    @show tl
    
    # Compute metrics
    M = map(i->get_metrics(sl, "examples/sellar/xp_jun28/$tl/xpu$i", zopt),1:20)
    metric = first.(M)
    
    # Pad
    l = maximum(length, metric)
    @show l
    for i=1:20
        metric[i] = [metric[i]; ones(l-length(metric[i]))*metric[i][end]]
    end

    plot!(p,mean(metric),label=tl)
end


##
methods = ["bco", "admm"]
for tl = methods
    for i=1:20
    fld = "examples/sellar/xp_jun28/$tl/xpu$i"
    data = load_data(fld, sl)
    Z = data.d1.Z
    fsb = map(d->d.fsb,data)
    sqJ = map(d->d.sqJ,data)
    obj = load("$fld/obj.jld2","obj")["obj"]
    @save "$fld/obj.jld2" Z obj sqJ fsb
    end
end