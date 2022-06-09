using BayesianCollaborativeOptimization
using HouseholderNets
using LinearAlgebra
using JLD2
using NLopt

variables = (; 
    A = (; x1=Var(lb=0, ub=1), x2=Var(lb=0, ub=1)),
    B = (; x1=Var(lb=0, ub=1), x2=Var(lb=0, ub=1))
)

cA = [0.5, 0.25]
cB = [0.5, 0.75]
subA(z) = z - ([(z[1]-cA[1]), (z[2]-cA[2])] / norm([(z[1]-cA[1]), (z[2]-cA[2])])).*max(0, sqrt((z[1]-cA[1])^2 +(z[2]-cA[2])^2)-.4)
subB(z) = z - ([(z[1]-cB[1]), (z[2]-cB[2])] / norm([(z[1]-cB[1]), (z[2]-cB[2])])).*max(0, sqrt((z[1]-cB[1])^2 +(z[2]-cB[2])^2)-.4)
opt = [sqrt(-0.25^2+.4^2)+0.5, 0.5]

subspace = (;
    A = subA,
    B = subB
    )

HOME = pwd()
savedir = "$HOME/test/twindisks"
disciplines = keys(variables)

variables_all = mergevar(values(variables)...)
idz = map(indexbyname, variables)
idz_all = indexbyname(variables_all)

##
for i=0:10
        savedir = "$HOME/test/twindisks_jun9_5/$i"
        options = BCOoptions(
                n_ite = 10, # number of iterations
                ini_samples= 2, # number of initial random samples. 0 to use provided z0
                savedir=savedir, nparticles=12, nlayers=20, lr=0.01,
                warm_start_sampler=2*i,
                αlr=.5, N_epochs=500_000, logfreq=2000, nresample=0,
                dropprob=0.01)
        data = bco(variables, subspace, options)
end

##
lo = Vector{Float64}[]
lf = Vector{Float64}[]
dz = Vector{Float64}[]
for i=0:10
        savedir = "$HOME/test/twindisks_jun9_5/$i"
        data    = NamedTuple{disciplines}(map(s->load_data("$savedir/$s.jld2"), disciplines))
        push!(lo, abs.(first.(data.A.Z) .- opt[1]))
        push!(lf, data.A.sqJ + data.B.sqJ)
        push!(dz, norm.(data.A.Z .- [opt]))
end
@save "$HOME/test/twindisks_jun9_5/metrics.jld2" lo lf dz
##
function reorder!(ls)
        for l=ls
                for i=2:length(l)
                        l[i] = min(l[i-1],l[i])
                end
        end
        return ls
end
lof = lo + lf
reorder!(lof)
reorder!(lf)
reorder!(lo)
##
# ite
# data     = NamedTuple{disciplines}(map(s->load_data("$savedir/$s.jld2"), disciplines))
# ensemble = NamedTuple{disciplines}(map(s->load_ensemble("$savedir/training/3/$s/ensemble.jld2"), disciplines));

## 
ite      = 1
savedir  = "$HOME/test/twindisks_jun9_5/1"
edir     = NamedTuple{disciplines}(map(s->"$savedir/training/$ite/$s/ensemble.jld2", disciplines))
ensemble = map(load_ensemble, edir);
ddir     = NamedTuple{disciplines}(map(s->"$savedir/$s.jld2", disciplines))
data     = map(load_data,ddir)
data     = trim_data!(data, ite)
@load "$savedir/eic/$ite/eic.jld2" EIc maxZ iniZ best stepsize
##
i = argmax(EIc)
eic(z) = max(0.,z[1]-best)*(ensemble.A(z) * ensemble.B(z)) * exp(- (z-iniZ[i])⋅(z-iniZ[i]) / stepsize[i]^2)
showpr(data.A.Z,
eic,
data.A.fsb.*data.B.fsb,[0.,1])
scatter!([iniZ[i][1]],[iniZ[i][2]])
scatter!([maxZ[i][1]],[maxZ[i][2]])
##
data = trim_data!(data, ite-1)
BayesianCollaborativeOptimization.maximize_ei("$savedir/tmp4/", ensemble, data, idz, idz_all, options)
file = "$savedir/tmp4/eic.jld2"
@load file EIc maxZ best iniZ stepsize
##
showfn(data.A.Z,
        ensemble.A[5],
        data.A.fsb,[0,1])
##
showpr(data.A.Z,
        ensemble.A,
        data.A.fsb,[0,1])
##
showpr(data.B.Z,
        ensemble.B,
        data.B.fsb,[0,1])

## Regular CO
function solve_co(z0, neval=100)
        k = 150.
        Z = [copy(z0) for z=1:neval]
        i = 1
        function fun(z, grad)
                Z[i] .= z
                i += 1
                zA = subA(z)
                zB = subB(z)
                if len(grad) > 0
                        grad .= [1., 0.] - 2*k*((z-zA) + (z-zB))
                end
                return z[1] - ((z-zA)⋅(z-zA) + (z-zB)⋅(z-zB))*k
        end
        opt = Opt(:LD_SLSQP, 2)
        opt.lower_bounds = [0., 0.]
        opt.upper_bounds = [1., 1.]
        opt.xtol_rel = 1e-6
        opt.maxeval = neval
        opt.max_objective = fun
        (maxf,maxz,ret) = optimize(opt, z0)
        return Z
end
for i=0:10
        savedir  = "$HOME/test/twindisks_jun9_5/$i"
        ddir     = NamedTuple{disciplines}(map(s->"$savedir/$s.jld2", disciplines))
        data     = map(load_data,ddir)
        Z = solve_co(data.A.Z[1])
        sqJA = norm.(subA.(Z) .- Z)
        sqJB = norm.(subB.(Z) .- Z)
        lo = [abs(z[1]-opt[1]) for z = Z]
        lf = sqJA + sqJB
        lof = lo + lf
        @save "$savedir/reg_co.jld2" Z sqJA sqJB lo lf lof
end

lof = map(i->load("$HOME/test/twindisks_jun9_5/$i/reg_co.jld2","lof"),0:10)
reorder!(lof)
plot(lof, yaxis=:log10)

##
Z = map(i->load("$HOME/test/twindisks_jun9_5/$i/reg_co.jld2","Z"),0:10)
dz_co = [norm.(z.-[opt]) for z=Z]
reorder!(dz_co)
reorder!(dz)
plot(dz, yaxis=:log10)
plot!(dz_co, yaxis=:log10)

##
using Statistics
plot(mean(dz), yaxis=:log10, formatter=:plain)
plot!(mean(dz_co)[1:12])
xticks!(1:12)