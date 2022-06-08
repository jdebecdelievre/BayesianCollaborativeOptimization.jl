using BayesianCollaborativeOptimization
using HouseholderNets
using LinearAlgebra
using JLD2

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
options = BCOoptions(
        n_ite = 2, # number of iterations
        ini_samples= 2, # number of initial random samples. 0 to use provided z0
        savedir=savedir, nparticles=12, nlayers=20, lr=0.01,
        αlr=.5, N_epochs=100000, logfreq=1000, nresample=0
)
##
data = bco(variables, subspace, options);
##
edir     = NamedTuple{disciplines}(map(s->"$savedir/training/1/$s/ensemble.jld2", disciplines))
ensemble = map(load_ensemble, edir);
ddir     = NamedTuple{disciplines}(map(s->"$savedir/$s.jld2", disciplines))
data     = map(load_data,ddir)
##
showfn(data.A.Z,
        ensemble.A[1],
        data.A.fsb,[0,1])
##
showpr(data.A.Z,
        ensemble.A,
        data.A.fsb,[0,1])
##
showpr(data.B.Z,
        ensemble.B,
        data.B.fsb,[0,1])

##
ite = 2
data = NamedTuple{disciplines}(map(d->load_data("$savedir/$d.jld2"), disciplines))
ensembleA = load_ensemble("$savedir/training/$ite/A/ensemble.jld2");
ensembleB = load_ensemble("$savedir/training/$ite/B/ensemble.jld2");
data = trim_data!(data, ite)
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
file = "$savedir/minimize/$ite/eic.jld2"
@load file EIc minZ best
eic(z) = -max(0.,(best+z[1])*(ensembleA(z)*ensembleB(z)))#*exp(-(z-data.A.Z[end-1])⋅(z-data.A.Z[end-1]))
showpr(data.A.Z,
        eic,
        data.A.fsb,[0.,1])
##

##
savedir = "tmp/"
ensembles = (; A=ensembleA, B=ensembleB)
minimize_ei(savedir, ensembles, data, idz, idz_all)