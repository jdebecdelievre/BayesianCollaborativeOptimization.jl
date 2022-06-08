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
        n_ite = 10, # number of iterations
        ini_samples= 2, # number of initial random samples. 0 to use provided z0
        savedir=savedir, nparticles=12, nlayers=20, lr=0.01,
        αlr=.5, N_epochs=100000, logfreq=1000, nresample=0
)
##
data = bco(variables, subspace, options);
##
ite      = 4
edir     = NamedTuple{disciplines}(map(s->"$savedir/training/$ite/$s/ensemble.jld2", disciplines))
ensemble = map(load_ensemble, edir);
ddir     = NamedTuple{disciplines}(map(s->"$savedir/$s.jld2", disciplines))
data     = map(load_data,ddir)
data     = trim_data!(data, ite)
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
file = "$savedir/minimize/$ite/eic.jld2"
@load file EIc maxZ best iniZ
i = argmax(EIc)
eic(z) = max(0.,z[1]-best)*(ensemble.A(z) * ensemble.B(z)) * exp(-10*(z-iniZ[i])⋅(z-iniZ[i]))
showpr(data.A.Z,
eic,
data.A.fsb.*data.B.fsb,[0.,1])
scatter!([iniZ[i][1]],[iniZ[i][2]])
##
data = trim_data!(data, ite-1)
maximize_ei("$savedir/tmp4/", ensemble, data, idz, idz_all, options)
##
file = "$savedir/tmp4/eic.jld2"
@load file EIc maxZ best iniZ