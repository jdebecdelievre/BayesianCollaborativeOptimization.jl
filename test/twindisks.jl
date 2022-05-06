using BayesianCollaborativeOptimization
using HouseholderNets

variables = (; 
    A = (; x1=Var(lb=0, ub=1), x2=Var(lb=0, ub=1)),
    B = (; x1=Var(lb=0, ub=1), x2=Var(lb=0, ub=1))
)

cA = [0.5, 0.25]
cB = [0.5, 0.75]
subA(z) = z - [(z[1]-cA[1]), (z[2]-cA[2])].*max(0, sqrt((z[1]-cA[1])^2 +(z[2]-cA[2])^2)-.5)
subB(z) = z - [(z[1]-cB[1]), (z[2]-cB[2])].*max(0, sqrt((z[1]-cB[1])^2 +(z[2]-cB[2])^2)-.5)

subspace = (;
    A = subA,
    B = subB
    )

HOME = pwd()
savedir = "$HOME/tmp"
disciplines = keys(variables)

variables_all = mergevar(values(variables)...)
idz = map(indexbyname, variables)
idz_all = indexbyname(variables_all)

##
data = bco(savedir, variables, subspace);
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
ensembleA = load_ensemble("$savedir/training/10/A/ensemble.jld2");
ensembleB = load_ensemble("$savedir/training/10/B/ensemble.jld2");
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
file = "tmp/minimize/10/eic.jld2"
@load file EIc ite minZ
