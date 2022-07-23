using Distributed
using Pkg
Pkg.activate("/home/adgboost/.julia/dev/BayesianCollaborativeOptimization/")
addprocs(12, exeflags="--project=$(Base.active_project())")

@everywhere begin 
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


    function run(i)
        savedir = "$HOME/test/twindisks_jun9_8/$i"
        options = SolveOptions(
                n_ite = 35, # number of iterations
                ini_samples= 2, # number of initial random samples. 0 to use provided z0
                savedir=savedir, nparticles=12, nlayers=0, lr=0.01,
                warm_start_sampler=2*i,
                αlr=.5, N_epochs=500_000, logfreq=2000, nresample=0,
                dropprob=0.02,tol=1e-6)
        data = bco(variables, subspace, options)
    end

end


pmap(i->run(i),0:11)

using LinearAlgebra
using JLD2
using BayesianCollaborativeOptimization

HOME = pwd()
disciplines = (:A, :B)
opt = [sqrt(-0.25^2+.4^2)+0.5, 0.5]

lo = Vector{Float64}[]
lf = Vector{Float64}[]
dz = Vector{Float64}[]
for i=0:11
        local savedir = "$HOME/test/twindisks_jun9_8/$i"
        data    = NamedTuple{disciplines}(map(s->load_data("$savedir/$s.jld2"), disciplines))
        push!(lo, abs.(first.(data.A.Z) .- opt[1]))
        push!(lf, data.A.sqJ + data.B.sqJ)
        push!(dz, norm.(data.A.Z .- [opt]))
end
@save "$HOME/test/twindisks_jun9_8/metrics.jld2" lo lf dz