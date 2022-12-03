using Pkg
Pkg.activate(joinpath(@__DIR__),"..")
using BayesianCollaborativeOptimization
using LinearAlgebra
using JLD2
using LaTeXStrings
using Statistics
using Parameters
using Printf
##
@consts begin
    idz_twins = (; A=[1,2], B=[1,2])
    cA = [0.5, 0.25]
    cB = [0.5, 0.75]
    opt = [sqrt(-0.25^2+.4^2)+0.5, 0.5]
end

struct TwinDisks <: AbstractProblem end # singleton type
BayesianCollaborativeOptimization.discipline_names(::TwinDisks) = (:A, :B)
BayesianCollaborativeOptimization.indexmap(::TwinDisks) = idz_twins
BayesianCollaborativeOptimization.number_shared_variables(::TwinDisks) = 2
BayesianCollaborativeOptimization.subspace(::TwinDisks, ::Val{:A}, z::AbstractArray,s::String) = (z - ([(z[1]-cA[1]), (z[2]-cA[2])] / norm([(z[1]-cA[1]), (z[2]-cA[2])])).*max(0, sqrt((z[1]-cA[1])^2 +(z[2]-cA[2])^2)-.4)), false
BayesianCollaborativeOptimization.subspace(::TwinDisks, ::Val{:B}, z::AbstractArray,s::String) = (z - ([(z[1]-cB[1]), (z[2]-cB[2])] / norm([(z[1]-cB[1]), (z[2]-cB[2])])).*max(0, sqrt((z[1]-cB[1])^2 +(z[2]-cB[2])^2)-.4)), false
BayesianCollaborativeOptimization.objective_opt(::TwinDisks) = [opt[1]]


#=
sqp = SQP(TwinDisks(), λ=1., tol=1e-6)
options = SolveOptions(n_ite=25, ini_samples=1, warm_start_sampler=100)
# solve(solver, options)
ddir = (; A="xpu/A.jld2", B="xpu/B.jld2")
data = map(load_data, ddir);

##
adm = ADMM(TwinDisks(), ρ=1., tol=1e-6)
options = SolveOptions(n_ite=25, ini_samples=1, warm_start_sampler=100)
# solve(solver, options)
ddir = (; A="xpu/A.jld2", B="xpu/B.jld2")
data = map(load_data, ddir);

##
options = SolveOptions(n_ite=15, ini_samples=1)
bco = BCO(TwinDisks(), N_epochs=100_000, stepsize=1.,tol=1e-6)
# solve(solver, options)
ddir = (; A="xpu/A.jld2", B="xpu/B.jld2")
data = map(load_data, ddir);
##
savedir = "$HOME/examples/twindisks/xp_jun27"
for i=0:19
    # options = SolveOptions(tol=1e-6, n_ite=15, ini_samples=1, warm_start_sampler=i, savedir="$savedir/xpu$i/bco/")
    # solve(bco, options)
    
    options = SolveOptions(n_ite=25, ini_samples=1, warm_start_sampler=i, savedir="$savedir/xpu$i/sqp/")
    solve(sqp, options)
    
    options = SolveOptions(n_ite=25, ini_samples=1, warm_start_sampler=i, savedir="$savedir/xpu$i/admm/")
    solve(adm, options)
end
=#
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

#=
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
=#

## Quadratic RS
function quadratic_fit(z, J, ∇J)
    npt = length(z)
    Z = ones(3*npt,6)
    Y = ones(3*npt)    
    for i=1:npt
        j = 3*(i-1)
        # z
        Z[j+1, 2:3]    = z[i]
        @. Z[j+1, 4:5] = z[i]^2
        Z[j+1, 6]      = prod(z[i])
        Y[j+1] = J[i]

        # ∇z[1]
        Z[j+2,1]     = 0.
        Z[j+2, 2:3]  = [1.,0]
        Z[j+2, 4]    = z[i][1]
        Z[j+2, 5]    = 0.
        Z[j+2, 6]    = z[i][2]
        Y[j+2] = ∇J[i][1]

        # ∇z[2]
        Z[j+3,1]     = 0.
        Z[j+3, 2:3]  = [0.,1]
        Z[j+3, 4]    = 0.
        Z[j+3, 5]    = z[i][2]
        Z[j+3, 6]    = z[i][1]
        Y[j+3] = ∇J[i][2]
    end
    V = (Z'*Z+1e-14*I) \ (Z' * Y)
    f(z)= V[1] + V[2]*z[1] + V[3]*z[2] + V[4]*z[1]^2 + V[5]*z[2]^2 + V[6]*z[1]*z[2]
    # @show f.(z) J Z
    # V = [0., 0., 0., 1., 1., 0.]
    return V
end     

function solve_quad(z0, objective, Jrs, Δ, k)
    n = length(z0)
    lower = zeros(n)
    upper =  ones(n)

    function fun(z, grad)
        grad .= 0.
        g = length(grad) > 0
        g ? (f = objective(z, grad)) : (f = objective(z))
        f -= (z-z0)'*(z-z0) / (1. *Δ)^2
        for G = Jrs
            f += k * (G[1] + G[2]*z[1] + G[3]*z[2] + G[4]*z[1]^2 +
                G[5]*z[2]^2 + G[6]*z[1]*z[2])
            if g
                grad[1] += k*(G[2] + 2 * G[4] * z[1] + G[6] * z[2])
                grad[2] += k*(G[3] + 2 * G[5] * z[2] + G[6] * z[1])
            end
        end
        g && (@. grad -= 2 * (z-z0) / (1. * Δ)^2)
        return f
    end

    # Use SLSQP
    opt = Opt(:LD_SLSQP, n)
    opt.lower_bounds = lower
    opt.upper_bounds = upper
    opt.xtol_rel = 1e-8
    opt.maxeval = 2500
    opt.max_objective = fun
    (maxf,maxz,ret) = optimize(opt, z0)

    if norm(maxz - z0) < 1e-5
        return solve_quad(z0, objective, Jrs, Δ, 10. * k)
    end

    # @show ret
    return maxf, maxz
end

function qrs_co(zref, objective, subspace, ncycles, savefile)
    k = -200.
    n = length(zref)
    npt = ceil((n+1)/2)
    
    zsref = map(s->s(zref), subspace)
    Jref = map(zs->(zref-zs)⋅(zref-zs), zsref)
    fref = objective(zref) + k * sum(Jref)
    
    # Store
    Zr = map(zs->[copy(zs) for i=1:1+npt*ncycles], zsref)
    Zs = map(zs->[copy(zs) for i=1:1+npt*ncycles], zsref)

    # b
    Δ = 0.2
    Δmin = 0.01
    Δmax = 0.3 
    
    @printf "%6s  %8s %8s %8s %8s %8s %8s\n" "icycle" "fref" "Jref" "f" "J" "Δ" "|z-zref|"
    @printf "%6d  %.2e %.2e %.2e %.2e %.2e %.2e\n" 0 fref sum(Jref) fref sum(Jref) Δ 0.
    i = 2
    evalcount = 1
    for icycle = 1:ncycles
        # c
        zi = normalize(randn(2))
        # if sum(Jref) > 1e-6 # if zref is infeasible, we can use itself + its projection
        #     Z = map(zs->[zs, zref + Δ * rand() * zi],zsref)
        # else 
        #     Z = map(s->[zref, zref + Δ * rand() * zi], zsref) # same pair for all disciplines
        # end
        Z = map(s->[zref, zref + Δ * rand() * zi], zsref) # same pair for all disciplines
        evalcount += 1
        
        # d
        Zstar = map((Z,s)->s.(Z),Z,subspace)
        ∇J = map((z, zs)->(z-zs), Z, Zstar)
        J = map(dJ->dJ.⋅dJ, ∇J)
        # @show ∇J
        
        # Store values
        for d=keys(Zstar)
            Zr[d][i] .= Z[d][1]
            Zr[d][i+1] .= Z[d][2]
            Zs[d][i] .= Zstar[d][1]
            Zs[d][i+1] .= Zstar[d][2]
        end
        i += 2
        
        # e
        Jrs = map((Z, j,dj)->quadratic_fit(Z, j, dj), Z, J, ∇J)

        # f
        frs, z = solve_quad(zref, objective, Jrs, Δ, k)
        # @show z frs

        # g
        zs = map(s->s(z), subspace)
        J = map(zs->(z-zs)⋅(z-zs), zs)
        f = objective(z) + k * sum(J)

        # i 
        nzz = norm(zref - z)
        # if nzz > Δ
        #     Δ /= 2
        # else
        #     Δ *= 2
        # end

        # h
        if f > fref
                zref .= z
                zsref = zs
                fref = f
                Jref = J
                Δ = min(2*Δ,Δmax)
        else
                Δ = max(Δ/2,Δmin)
        end
        @printf "%6d  %.2e %.2e %.2e %.2e %.2e %.2e\n" icycle fref sum(Jref) f sum(J) Δ nzz
    end
    Zstar = Zs
    Z = Zr
    @save savefile Z Zstar
end
