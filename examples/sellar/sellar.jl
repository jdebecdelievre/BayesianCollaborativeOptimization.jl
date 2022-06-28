using BayesianCollaborativeOptimization
using LinearAlgebra
import SNOW
using JLD2
using Parameters

@consts begin
    V = (;
        z = Var(lb=-10., ub=10.),
        x = Var(lb=0., ub=10., N=2),
        y1 = Var(lb=8.,ub=100.),
        y2 = Var(lb=0.,ub=24.)
    )
    lb = lower(V)
    ub = upper(V)
    variables = (; d1=V, d2=V)
    idz = (; d1=collect(1:5), d2=collect(1:5))
    idx = indexbyname(V)

    vopt = (; z = -2.8014207800091704, x = [0.07572497323210897, 0.10214619566441839], y1 = 8.001, y2 = 0.12915237776863955)
    zopt = ([-2.8014207800091704, 0.07572497323210897, 0.10214619566441839, 8.001, 0.12915237776863955] - lower(V)) ./ (upper(V) - lower(V))
end

##

struct Sellar <: AbstractProblem end
BayesianCollaborativeOptimization.discipline_names(::Sellar) = (:d1, :d2)
BayesianCollaborativeOptimization.indexmap(::Sellar) = idz
BayesianCollaborativeOptimization.number_shared_variables(::Sellar) = 5
BayesianCollaborativeOptimization.objective_lowerbound(::Sellar) =  - (10. ^2 + 10. + 100. + exp(-24))

function BayesianCollaborativeOptimization.objective(::Sellar, z::AbstractArray, grad=nothing)
    v = unscale_unpack(z, idx, V)
    obj = -(v.x[1]^2 + v.x[2] + v.y1 + exp(-v.y2))
    if typeof(grad) <: AbstractArray
        grad          .= 0.
        grad[idx.x[1]] = -2*v.x[1]
        grad[idx.x[2]] = -1
        grad[idx.y1]   = -1
        grad[idx.y2]   = + exp(-v.y2)
        @. grad *= (ub-lb)
    end
    return obj
end

function BayesianCollaborativeOptimization.subspace(::Sellar, ::Val{:d1},  z0::AbstractArray, filename::String)
    
    ipoptions=Dict{Any,Any}("print_level"=>0)
    k = upper(V)-lower(V)

    function fun(g, df, dg, z)
        v = unscale_unpack(z,idx,V)
        
        g[1] = v.y1 - (v.z^2 + v.x[1] + v.x[2]-0.2*v.y2)
        dg[1,idx.y1] = 1.
        dg[1,idx.z]  = -2*v.z
        @. dg[1,idx.x]  = -1.
        dg[1,idx.y2] = 0.2
        @. dg[1,:] *= k

        df[:] = 2 * (z-z0)
        return (df ⋅ df) / 4
        # return (z-z0) ⋅ (z-z0)
    end
    
    Nx = len(V)
    lx = zeros(Nx) # lower bounds on x
    ux = ones(Nx) # upper bounds on x
    Ng = 1
    lg = [0.]
    ug = [0.]
    
    options = SNOW.Options(derivatives=SNOW.UserDeriv(), solver=SNOW.IPOPT(ipoptions), sparsity=SNOW.DensePattern())
    xopt, fopt, info = SNOW.minimize(fun, copy(z0), Ng, lx, ux, lg, ug, options)
    return copy(xopt)
end

function BayesianCollaborativeOptimization.subspace(::Sellar, ::Val{:d2},  z0::AbstractArray, filename::String)
    
    ipoptions=Dict{Any,Any}("print_level"=>0)
    k = upper(V)-lower(V)

    function fun(g, df, dg, z)
        v = unscale_unpack(z,idx,V)
        
        g[1] = v.y2 - (sqrt(v.y1) + v.z + v.x[2])
        dg[1,idx.y2]    = 1.
        dg[1,idx.y1]    = -1/(2*sqrt(v.y1))
        dg[1,idx.z]     = -1.
        dg[1,idx.x[2]]  = -1.
        @. dg[1,:] *= k

        df[:] = 2 * (z-z0)
        return (df ⋅ df) / 4
    end

    Nx = len(V)
    lx = zeros(Nx) # lower bounds on x
    ux = ones(Nx) # upper bounds on x
    Ng = 1
    lg = [0.]
    ug = [0.]
    
    options = SNOW.Options(derivatives=SNOW.UserDeriv(), solver=SNOW.IPOPT(ipoptions))
    xopt, fopt, info = SNOW.minimize(fun, copy(z0), Ng, lx, ux, lg, ug, options)
    return copy(xopt)
end

function sellar_aao(z0)
    prb = Sellar()

    function fun(g,z)
        v = unscale_unpack(z,idx,V)
        g[1] = v.y1 - (v.z^2 + v.x[1] + v.x[2]-0.2*v.y2)
        g[2] = v.y2 - (sqrt(v.y1) + v.z + v.x[2])
        return -objective(prb, z)
    end

    Nx = len(V)
    lx = zeros(Nx) # lower bounds on x
    ux = ones(Nx) # upper bounds on x
    Ng = 2
    lg = [0.,0.]
    ug = [0.,0.]
    
    options = SNOW.Options(derivatives=SNOW.CentralFD(), solver=SNOW.IPOPT(), sparsity=SNOW.DensePattern())
    xopt, fopt, info = SNOW.minimize(fun, z0, Ng, lx, ux, lg, ug, options)
    return  objective(prb, xopt), unscale_unpack(xopt, idx, V)
end

##
faao, vaao = sellar_aao(ini_scaled(V))
# options = SolveOptions(tol=1e-6, n_ite=15, ini_samples=1, warm_start_sampler=i, savedir="$savedir/xpu$i/bco/")
# solve(bco, options)
solver = BCO(Sellar(), N_epochs=100_000, stepsize=1.)
options = SolveOptions(n_ite=15, ini_samples=1, warm_start_sampler=100, tol=1e-6)
obj, sqJ, fsb, Z = solve(solver, options)
v = unscale_unpack(Z[end],idx,V)

## Load
ite = 2
data = load_data("xpu",(:d1,:d2))
ensembles = load_ensemble("xpu/solver/training/$ite",(:d1,:d2));
@load "xpu/solver/eic/$ite/eic.jld2" EIc maxZ iniZ msg ite best
@load "xpu/data.jld2" Z sqJ fsb