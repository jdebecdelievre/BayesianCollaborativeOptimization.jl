using Plots
using JLD2
using HouseholderNets
using BayesianCollaborativeOptimization
using Statistics
include("$(pwd())/examples/tailless/tailless.jl")

##

T = Tailless()
solver = SQP(T, λ=1.)
options = SolveOptions(n_ite=25, ini_samples=1, warm_start_sampler=100)
solve(solver, options)
data = map(load_data, T);

##
# z = [0.7541980150409331, 0.6648501320476052, 0.15959821759870685, 0.5888229293125663]
z =  [0.753436520478803, 0.7195552862986361, 0.1511257439801058, 0.6374228055576816]
variables = mergevar((; xcg=aero_global.xcg, D=aero_global.D), aero_local)
idx = indexbyname(variables)
idz = indexbyname(aero_global)
idg = indexbyname(aero_output)
k = upper(variables) - lower(variables) 
b = lower(variables)

# unscale z
kz = upper(aero_global) - lower(aero_global) 
bz = lower(aero_global)

function fun(g, x)
    g.= 0.
    # unscale
    x .*= k
    x .+= b

    # Aero
    D, load_4g, Cl, Cm, Cm_4g, q, Cm_al, Cm_al_4g = aero(x[idx.alpha], x[idx.delta_e], x[idx.twist], x[idx.xcg])

    g[idg.qpos] = -q/W 
    @. g[idg.Cl] = (Cl / 1.45 - 1) / 1000
    g[idg.Cm_al] = Cm_al
    g[idg.Cm_al_4g] = Cm_al_4g 
    g[idg.load_4g] = (2 * W - sum(load_4g))/2/W
    g[idg.Cm] = Cm
    g[idg.Cm_4g] = Cm_4g

    # Compute Loads ROM
    a1 = half_span / (pi * W) * ((load_4g .* sin.(  TLmesh.theta_c)) ⋅ TLmesh.theta_panel_size)
    a3 = half_span / (pi * W) * ((load_4g .* sin.(3*TLmesh.theta_c)) ⋅ TLmesh.theta_panel_size)
    
    # rescale
    x .-= b
    x ./= k

    # objective function
    g[idg.D]  =  (D-bz[idz.D]) / kz[idz.D] - x[idx.D]
    g[idg.a1] = (a1-bz[idz.a1]) / kz[idz.a1]
    g[idg.a3] = (a3-bz[idz.a3]) / kz[idz.a3]
    f = ((x[idx.D]  - z[idz.D] )^2 + (g[idg.a1] - z[idz.a1] )^2+
         (g[idg.a3] - z[idz.a3])^2 + (x[idx.xcg] - z[idz.xcg])^2)
    return f
end
Ng = len(aero_output)
Nx = len(variables)

x0 = ini_scaled(variables)  # starting point
x0[idx.xcg] = z[idz.xcg]
x0[idx.D]   = z[idz.D]

ipoptions = Dict{Any,Any}()
ipoptions["tol"] = 1e-8
ipoptions["max_iter"] = 150
# ipoptions["linear_solver"] = "ma97"
options = SNOW.Options(derivatives=ForwardAD(), solver=SNOPT())#IPOPT(ipoptions))

gg = zeros(Ng)  # starting point
lx = zeros(Nx) # lower bounds on x
ux = ones(Nx) # upper bounds on x
lg = lower(aero_output)
ug = upper(aero_output) # upper bounds on g
xopt, fopt, info = minimize(fun, x0, Ng, lx, ux, lg, ug, options)

# Compute z star
zs          = copy(z)
fun(gg, xopt)
zs[idz.xcg] = xopt[idx.xcg]
zs[idz.D]   = xopt[idx.D]
zs[idz.a1]  = gg[idg.a1]
zs[idz.a3]  = gg[idg.a3]

viol = sum(max(0., lg[i]-gg[i])+ max(0., gg[i]-ug[i]) for i=1:Ng)
@assert (viol < 1e-6) "$viol"
##
using FiniteDiff
gd = zeros(len(idg))
function gdt(gd,x)
    gd = copy(gd)
    fun(gd,x)
    return gd
end
# dg = FiniteDiff.finite_difference_gradient(x->fun(gd,x), x0)
dg = FiniteDiff.finite_difference_jacobian(x->gdt(gd,x), x0)

##
nruns = 7
svd = "examples/tailless"
metric = [get_metrics("examples/tailless/xpu$i",disciplines,zopt,default_obj)[1] for i=1:nruns]
plot(metric, yaxis=:log10)
lm = minimum(length.(metric))
plot!(mean([m[1:lm] for m=metric]), yaxis=:log10,label="mean")
met = [1.813618, 1.326392, 0.467193, 0.332232, 0.251307, 0.203285, 0.179433, 0.143485, 0.113937, 0.097779, 0.085133, 0.071687, 0.061396, 0.055545, 0.054146, 0.052986, 0.049264, 0.047197, 0.040634, 0.038498, 0.035806, 0.030987, 0.028716, 0.024090, 0.022092, 0.020837, 0.020427, 0.019728, 0.017669, 0.016691,]/2
plot!(met, yaxis=:log10,label="gsort")

## Ref result
savedir = "examples/tailless/xpu6"
metric, _, _ = get_metrics(savedir, disciplines, zopt, default_obj)
iteration = 16
@load "$savedir/data.jld2" Z sqJ fsb obj
data = load_data(savedir, disciplines);
datacheck(data)
data = trim_data!(data, iteration)

ensemble = load_ensemble("$savedir/training/$iteration", disciplines)

@load "$savedir/training/$iteration/aero/losses.jld2" loss
@load "$savedir/eic/$iteration/eic.jld2" EIc maxZ iniZ msg ite best stepsize

##
file = "tmp"
options = BCOoptions(
    n_ite = 10, # number of iterations
    ini_samples= 2, # number of initial random samples. 0 to use provided z0
    savedir=savedir, nparticles=12, nlayers=20, lr=0.01,
    αlr=.95, N_epochs=500_000, logfreq=2000, nresample=0, stepsize=10.,
    dropprob=0.02)

##
ipoptions = Dict("print_level"=>5, "file_print_level"=>5, "tol"=>1e-8, "output_file"=>"tmp.txt", 
                "linear_solver"=>"ma97")
##
mkpath("$savedir/eval/aero")
for i=1:length(data.aero.Z)
    aero_subspace(data.aero.Z[i],"$savedir/eval/aero/$i.txt", ipoptions)
end
##
include("tailless.jl")
mkpath("$savedir/eval/struc")
##
sqJs = copy(sqJ.struc)
for i=1:length(data.struc.Z)
    zs = struc_subspace(data.struc.Z[i],"$savedir/eval/struc/$i.txt", ipoptions)
    sqJs[i] = norm(zs - data.struc.Z[i])
    data.struc.sqJ[i] = norm(zs - data.struc.Z[i])
end

##
data = trim_data!(data, iteration-1)
z, eic = maximize_ei(file, ensemble, data, idz, idz_all, options)
                                                                                                                     