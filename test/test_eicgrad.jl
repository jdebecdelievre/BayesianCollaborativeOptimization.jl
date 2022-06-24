using BayesianCollaborativeOptimization
using FiniteDiff
using LinearAlgebra
using Statistics
using SpecialFunctions
using HouseholderNets


edir = (; A="xpu/solver/training/A/ensemble.jld2", B="xpu/solver/training/A/ensemble.jld2")
ensembles = map(e->load_ensemble(e),edir);

##
n = 2
CACHE = [TrainingCacheGrad(net) for net=ensembles.A]
z = rand(2)
predict.([z], CACHE)

## Derivative of ensemble
function Ep(x, C)
    m = Float64[]
    for cache=C
        (; W,b,WZ) = cache
        local z = copy(x)
        for i=1:length(W)-1
            if WZ[i] > 0
                wz = z⋅W[i] + b[i]
                @. z -= 2 * wz * W[i]
            end
        end
        push!(m,W[end] ⋅ z + b[end])
    end
    return (1+erf(-mean(m) / std(m)))/2
end
J = FiniteDiff.finite_difference_gradient(z->Ep(z,CACHE), z)

grad = zeros(2)
cache = eic_cache(ensembles)
p = predict_grad!(grad, z, ensembles.A, h=cache.h.A, dhdx = cache.dhdx.A)
println(norm(J-grad))

## Derivative of EIC
cache = map(e->[TrainingCacheGrad(net) for net=e], ensembles);
map(C->predict.([z], C), cache);

function eic_tmp(x)
    m = map(c->Ep(x,c),cache)
    return -x[1]*prod(m)
end
J = FiniteDiff.finite_difference_gradient(eic_tmp, z)


##
cache = eic_cache(ensembles)
idz = (;
A = [1, 2],
B = [1, 2],
)
best = 0.
stepsize = 100.
grad = zeros(2)
e = eic(z, grad, z, stepsize, default_obj, ensembles, best, idz, cache)
println(norm(J-grad))