using HouseholderNets
using StaticArrays
using JLD2
using Plots 

@load "examples/tailless/xp/struc.jld2" Z Zs sqJ fsb ite
data = load_data("examples/tailless/xp/struc.jld2");

nparticles=3
nlayers = 20
n = 5

##
ensemble = learn_feasible_set(data, "examples/subspace",  nparticles,  nlayers)

ifs = .!data.fsb
sum((ensemble[1].(data.Z[ifs]) - data.sqJ[ifs]).^2)

## Home made

# Prep data
ifs = .!data.fsb
X = [SVector{n}.(data.Z); SVector{n}.(data.Zs[ifs])]
Y = [data.sqJ; 0*data.sqJ[ifs]]
fsb = [data.fsb; data.fsb[ifs]]

# Infer value of feasible points
for i=eachindex(Y)
    idX = idXs = 1
    dX = dXs = Inf
    if fsb[i]
        # Find closest pair of infeasible points
        for j=eachindex(data.Z)
            if !data.fsb[j]
                dX̄ = norm(data.Z[j] -X[i])
                if (dX̄<dX)
                    idX = j
                    dX = dX̄
                end
                dX̄s = norm(data.Zs[j] -X[i])
                if dX̄s<dXs   
                    idXs = j; 
                    dXs = dX̄s
                end
            end
        end
        # Assign to distance to infeasible point, or distance to projection along gradient of function
        if isfinite(dX)
            if dX<dXs
                Y[i] = data.sqJ[idX]-dX
            else
                Y[i] = ((data.Z[idXs]-data.Zs[idXs])⋅(X[i]-data.Zs[idXs])) / data.sqJ[idXs]
            end
        else # no feasible points
            Y[i] = -10. # will be corrected by negative boundary check below
        end

        # Make sure boundary is not pulled into being feasible
        mindist_boundary = minimum(min(abs(x-bounds[1]),abs(x-bounds[2])) for x=X[i])
        Y[i] = max(Y[i], -mindist_boundary-tol)
    end
end

##
mkpath(savedir)
bounds=[0.,1.]
net = HouseholderNet(nlayers,n)
SGD(lr=0.01, αlr=.999, N_epochs=3000000, logfreq=100, bounds=bounds)
cache = TrainingCache(net)
initialization!(net, data.Z,data.Zs, Ntrials=100000, rng=opt.rng, bounds=bounds)

# train
while mse_update(X, Y, cache, opt, verbose=true); end

# save training history
hist = historydf(opt.hist)
p = plot(hist.loss, yscale=:log10, label="")

##
net = ensemble[1];
nt = prune(Z,net);
