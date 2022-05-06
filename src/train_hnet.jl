function learn_feasible_set(data, savedir,  nparticles::Int64=3,  nlayers::Int64=10)
    mkpath(savedir)
    bounds=[0.,1.]
    n = size(data.Z[1],1)
    ensemble = [HouseholderNet(nlayers,n) for _=1:nparticles]
    optim = SGD(lr=0.01, αlr=.98, N_epochs=30000, logfreq=100, bounds=bounds)
    trainingcaches = [TrainingCache(net) for net in ensemble]
    p = plot()

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
            Y[i] = max(Y[i], -mindist_boundary)
        end
    end

    # Train nparticles nets
    for np=1:nparticles
        # reset optimizer
        opt = resetopt(optim)
        net = ensemble[np]
        cache = trainingcaches[np]

        # initialize
        initialization!(net, data.Z,data.Zs, Ntrials=100000, rng=opt.rng)

        # train
        while mse_update(X, Y, cache, opt, verbose=false); end

        # save training history
        hist = historydf(opt.hist)
        plot!(p,hist.loss, yscale=:log10, label="")
        
        # save training history
        CSV.write("$savedir/traininghistory$np.csv", hist)
    end
    save_ensemble("$savedir/ensemble.jld2", ensemble)
    savefig(p, "$savedir/trainingcurves.pdf")
    return ensemble
end

function save_ensemble(filename, ensemble)
    @assert (split(filename, ".")[end] == "jld2") "Filename extension must be jld2"
    # StaticVectors prevent serialization
    ensemble_ = [HouseholderNet(Vector.(e.W), e.b) for e=ensemble];
    save(filename, "ensemble", ensemble_)
end

function load_ensemble(filename)
    @assert (split(filename, ".")[end] == "jld2") "Filename extension must be jld2"
    ensemble = load(filename)["ensemble"]
    # for i = eachindex(ensemble_)
    #     for j = eachindex(ensemble[i].W)
    #         ensemble[i].W[j] .= ensemble_[i].W[j]
    #     end
    #     ensemble[i].b .= ensemble_[i].b
    # end
    return ensemble
end


# function train_ensemble(X,Xs,fsb, folder; nparticles::Int64=10,  nlayers::Int64=size(X,1))
#     n = size(X[1],1)
#     ensemble = [HouseholderNet(nlayers,n) for np=1:nparticles]
#     optim = SGD(lr=0.01, αlr=.98, N_epochs=2000000, logfreq=10000, bounds=[-1.,1.])
#     cache = TrainingCache(net)
#     Wall = [zeros(n,nparticles) for _=1:nlayers]
#     ball = zeros(nlayers,nparticles)
#     p = plot()
#     for np=1:nparticles
#         # reset optimizer
#         opt = resetopt(optim)
#         net = ensemble[np]

#         # initialize
#         initialization!(net, X,Xs, Ntrials=100000, rng=opt.rng)

#         # train
#         while sqJ_update(X,sqJ,fsb,cache,opt,verbose=false); end

#         # save training history
#         hist = historydf(opt.hist)
#         plot!(p,hist.loss, yscale=:log, label="")
        
#         # set weight
#         for l=1:nlayers
#             Wall[l][:,np] .= wb[1]
#         end

#         # set bias
#         ball[:,np] .= wb[2]
        
#         # save training history
#         CSV.write("$folder/traininghistory$np.csv", hist)
#     end
#     jldsave("$folder/ensemble.jl2", ensemble)
#     savefig(p, "$folder/trainingcurves.pdf")
#     return ensemble
# end