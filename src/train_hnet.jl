function learn_feasible_set(data, savedir, options)
    (; nparticles,  nlayers, nresample, N_epochs, logfreq, lr, αlr, ntrials) = options
    ENV["GKSwstype"] = "100"
    
    mkpath(savedir)
    bounds= @SVector [0.,1.]
    n = size(data.Z[1],1)
    
    if nlayers == 0
        nlayers = length(data.Z)
    end
    
    ensemble = [HouseholderNet(nlayers,n) for _=1:nparticles*ntrials]
    sgd = SGD(lr=lr, αlr=αlr, N_epochs=N_epochs, logfreq=logfreq, bounds=bounds, dropprob=options.dropprob)
    optim = [resetopt(sgd) for _=1:nparticles*ntrials]
    trainingcaches = [TrainingCache(net) for net in ensemble]
    p = plot()

    # Prep data
    ifs = .!data.fsb
    X   = [SVector{n}.(data.Z); SVector{n}.(data.Zs[ifs])]
    Y   = [data.sqJ; 0*data.sqJ[ifs]]
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
    loss = 10*ones(nparticles*ntrials)
    nvalid = 0
    for np=1:nparticles*ntrials
        # reset optimizer
        opt = optim[np]
        net = ensemble[np]
        cache = trainingcaches[np]

        # initialize
        initialization!(net, data.Z, data.Zs, Ntrials=100000, rng=opt.rng, bounds=bounds)

        # train
        while mse_update(X, Y, cache, opt, verbose=false); end
        
        # save training history
        hist = historydf(opt.hist)
        plot!(p,hist.loss, yscale=:log10, label="$np")
        CSV.write("$savedir/traininghistory$np.csv", hist)
        
        # Break if needed
        loss[np] = hist.loss[end]
        if hist.loss[end]<1e-4
            nvalid += 1
        end
        if nvalid == nparticles
            break
        end
    end

    ## Select best
    @save "$savedir/losses.jld2" loss
    ensemble[partialsortperm(loss,1:nparticles)]

    # # Resample inactive layers to diversify network
    if nresample > 0
        ensemble = vcat(ensemble,([HouseholderNets.resampleinactive(X, ensemble[np], bounds, optim[np].rng) for i=1:nresample] for np=1:nparticles)...)
    end
    
    df = DataFrame(X=X, Y=Y, fsb=fsb)
    CSV.write("$savedir/trainingdata.csv", df)
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