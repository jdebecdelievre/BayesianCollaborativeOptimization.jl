function learn_feasible_set(options, data, savedir)
    (; nparticles,  nlayers, nresample, N_epochs, logfreq, lr, αlr, ntrials) = options
    ENV["GKSwstype"] = "100"
    
    mkpath(savedir)
    bounds= @SVector [0.,1.]
    n = size(data.Z[1],1)
    
    if nlayers == 0
        nlayers = 5*length(data.Z)
    end
    
    ensemble = [HouseholderNet(nlayers,n) for _=1:nparticles*ntrials]
    sgd = SGD(lr=lr, αlr=αlr, N_epochs=N_epochs, logfreq=logfreq, bounds=bounds, dropprob=options.dropprob)
    optim = [resetopt(sgd) for _=1:nparticles*ntrials]
    trainingcaches = [TrainingCacheGrad(net) for net in ensemble]
    # p = plot()

    # Prep data
    fsb = data.sqJ .< options.training_tol 
    ifs = .!fsb
    X   = [SVector{n}.(data.Z); SVector{n}.(data.Zs[ifs])]
    Y   = [data.sqJ; 0*data.sqJ[ifs]]
    fsb = [fsb; fsb[ifs]]
    ∇Y  = (data.Z-data.Zs)
    ∇Y  = [∇Y; ∇Y[ifs]]
    normalize!.(∇Y)
    ∇Y  = SVector{n}.(∇Y)
    @assert length(X) == length(Y) == length(fsb) == length(∇Y)
    
    # Train nparticles nets
    loss = 10*ones(nparticles*ntrials)
    nvalid = 0
    for np=1:nparticles*ntrials
        # reset optimizer
        opt = optim[np]
        net = ensemble[np]
        cache = trainingcaches[np]

        # initialize
        initialization!(net, rng=opt.rng, bounds=bounds)

        # train
        while sqJ_grad_update(X, Y, ∇Y, fsb, cache, opt, verbose=false); end
        
        # save training history
        hist = historydf(opt.hist,NamedTuple)
        # plot!(p,hist.loss .+ eps(), yscale=:log10, label="$np")
        CSV.write("$savedir/traininghistory$np.csv", hist)
        
        # Break if needed
        loss[np] = hist.loss[end] .+ eps() 
        if hist.loss[end]<1e-4
            nvalid += 1
        end
        if nvalid == nparticles
            break
        end
    end

    ## Select best
    @save "$savedir/losses.jld2" loss
    ensemble = ensemble[partialsortperm(loss,1:nparticles)]

    # # Resample inactive layers to diversify network
    if nresample > 0
        ensemble = vcat(ensemble,([HouseholderNets.resampleinactive(X, ensemble[np], bounds, optim[np].rng) for i=1:nresample] for np=1:nparticles)...)
    end
    
    df = (; X, Y, fsb)
    CSV.write("$savedir/trainingdata.csv", df)
    save_ensemble("$savedir/ensemble.jld2", ensemble)
    # savefig(p, "$savedir/trainingcurves.pdf")
    return ensemble
end

function save_ensemble(filename, ensemble)
    @assert (split(filename, ".")[end] == "jld2") "Filename extension must be jld2"
    # StaticVectors prevent serialization
    ensemble_ = [HouseholderNet(Vector.(e.W), e.b) for e=ensemble];
    save(filename, "ensemble", ensemble_)
end

function load_ensemble(filename::String)
    @assert (split(filename, ".")[end] == "jld2") "Filename extension must be jld2"
    ensemble = load(filename)["ensemble"]
    return ensemble
end

function load_ensemble(savedir::String, disciplines::Tuple{Symbol,Symbol})
    netdir = NamedTuple{disciplines}(map(d->"$savedir/$d/ensemble.jld2",disciplines))
    ensemble = map(load_ensemble, netdir)
    return ensemble
end