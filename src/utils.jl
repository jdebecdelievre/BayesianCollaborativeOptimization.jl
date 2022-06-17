function get_metrics(savedir::String, disciplines::Tuple, zopt::AbstractVector, objective)
    Nd = length(disciplines)
    fopt = objective(zopt)

    sqJ = NamedTuple{disciplines}(map(d->load("$savedir/$d.jld2", "sqJ"),disciplines))
    obj = load("$savedir/data.jld2","obj")
    
    metric = abs.(obj .- fopt) + sum(sqJ)
    for i=2:length(metric)
        metric[i] = min(metric[i],metric[i-1])
    end
    return metric, obj, sqJ
end