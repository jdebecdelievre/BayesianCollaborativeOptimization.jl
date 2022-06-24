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

function avg_z(data::NamedTuple{disciplines}, row::Int64, idz::NamedTuple{disciplines}) where disciplines
    Nz  = maximum(map(maximum,idz))
    z   = zeros(Nz)
    nid = zeros(Nz)
    for d=disciplines
        z[idz[d]] .+= data[d].Z[row]
        nid[idz[d]] .+= 1.
    end
    @. z = z / nid
    return z
end

function avg_zstar(data::NamedTuple{disciplines}, row::Int64, idz::NamedTuple{disciplines}, idz_all::NamedTuple) where disciplines
    Nz  = maximum(map(maximum,idz))
    z   = zeros(Nz)
    nid = zeros(Nz)
    for d=disciplines
        z[idz[d]] .+= data[d].Zs[row]
        nid[idz[d]] .+= 1.
    end
    @. z = z / nid
    return z
end