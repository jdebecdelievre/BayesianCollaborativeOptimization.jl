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

function datacheck(data::NamedTuple)
    for d = keys(data)
        D = data[d]
        (; Z, Zs, sqJ) = D
        Z = [Z; Zs]
        sqJ = [sqJ; sqJ*0.]
        l = length(Z)
        for i=1:l
            for j=i+1:l
                dist = norm(Z[i]-Z[j])
                    del = dist - abs(sqJ[i] - sqJ[j])
                    if del < 0
                        println("Lipshitz error between points $i and $j for $d: $del")
                    end
                end
            end
        end
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