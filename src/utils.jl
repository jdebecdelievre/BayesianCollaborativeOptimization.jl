function get_metrics(mdl::AbstractProblem, savedir::String)
    disciplines = discipline_names(mdl)
    Nd = length(disciplines)
    fopt = objective_opt(mdl)
    lb_opt = objective_lowerbound(mdl)
    ub_opt = objective_upperbound(mdl)
    
    # Compute metrics
    if isfile("$savedir/obj.jld2")
        obj, sqJ = load("$savedir/obj.jld2","obj","sqJ")
    else
        obj, sqJ = load("$savedir/data.jld2","obj","sqJ")
    end
    metric = [abs.(obj .- f) / (ub_opt - lb_opt) + sum(sqJ) for f=fopt]
    i = [argmin([m[k] for m=metric]) for k=1:length(obj)]
    metric = [metric[i[k]][k] for k=1:length(obj)]
    dobj = [abs.(obj[k] - fopt[i[k]]) for k=1:length(obj)]

    # Reorder
    for i=2:length(metric)
        # metric[i] = min(metric[i],metric[i-1])
        if metric[i] > metric[i-1]
        # if dobj[i] > dobj[i-1]
            metric[i] = metric[i-1]
            obj[i] = obj[i-1]
            dobj[i] = dobj[i-1]
            for j=1:Nd
                sqJ[j][i] = sqJ[j][i-1]
            end
        end
    end

    return metric, obj, dobj, sqJ
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