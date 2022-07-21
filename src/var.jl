
"""
Convenience structure for input and output variables name, size, initial value, upper and lower bound.
An optional `group` field also allows to indicate that two vars are part of the same group (for instance: inequality constraints)
Creation examples:
    Var(:pitch, 0., -1., 1.)
    Var(:lift, zeros(10), -ones(10), ones(10), :inequality_constraint)
    Var(:twist, 0., -1., 1., N=10)

The following methods are implemented on tuples of vars V:
    index(V, i): indices of ith variable in state vector
    index(V, :ineq): indices of variable from group :ineq in state vector
    indexbygroup(V): NamedTuple of indices for all groups
    indexbyname(V): NamedTuple of indices for all variables

    len(V): length of associated state space
    ini: vector of initial conditions
    upper: vector of upper bounds
    lower: vector of lower bounds

"""
vect(k::Number,N::Int64) = fill(k,N)
vect(k::Vector,N::Int64) = k
function checkinput(k,N::Int64)
    @assert (length(k) == N) || (length(k) == 1)
    return vect(k, N)
end

struct Var{N,ng}
    ini::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    group::NTuple{ng,Symbol}
    function Var(; lb, ub, ini=copy(lb), group::Union{Symbol,NTuple{ng,Symbol}} where ng =(), 
                    N=max(length(ini),length(lb),length(ub)))
        ini, lb, ub = checkinput(ini, N), checkinput(lb, N), checkinput(ub, N)
        ini, lb, ub = promote(ini, lb, ub)
        if typeof(group) == Symbol
            group = (group,)
        end
        ng = length(group)
        @assert (all(ini <= ub) && (ini >= lb)) "Initial guess not within bounds: $lb <= $ini <= $ub"
        return new{N,ng}(ini,lb,ub,group)
    end
end
Var(group::Symbol; N::Int64=1) = Var(lb=zeros(N), ub=ones(N), group=(group,))
# Base.show(io::IO, v::Var) = println(io, "$(v.lb) ≤ $(v.ini) ≤ $(v.ub)")

# Convenient index functions
NTV = NamedTuple
# Base.show(io::IO, V::NTV) = (for (vk,vi)=pairs(V); println(io, "$vk:$vi"); end)

ini(V::NTV) = vcat((v.ini for v = V)...)
lower(V::NTV) = vcat((v.lb  for v = V)...)
upper(V::NTV) = vcat((v.ub  for v = V)...)
ini_scaled(V::NTV) = (ini(V) .- lower(V)) ./ (upper(V) .- lower(V)) # inefficient only called once
varnames(V::NTV) = collect(keys(V))
len(v::Any) = length(v)
len(::Var{N}) where N = N
len(V::NTV) = isempty(V) ? 0 : sum(len(v) for v=V)
len(V::NTV, i::Int64) = len(V[i])
len(V::NTV, idx::UnitRange) = isempty(idx) ? 0 : sum(len(V[i]) for i=idx)
index(V::NTV) = (len(V)>1) ? (1:len(V)) : 1

function index(V::NTV, idx::Int64) 
    L = len(V,1:idx-1)
    if (len(V[idx])>1)
        return ((L+1):(L+len(V[idx])))
    else
        return 1+L
    end
end

function index(V::NTV, group::Symbol) 
    i = 0
    ind = Int64[]
    for v=V
        (group in v.group) && append!(ind, collect(1:len(v)) .+ i)
        i+=len(v)
    end
    return ind
end

"""
All indexbyname and indexbygroup functions return an IndexMap:
a NamedTuple connecting a symbol to a list of integers. Typical usage:
1/ Create subvector:
z_aero = z_all[idz.aero] 
2/ Populate supervector
z_all[idz.aero] .= z_aero (julia will automatically use a view of z_all to avoid copying)
"""
const IndexMap{fields,nfields} = NamedTuple{fields,NTuple{nfields,Vector{Int64}}}
indexbyname(V::NTV) = NamedTuple(((keys(V)[i], index(V,i)) for i=1:length(V)))
function indexbygroup(V::NTV) 
    groups = Tuple(unique(vcat((collect(v.group) for v=V)...))) # TODO: find a better way to concatenate tuples!
    inds = [index(V,g) for g=groups]
    return NamedTuple{groups}(inds)
end

"""
Subset of NamedTuple of Vars from the right group
"""
function subset(V::NTV, group::Symbol)
    k = Tuple(k for (k,v)=pairs(V) if group in v.group)
    v = (v for v=V if group in v.group)
    return NamedTuple{k}(v)
end

unpack(x::Vector, idxbname::NamedTuple) = map(i->x[i], idxbname)
unscale(x::Real, v::Var{1}) = x * (v.ub[1] - v.lb[1]) + v.lb[1]
unscale(x::SVector{N,<:Real}, v::Var{N}) where N = SA[(x[i] * (v.ub[i] - v.lb[i]) + v.lb[i] for i=1:N)...]
unscale(x::Vector{<:Real}, v::Var{N}) where N = [x[i] * (v.ub[i] - v.lb[i]) + v.lb[i] for i=1:N]
unscale(Xv::NamedTuple, V::NamedTuple)  = map(unscale, Xv,V)
unscale_unpack(x::Vector{<:Real}, idxbname::NamedTuple, V::NamedTuple) = map((i,v)->unscale(x[i],v), idxbname,V)

scale(x::Real, v::Var{1}) = (x - v.lb[1]) / (v.ub[1] - v.lb[1])
scale(x::SVector{N,<:Real}, v::Var{N}) where N = SA[((x[i] - v.lb[i]) * (v.ub[i] - v.lb[i]) for i=1:N)...]
scale(x::Vector{<:Real}, v::Var{N}) where N = [(x[i] - v.lb[i])/(v.ub[i] - v.lb[i]) for i=1:N]

function mergevar(V1::NTV, V2::NTV)
    # not very efficient but ok
    L = Pair{Symbol,Var}[]
    V = merge(V1,V2)
    shared_keys = intersect(keys(V1), keys(V2))
    for v = pairs(V)
        vk = v[1]
        if vk in shared_keys
            @assert len(V1[vk]) == len(V2[vk]) "Variable $vk has length $(len(V1[vk])) and $(len(V2[vk]))"
            ub = min.(V1[vk].ub, V2[vk].ub)
            lb = max.(V1[vk].lb, V2[vk].lb)
            ini = (V1[vk].ini .+ V2[vk].ini)./2
            @. ini = min(ini, ub)
            @. ini = max(ini, lb)
            group = (V1[vk].group == V2[vk].group) ? V1[vk].group : Symbol(V1[vk].group, V2[vk].group)
            push!(L, (vk=>Var(ini=ini, lb=lb, ub=ub, group=group)))
        else
            push!(L, v)
        end
    end
    return NamedTuple(L)
end
function mergevar(V1, V2, Vmore...)
    V = mergevar(V1,V2)
    for Vm=Vmore
        V = mergevar(V,Vm)
    end
    return V
end

"""
z and Z are vectors.
idz and idZ are NamedTuples returned by indexbyname 
This function sets the values of z with the corresponding values of Z. z is a subset of Z.
It will return an error of a field of idz is not present in idZ, or if Z is smaller than z.
Z is not modified.
"""
function getvar!(z, Z, idz, idZ)
    for k=keys(idz)
        for (iz,iZ)=zip(idz[k],idZ[k]) 
            z[iz] = Z[iZ]
        end
    end
end

"""
z and Z are vectors.
idz and idZ are NamedTuples returned by indexbyname 
This function sets the values of Z with the corresponding values of z.
z is not modified.
"""
function setvar!(Z, z, idZ, idz)
    for k=keys(idz)
        for (iz,iZ)=zip(idz[k],idZ[k]) 
            Z[iZ] = z[iz]
        end
    end
end