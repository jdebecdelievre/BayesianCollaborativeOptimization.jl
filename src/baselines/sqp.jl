"""
Sequential quadratic programming solution of CO
"""
struct SQP{prb} <: AbstractSolver{prb}
    problem::prb
    λ::Float64
    to::TimerOutput
    tol::Float64
    function SQP(prb::problem; tol::Float64 = 5e-3, λ::Float64=1.) where {problem<:AbstractProblem}
        to = TimerOutput()
        new{problem}(prb, λ, to, tol)
    end
end


function solve(solver::SQP, options::SolveOptions; 
                z0::Union{Nothing, <:AbstractArray}=nothing, terminal_print=true)
    (; n_ite, ini_samples, iteration_restart, warm_start_sampler, savedir) = options
    ipoptions=Dict("print_level"=>2, "max_iter"=>500)
    tol = solver.tol
    
    # Problem def
    problem     = solver.problem
    disciplines = discipline_names(problem)
    Nd          = length(disciplines)
    Nz          = number_shared_variables(problem)
    idz         = indexmap(problem)
    
    # initial guess
    if isnothing(z0)
        Sz = SobolSeq(Nz)
        for _=1:warm_start_sampler
            next!(Sz)
        end
        z0 = next!(Sz)
    else
        z0 = copy(z0)
    end
    
    # Initialize storage variables
    data = NamedTuple{disciplines}(Tuple((; 
        Z   = Vector{Float64}[], 
        Zs  = Vector{Float64}[], 
        sqJ = Float64[],
        fsb = BitVector(),
        ite = Int64[]) for d = disciplines))
    Z  = Vector{Float64}[]
    obj = Float64[]
    Zd = map(id->z0[id], idz)
    Zs = map(id->z0[id], idz)
    ite = 1
    mkpath("$savedir/eval/")

    # fcache = 0
    # gcache = zeros(Nd)
    # dfcache = zeros(Nz)
    # dgcache = zeros(Nd,Nz)
    # zcache = zeros(Nz)
    function cofun(g, df, dg, z)
        # return saved values if duplicate z

        push!(Z, copy(z))
        
        # Objective
        o = objective(solver.problem, z, df)
        df .*= -1.
        push!(obj,o)
        
        for (i,d)=enumerate(disciplines)
            push!(data[d].ite, ite)

            # Assign value
            @. Zd[d] = z[idz[d]]
            push!(data[d].Z, copy(Zd[d]))
            
            # Subspace projection
            Zs[d] .= subspace(solver.problem, Val(d), Zd[d], "$savedir/eval/$(d)_$ite.txt")[1]
            push!(data[d].Zs, copy(Zs[d]))
            
            # Gradient calc
            @. dg[i, idz[d]] = 2*(Zd[d] - Zs[d])
            @. df[idz[d]] += 2*solver.λ*(Zd[d] - Zs[d])

            # Constraint calc
            g[i] = (dg[i,:] ⋅ dg[i,:]) / 4

            push!(data[d].sqJ, sqrt(g[i]))
            fsb = (sqrt(g[i]) < tol)
            push!(data[d].fsb, fsb)
        end
        ite += 1
        
        return -o + solver.λ * sum(g)
    end
    
    lz = zeros(Nz) # lower bounds on z
    uz = ones(Nz) # upper bounds on z
    lg = -Inf * ones(Nd)
    ug = tol^2 * ones(Nd)# upper bounds on g
    
    co_options = Dict{Any,Any}("max_iter"=>n_ite, "output_file"=>"$savedir/solver.txt")
    options = SNOW.Options(derivatives=SNOW.UserDeriv(), solver=SNOW.IPOPT(co_options))
    xopt, fopt, info = SNOW.minimize(cofun, z0, Nd, lz, uz, lg, ug, options)
    
    sqJ = map(D->D.sqJ,data)
    fsb = map(D->D.fsb,data)
    save("$savedir/obj.jld2","Z",Z,"obj",obj,"sqJ",sqJ,"fsb",fsb)
    map(d->save_data("$savedir/$d.jld2",data[d]), disciplines)
    return obj, sqJ, fsb, Z
end