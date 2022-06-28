import SNOW
"""
Sequential quadratic programming solution of CO
"""
struct SQP{prb} <: AbstractSolver{prb}
    problem::prb
    λ::Float64
    to::TimerOutput
    function SQP(prb::problem; λ::Float64=1.) where {problem<:AbstractProblem}
        to = TimerOutput()
        new{problem}(prb, λ, to)
    end
end


function solve(solver::SQP, options::SolveOptions; 
                z0::Union{Nothing, <:AbstractArray}=nothing, terminal_print=true)
    (; n_ite, ini_samples, iteration_restart, warm_start_sampler, tol, savedir) = options
    cotol = 1e-4
    ipoptions=Dict("print_level"=>2, "tol"=>1e-6, "max_iter"=>500)
    
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
    glb = Int64[]
    Zd = map(id->z0[id], idz)
    Zs = map(id->z0[id], idz)
    ite = 1

    function cofun(g, df, dg, z)
        push!(Z, copy(z))
        push!(glb,0)
        
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
            Zs[d] .= subspace(solver.problem, Val(d), Zd[d], "")
            push!(data[d].Zs, copy(Zs[d]))
            
            # Gradient calc
            @. dg[i, idz[d]] = 2*(Zd[d] - Zs[d])
            @. df += 2*solver.λ*(Zd[d] - Zs[d])

            # Constraint calc
            g[i] = (dg[i,:] ⋅ dg[i,:]) / 4

            push!(data[d].sqJ, g[i])
            fsb = (g[i] < cotol)
            push!(data[d].fsb, fsb)
            glb[end] += fsb
        end
        ite += 1
        
        return -o + solver.λ * sum(g)
    end
    
    lz = zeros(Nz) # lower bounds on z
    uz = ones(Nz) # upper bounds on z
    lg = -Inf * ones(Nd)
    ug = cotol * ones(Nd)# upper bounds on g
    
    co_options = Dict("tol"=>1e-4, "max_iter"=>50)
    options = SNOW.Options(derivatives=SNOW.UserDeriv(), solver=SNOW.IPOPT(co_options))
    xopt, fopt, info = SNOW.minimize(cofun, z0, Nd, lz, uz, lg, ug, options)
    
    sqJ = map(D->D.sqJ,data)
    fsb = map(D->D.fsb,data)
    save("$savedir/obj.jld2","Z",Z,"obj",obj,"sqJ",sqJ,"glb",glb)
    map(d->save_data("$savedir/$d.jld2",data[d]), disciplines)
    return obj, sqJ, fsb, Z
end