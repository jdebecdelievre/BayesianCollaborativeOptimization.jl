"""
All solvers are subtypes of AbstractSolver.
They must all implement the get_new_point method
"""
abstract type AbstractSolver{pb<:AbstractProblem} end

"""
Options structure for solution process
"""
@with_kw struct SolveOptions
    n_ite::Int64 = 30 # number of iterations
    ini_samples::Int64 = 2 # number of initial random samples. 0 to use provided z0
    iteration_restart::Int64 = 0 # Iteration Number at which to restart.
    warm_start_sampler::Int64 = 0
    savedir::String = "xpu"
end

"""
Receives problem definition, solver type, options.
"""
function solve(solver::AbstractSolver, options::SolveOptions; 
        z0::Union{Nothing, <:AbstractArray}=nothing, terminal_print=true)
    (; n_ite, ini_samples, iteration_restart, warm_start_sampler, savedir) = options
    
    problem     = solver.problem
    disciplines = discipline_names(problem)
    Nz          = number_shared_variables(problem)
    idz         = indexmap(problem)
    to          = solver.to
    tol         = solver.tol
    
    # create xp directory, save options
    mkpath(savedir)
    io = terminal_print ? stdout : open("$savedir/msg.txt","w")
    println(io,options)
    JLD2.save("$savedir/inputs.jld2", "options", options, "idz", idz)

    # Create sampler
    Sz = SobolSeq(Nz)
    for _=1:warm_start_sampler
        next!(Sz)
    end

    # Initial samples and ensemble
    map(d->mkpath("$savedir/eval/$d"), disciplines)
    if iteration_restart < 1        
        # Get ini samples
        if ini_samples == 0
            @assert typeof(z0)!= Nothing
            Z = [z0]
            ini_samples += 1
        else
            Z = [next!(Sz) for _=1:ini_samples]
        end
        obj = map(z->objective(problem, z),Z)
        save("$savedir/obj.jld2","Z",Z,"obj",obj)

        # Create data storage 
        data = NamedTuple{disciplines}(Tuple((; 
            Z   = [zeros(length(idz[d])) for _=1:(ini_samples)], 
            Zs  = [zeros(length(idz[d])) for _=1:(ini_samples)], 
            sqJ = zeros(ini_samples),
            fsb = BitVector(zeros(ini_samples)),
            ite = (zeros(Int64,ini_samples)))
        for d = disciplines))

        # Evaluate subspaces
        for d=disciplines
            D = data[d]
            @timeit to "ite0" begin
            for (i,z)=enumerate(Z)
                D.Z[i]  .= z[idz[d]]
                @timeit to "$d" (D.Zs[i] .= subspace(problem, Val(d), D.Z[i], "$savedir/eval/$d/0_$i.txt"))
                D.sqJ[i] = norm(D.Z[i] .- D.Zs[i])
                D.fsb[i] = (D.sqJ[i]<tol)
            end
            end
            save_data("$savedir/$d.jld2",D)
        end
    else
        # Load discipline data
        data = NamedTuple{disciplines}(map(d->load_data("$savedir/$d.jld2"), disciplines))
        trim_data!(data, iteration_restart)

        # Load objective data
        Z, obj, sqJ, fsb = load("$savedir/obj.jld2","Z","obj","sqJ","fsb")
    end

    # Optimization
    @printf io "%3s %9s %9s %9s %4s \n" "ite" "obj" "∑√J" "EIc" "Nfsb"
    !terminal_print && flush(io)
    ite = iteration_restart+1
    idata = length(Z)
    mkpath("$savedir/solver")
    while ite < n_ite+1 
        @timeit to "ite$ite" begin

        # A/ Get new point
        @timeit to "solver" begin
            z, Zd, eic_max = get_new_point(ite, solver, data, "$savedir/solver")
        end

        # B/ Evaluate and save new point
        for d=disciplines
            @timeit to "$d" (zs = subspace(problem, Val(d), Zd[d], "$savedir/eval/$d/$(ite).txt"))
            sqJd = norm(zs-Zd[d])
            new_data = (;
                Z=Zd[d], Zs=zs, sqJ=sqJd, fsb=(sqJd<tol), ite=ite
            )
            map((D,nd)->push!(D,nd), data[d], new_data)
            save_data("$savedir/$d.jld2",data[d])
        end
        idata += 1
        
        # C/ Save progress
        o = objective(problem, z)
        push!(Z,z)
        push!(obj,o)
        fsb = map(D->D.fsb,data)
        sqJ = map(D->D.sqJ,data)
        save("$savedir/obj.jld2","Z",Z,"obj",obj,"sqJ",sqJ,"fsb",fsb)
        @printf io "%3i %.3e %.3e %.3e %2i \n" ite o sum(sqJ)[end] eic_max sum(fsb)[end]
        !terminal_print && flush(io)
        ite += 1
        end
    end

    # Print timer
    to_flatten = TimerOutputs.flatten(to)
    show(io, to)
    show(io, to_flatten)
    !terminal_print && close(io)

    fsb = map(D->D.fsb,data)
    sqJ = map(D->D.sqJ,data)
    return obj, sqJ, fsb, Z
end

function load_data(filename::String)
    @assert (split(filename, ".")[end] == "jld2") "Filename extension must be jld2"
    dat = load(filename)
    return (; 
        Z  = dat["Z"]  ,
        Zs = dat["Zs"] ,
        sqJ= dat["sqJ"],
        fsb= dat["fsb"],
        ite= dat["ite"],
    )
end

load_data(savedir::String, disciplines::Tuple{Symbol,Symbol}) = NamedTuple{disciplines}(map(d->load_data("$savedir/$d.jld2"), disciplines))
load_data(savedir::String, pb::AbstractProblem) = NamedTuple{discipline_names(pb)}(map(d->load_data("$savedir/$d.jld2"), discipline_names(pb)))

function save_data(filename, D)
    @assert (split(filename, ".")[end] == "jld2") "Filename extension must be jld2"
    save(filename,
            "Z",D.Z,"Zs",D.Zs,
            "sqJ",D.sqJ,"fsb",D.fsb,
            "ite",D.ite)
end

function trim_data!(data, ite)
    function trim(D)
        keep = (D.ite .<= ite)
        map(v->v[keep],D)
    end
    map(trim, data)
end 