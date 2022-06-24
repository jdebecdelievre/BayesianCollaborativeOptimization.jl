"""
All solvers are subtypes of AbstractSolver.
They must all implement the get_new_point method
"""
abstract type AbstractSolver end

"""
Options structure for solution process
"""
@with_kw struct Options
    n_ite::Int64 = 30 # number of iterations
    ini_samples::Int64 = 2 # number of initial random samples. 0 to use provided z0
    iteration_restart::Int64 = 0 # Iteration Number at which to restart.
    warm_start_sampler::Int64 = 0
    tol::Float64 = 5e-3
    savedir::String = "xpu"
end

"""
Default objective function to maximize.
"""
function default_obj(z,grad=nothing)
    if typeof(grad) <: AbstractArray
        grad .= 0.
        grad[1] = 1.
    end
    return z[1]
end

"""
Receives problem definition, solver type, options.
"""
function solve(solver::AbstractSolver, 
            subspace_fun::NamedTuple{disciplines}, 
            idz::IndexMap{disciplines}, 
            options::Options; objective=default_obj, z0=nothing) where disciplines
    (; n_ite, ini_samples, iteration_restart, warm_start_sampler, tol, savedir) = options

    # create xp directory, save options
    mkpath(savedir)
    open("$savedir/inputs.txt","w") do io
        println(io,options)
    end
    JLD2.save("$savedir/inputs.jld2","options",options,"idz",idz)

    # Create sampler
    Nz = maximum(map(maximum,idz))
    Sz = SobolSeq(Nz)
    for _=1:warm_start_sampler
        next!(Sz)
    end

    # Initial samples and ensemble
    map(d->mkpath("$savedir/eval/$d"), disciplines)
    if iteration_restart < 1
        # Create data storage 
        data = NamedTuple{disciplines}(Tuple((; 
            Z   = [zeros(length(idz[d])) for _=1:(ini_samples)], 
            Zs  = [zeros(length(idz[d])) for _=1:(ini_samples)], 
            sqJ = zeros(ini_samples),
            fsb = BitVector(zeros(ini_samples)),
            ite = (zeros(Int64,ini_samples)))
        for d = disciplines))

        # get ini samples
        if ini_samples == 0
            @assert typeof(z0)!= Nothing
            Z = [z0]
        else
            Z = [next!(Sz) for _=1:ini_samples]
        end
        obj = objective.(Z)
        save("$savedir/obj.jld2","Z",Z,"obj",obj)

        # Evaluate subspaces
        for d=disciplines
            D = data[d]
            for (i,z)=enumerate(Z)
                D.Z[i]  .= z[idz[d]]
                D.Zs[i] .= subspace_fun[d](D.Z[i], "$savedir/eval/$d/0_$i.txt")
                D.sqJ[i] = norm(D.Z[i] .- D.Zs[i])
                D.fsb[i] = (D.sqJ[i]<tol)
            end
            save_data("$savedir/$d.jld2",D)
        end
    else
        # Load data
        data = NamedTuple{disciplines}(map(d->load_data("$savedir/$d.jld2"), disciplines))
        trim_data!(data, iteration_restart)
    end

    # Optimization
    @printf "%3s %9s %9s %9s %4s \n" "ite" "obj" "∑√J" "EIc" "Nfsb"
    ite = iteration_restart+1
    idata = length(Z)
    mkpath("$savedir/solver")
    while ite < n_ite+1 
        # A/ Get new point
        z, Zd, eic = get_new_point(ite, solver, objective, data, idz, "$savedir/solver")

        # B/ Evaluate and save new point
        for d=disciplines
            zs = subspace_fun[d](Zd[d], "$savedir/eval/$d/$(ite).txt")
            sqJd = norm(zs-Zd[d])
            new_data = (;
                Z=Zd[d], Zs=zs, sqJ=sqJd, fsb=(sqJd<tol), ite=ite
            )
            map((D,nd)->push!(D,nd), data[d], new_data)
            save_data("$savedir/$d.jld2",data[d])
        end
        idata += 1
        
        # C/ Save progress
        o = objective(z)
        glb = sum(map(D->D.fsb,data))
        sqJ = sum(map(D->D.sqJ,data))
        push!(Z,z)
        push!(obj,o)
        save("$savedir/obj.jld2","Z",Z,"obj",obj,"sqJ",sqJ,"glb",glb)
        @printf "%3i %.3e %.3e %.3e %2i \n" ite o sqJ[end] eic glb[end]
        ite += 1
    end
    return data
end

function load_data(filename)
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