using BayesianCollaborativeOptimization
using LinearAlgebra
using SNOW
##
V = (;
    z = Var(lb=-10., ub=10.),
    x = Var(lb=0., ub=10., N=2),
    y1 = Var(lb=8.,ub=100.),
    y2 = Var(lb=0.,ub=24.)
)
lb = lower(V)
ub = upper(V)
variables = (; d1=V, d2=V)
idx = indexbyname(V)

vopt = (; z = -2.8014207800091704, x = [0.07572497323210897, 0.10214619566441839], y1 = 8.001, y2 = 0.12915237776863955)
zopt = ([-2.8014207800091704, 0.07572497323210897, 0.10214619566441839, 8.001, 0.12915237776863955] - lower(V)) ./ (upper(V) - lower(V))
##
ipoptions = Dict{Any,Any}("print_level"=>0)


function objective(z)
    v = unscale_unpack(z, idx, V)
    return v.x[1]^2 + v.x[2] + v.y1 + exp(-v.y2)
end

function subspace1(z0, ipoptions=ipoptions)
    function fun(g, z)
        v = unscale_unpack(z,idx,V)
        g[1] = v.y1 - (v.z^2 + v.x[1] + v.x[2]-0.2*v.y2)
        return (z-z0) ⋅ (z-z0)
    end
    Nx = len(V)
    lx = zeros(Nx) # lower bounds on x
    ux = ones(Nx) # upper bounds on x
    Ng = 1
    lg = [0.]
    ug = [0.]
    
    options = SNOW.Options(derivatives=CentralFD(), solver=IPOPT(ipoptions))
    xopt, fopt, info = minimize(fun, copy(z0), Ng, lx, ux, lg, ug, options)
    return copy(xopt)
end

function subspace2(z0, ipoptions=ipoptions)
    function fun(g, z)
        v = unscale_unpack(z,idx,V)
        g[1] = v.y2 - (sqrt(v.y1) + v.z + v.x[2])
        return (z-z0) ⋅ (z-z0)
    end
    Nx = len(V)
    lx = zeros(Nx) # lower bounds on x
    ux = ones(Nx) # upper bounds on x
    Ng = 1
    lg = [0.]
    ug = [0.]
    
    options = SNOW.Options(derivatives=CentralFD(), solver=IPOPT(ipoptions))
    xopt, fopt, info = minimize(fun, copy(z0), Ng, lx, ux, lg, ug, options)
    return copy(xopt)
end

function sellaraao(z0)
    function fun(g,z)
        v = unscale_unpack(z,idx,V)
        g[1] = v.y1 - (v.z^2 + v.x[1] + v.x[2]-0.2*v.y2)
        g[2] = v.y2 - (sqrt(v.y1) + v.z + v.x[2])
        return objective(z)
    end

    Nx = len(V)
    lx = zeros(Nx) # lower bounds on x
    ux = ones(Nx) # upper bounds on x
    Ng = 2
    lg = [0.,0.]
    ug = [0.,0.]
    
    options = SNOW.Options(derivatives=CentralFD(), solver=IPOPT())
    xopt, fopt, info = minimize(fun, z0, Ng, lx, ux, lg, ug, options)
    return unscale_unpack(xopt, idx, V)
end


function solve_co()
    cotol = 1e-4
    # ipoptions=Dict( "tol"=>1e-6, "max_iter"=>500)
    ipoptions=Dict("print_level"=>2, "tol"=>1e-6, "max_iter"=>500)
    idz = indexbyname(V)

    function cofun(g,z)
        # Constraints
        zStar1 = subspace1(deepcopy(z),ipoptions)
        zStar2 = subspace1(deepcopy(z),ipoptions)
        g1 = (z-zStar1) ⋅ (z-zStar1)
        g2 = (z-zStar2) ⋅ (z-zStar2)
        # f =  objective(z) + log(g1+g2)
        f =  g1+g2
        return f
    end
    
    # function cofun(g, df, dg, z)
    #     # Constraints
    #     zStar1 = subspace1(z,ipoptions)
    #     @. dg[1,:] = z-zStar1 
    #     zStar2 = subspace2(z,ipoptions)
    #     @. dg[2,:] = z-zStar2
    #     g[1] = (dg[1,:] ⋅ dg[1,:])/2
    #     g[2] = (dg[2,:] ⋅ dg[2,:])/2

    #     # Objective
    #     v = unscale_unpack(z, idx, V)
    #     df[idx.x[1]] = 2*v.x[1]
    #     df[idx.x[2]] = 1.
    #     df[idx.y1]   = 1.
    #     df[idx.y2]   = - exp(-v.y2)

    #     return v.x[1]^2 + v.x[2] + v.y1 + v.y2
    # end

    Nz = len(V)
    z0 = copy(zopt)#ini_scaled(V)  # starting point
    lz = zeros(Nz) # lower bounds on z
    uz = ones(Nz) # upper bounds on z

    co_options = Dict("tol"=>1e-4, "max_iter"=>150, "tol"=>1e-3,"print_level"=>5)
    options = SNOW.Options(derivatives=SNOW.CentralFD(), solver=SNOPT())
    # options = SNOW.Options(derivatives=SNOW.UserDeriv(), solver=IPOPT(co_options))
    xopt, fopt, info = minimize(cofun, z0, 0, lz, uz, zeros(0), zeros(0), options)
    println("RESULTS ",cofun([], z0))
    println("RESULTS pred",fopt)
    
    # # print result
    # v = (upper(V) - lower(V)) .* xopt + lower(V)
    # v = unpack(v, idz)

    # for k = keys(v)
    #     println("$k: $(v[k])")
    # end
    return xopt
end
##
subspace = (; d1=subspace1, d2=subspace2)
options = BCOoptions(savedir="$(pwd())/sellarbco")
##
bco(variables, subspace, options)
# function solve_co()
#     cotol = 1e-4
#     ipoptions=Dict("print_level"=>2, "tol"=>1e-6, "max_iter"=>500)
#     idz = indexbyname(V)

#     function cofun(g, z)
#         # Constraints
#         zStar1 = subspace1(z,ipoptions)
#         g[1] = (z-zStar1) ⋅ (z-zStar1)
#         zStar2 = subspace2(z,ipoptions)
#         g[2] = (z-zStar2) ⋅ (z-zStar2)

#         return objective(z)
#     end
    
#     # function cofun(g, df, dg, z)
#     #     # Constraints
#     #     zStar1 = subspace1(z,ipoptions)
#     #     @. dg[1,:] = z-zStar1
#     #     zStar2 = subspace2(z,ipoptions)
#     #     @. dg[2,:] = z-zStar2
#     #     g[1] = (dg[1,:] ⋅ dg[1,:])/2
#     #     g[2] = (dg[2,:] ⋅ dg[2,:])/2

#     #     # Objective
#     #     v = unscale_unpack(z, idx, V)
#     #     df[idx.x[1]] = 2*v.x[1]
#     #     df[idx.x[2]] = 1.
#     #     df[idx.y1]   = 1.
#     #     df[idx.y2]   = - exp(-v.y2)

#     #     return v.x[1]^2 + v.x[2] + v.y1 + v.y2
#     # end