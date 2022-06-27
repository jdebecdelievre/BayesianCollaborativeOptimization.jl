import LinearAlgebra, OpenBLAS32_jll
LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
using Snopt
using SNOW

function simple!(g, x)
    # objective
    f = 4*x[1]^2 - x[1] - x[2] - 2.5

    # constraints
    g[1] = -x[2]^2 + 1.5*x[1]^2 - 2*x[1] + 1
    g[2] = x[2]^2 + 2*x[1]^2 - 2*x[1] - 4.25
    g[3] = x[1]^2 + x[1]*x[2] - 10.0

    return f
end

x0 = [1.0; 2.0]  # starting point
lx = [-5.0, -5]  # lower bounds on x
ux = [5.0, 5]  # upper bounds on x
ng = 3  # number of constraints
lg = -Inf*ones(ng)  # lower bounds on g
ug = zeros(ng)  # upper bounds on g

ipoptions = Dict{Any,Any}("linear_solver"=>"ma97")
# options = Options(solver=IPOPT(ipoptions))  # choosing IPOPT solver
options = Options(solver=SNOPT())  # choosing IPOPT solver

xopt, fopt, info = minimize(simple!, x0, ng, lx, ux, lg, ug, options)

println("xstar = ", xopt)
println("fstar = ", fopt)
println("info = ", info)