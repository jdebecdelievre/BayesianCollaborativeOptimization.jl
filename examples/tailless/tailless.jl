using LinearAlgebra
using BayesianCollaborativeOptimization
using Snopt
using SNOW
import Parameters: @consts

#### ---- Constant Inputs ----
@consts begin
    # Dimensionless data
    AR       = 15.
    taper    = 0.5
    t_c      = 0.12
    sweep    = 15.
    eta_prop = 0.8
    CDpw     = 0.009
    c_ref    = 1.
    ny       = 20 # total number of panels
    nde      = 4 # number of panels equipped with control surfaces

    # Data in imp units
    S             = 200. # ft^2
    sigma_failure = 40e3 * 12^2 # psf (pounds per sq. foot), 40 ksi
    FuseDragArea  = 1.5 # sqft
    g             = 32.1740 # ft/s^2
    W             = 1000. # lbs
    W0            = 300. # lbs
    SFC           = 0.8 / 550 / 3600 * 6076.12 # 0.8 lb / hr / hp in per nautical mile (hp = 550 ft lb / s)
    rho_al        = .101 * 12^3 # lb/ft^3

    # # Data in SI units
    # S = 18.59 # m^2
    # sigma_failure = 275e6 #Pa 40 ksi
    # FuseDragArea = 0.139
    # g = 9.81
    # W = 453.5 * g
    # W0 = 136.1 * g
    # SFC = 0.0001351726666667  / g # J/kg, 0.8 lb / hr / hp x 608.277 / (3.6 x 109)
    # rho_al = 2795.67038 * g # N/m^3, or .101 lb/in^3

    # Planform specifications
    span       = sqrt(S * AR)
    half_span  = span / 2             # wing half-span in m
    mean_chord = S / span
    root_chord = mean_chord * 2 / (1+taper) # root chord in m
    tip_chord  = root_chord  * taper # tip chord in m

    # Precompute friction drag
    CDp = (FuseDragArea +  2 * S) * CDpw / S
end


#### ---- Global variables ----
@consts begin
    global_variables = (;
        R   = Var(ini=4e3, lb=3000, ub=7000, group=(:struc,)),
        D   = Var(ini=50., lb=W/30, ub=W/15, group=(:aero,:struc)),
        a1  = Var(ini=3/4, lb=.5,   ub=1,    group=(:aero,:struc)),
        a3  = Var(ini=0.,  lb=-.5,  ub=.5,   group=(:aero,:struc)),
        xcg = Var(ini=3.,  lb=1,    ub=5,    group=(:aero,:struc))
    )

    tailless_idz = indexbygroup(global_variables)
    aero_global  = subset(global_variables, :aero)
    struc_global = subset(global_variables, :struc)
end

#### ---- Aero ----
function vtx_influence(x_c, y_c, x_l, y_l, x_r, y_r)
    AIC = zeros(length(y_c),length(y_l))
    for i=1:size(AIC,1)
        for j=1:size(AIC,2)
            y_cl = y_c[i] - y_l[j]
            y_cr = y_c[i] - y_r[j]
            x_cl = x_c[i] - x_l[j]
            x_cr = x_c[i] - x_r[j]

            n_cl = sqrt(y_cl^2 + x_cl^2)
            n_cr = sqrt(y_cr^2 + x_cr^2)
        
            k1 = 1/(4*pi) / (n_cl*n_cr + x_cr*x_cl + y_cr*y_cl) * (1/n_cr + 1/n_cl)
            k2 = 1/(4*pi) / (n_cl - x_cl) / n_cl
            k3 = 1/(4*pi) / (n_cr - x_cr) / n_cr
            AIC[i,j] = (x_cl*y_cr - y_cl*x_cr)*k1 - y_cl * k2 + y_cr * k3
        end
    end
    
    return AIC
    end

function prepare_aero()
    # TLMesh
    x_vtx = root_chord / 4 .+ collect(LinRange(0, half_span, ny+1)) .* tan(sweep / 180. * pi)
    y_vtx = collect(LinRange(0, half_span, ny+1))
    theta_vtx = acos.(y_vtx ./ half_span)
    theta_panel_size = (theta_vtx[1:end-1] - theta_vtx[2:end])
    chord_vtx = collect(LinRange(root_chord, tip_chord, ny+1))
    chord_c = (chord_vtx[2:end] + chord_vtx[1:end-1]) / 2
    y_l = y_vtx[1:end-1] # left corner
    y_r = y_vtx[2:end] # right corner
    x_l = x_vtx[1:end-1] # left corner
    x_r = x_vtx[2:end] # right corner
    y_c = (y_l + y_r) ./ 2
    x_c = (x_l + x_r) ./ 2 .+ chord_c ./ 2
    theta_c = acos.(y_c ./ half_span)
    beam_height = t_c .* chord_c .* 0.75
    sparcap_width = chord_c .* 0.3
    TLmesh = (; x_vtx = x_vtx, y_vtx = y_vtx, theta_vtx=theta_vtx, chord_vtx = chord_vtx,
                y_c = y_c, x_c = x_c, theta_c=theta_c, chord_c = chord_c, 
                y_l = y_l, y_r = y_r, x_l = x_l, x_r = x_r, 
                pairwise_y_invdist=(1 ./(y_c' .- y_vtx) - # right wing
                                    1 ./(y_c' .+ y_vtx) # left wing
                                    )./ (2*pi), # pre-divide by 2pi,
                panel_size = half_span / ny,
                theta_panel_size = theta_panel_size,
                beam_height = beam_height,
                sparcap_width = sparcap_width
            )

    # AIC matrix
    AIC = (vtx_influence(x_c, y_c, x_l, y_l, x_r, y_r)  # right wing
            + vtx_influence(x_c, y_c, x_r, -y_r, x_l, -y_l)# symmetric left wing
            )
    # n = (x_c - x_l) * (y_c - y_r) - (y_c - y_l) * (x_c - x_r)
    
    # LU decomposition
    AIC_LU = lu(AIC)

    return AIC_LU, TLmesh
end
@consts begin
    AIC_LU, TLmesh = prepare_aero()

    aero_local = (
        twist   = Var(ini= ones(ny)*5., lb=-10,  ub=25),
        alpha   = Var(ini= 5, lb=0,   ub=15),
        delta_e = Var(ini=-5, lb=-25, ub=0)
    )
    aero_output = (
        qpos     = Var(lb=-Inf, ub=0.),
        Cl       = Var(lb=-Inf, ub=0., N=ny),
        Cm_al    = Var(lb=-Inf, ub=0.),
        Cm_al_4g = Var(lb=-Inf, ub=0.),
        load_4g  = Var(lb=-Inf, ub=0.),
        D        = Var(lb=-Inf, ub=0.),
        Cm       = Var(lb=0., ub=0.),
        Cm_4g    = Var(lb=0., ub=0.),
        a1       = Var(lb=0., ub=1.),
        a3       = Var(lb=0., ub=1.),
    )
end

function aero(alpha, delta_e, twist, xcg)
    pnl = TLmesh.panel_size
    twist = twist /180 * pi
    alpha = alpha /180 * pi
    delta_e = delta_e /180 * pi

    # Circulations
    RHS = -sin.(twist)
    Sig = AIC_LU \ RHS

    # Solve at 4g
    tw_4g = twist .+ alpha
    for e in 0:nde-1
        tw_4g[end-e] += delta_e
    end
    @. RHS = -sin(tw_4g)
    Sig_4g = AIC_LU \ RHS

    # Dynamic Pressure
    L = 4 * sum(Sig) * pnl # in units of q
    CL = L / S
    q = W / (L + 1e-10)

    # Local Cl
    Cl = 2 .* Sig_4g ./ TLmesh.chord_c
    
    # Loads
    load_4g = 2 .* Sig_4g .* ( 2 * q ) .* pnl # force in lb

    # Trefftz plane induced drag in cruise + provided parasitic drag
    Gamma = zeros(eltype(Sig), length(Sig)+1)
    Gamma[2:end-1] .= diff(Sig)
    Gamma[end] = -Sig[end]
    
    Gradient = TLmesh.pairwise_y_invdist' * Gamma 
    CDi = 2 * (Gradient ⋅ Sig) * pnl ./ S
    D = q * S * (CDi + CDp)

    # Moments calculation (in units of q)
    ΔX = ((TLmesh.x_l + TLmesh.x_r)./2 .- xcg)
    Cm = - 2 * (2 * Sig ⋅ ΔX) * pnl / S / c_ref
    Cm_4g = - 2 * (2 * Sig_4g ⋅ ΔX) * pnl / S / c_ref

    # Pitch stiffness
    @. RHS = -cos(twist)
    dSig_dal = AIC_LU \ RHS
    Cm_al = - 2 * (2 * dSig_dal ⋅ ΔX) * pnl / S / c_ref
    Cm_al_4g = Cm_al

    # # Pitch stiffness at 4g
    # @. RHS = -cos(twist)
    # dSig_dal = AIC_LU \ RHS
    # Cm_al_4g = - 2 * (2 * dSig_dal ⋅ ΔX) * pnl / S / c_ref


    return D, load_4g, Cl, Cm, Cm_4g, q, Cm_al, Cm_al_4g
end

#### ---- Structures ----
@consts begin
    struc_local = (
        thickness = Var(ini=TLmesh.beam_height/16, lb=1/12/32, ub=TLmesh.beam_height/2),
        Wt        = Var(ini=50., lb=1, ub=100)
    )
    struc_output = (
        sigma = Var(lb=-Inf, ub=0., N=ny),
        Wfuel = Var(lb=-Inf, ub=0. ),
        Wwing = Var(lb=-Inf, ub=0. ),
        R     = Var(lb=-Inf, ub=0.),
        xcg   = Var(lb=0., ub=1.),
    )
end

function structures(thickness, Wt, loads)
    (; y_c, y_vtx, beam_height, sparcap_width) = TLmesh
    pnl = TLmesh.panel_size
    
    # Wing weight and xcg
    web_thickness = thickness
    section_area = (thickness .* (sparcap_width-web_thickness) .* 2 + beam_height .* web_thickness) 
    beam_volume = sum(section_area) * pnl
    Wwing = (2 * 1.5 * rho_al) .* beam_volume # 2 wings, 1.5 safety factor
    wing_cg = (section_area ⋅ (TLmesh.x_c - TLmesh.chord_c/2)) * pnl / beam_volume
    xcg = (wing_cg * Wwing +
            (root_chord / 4 + half_span * tan(sweep*pi/180)) * Wt + 
            W0 * root_chord * .6) / (Wwing + Wt + W0)

    # Add wing weight to loads
    loads = loads - (1.5 * pnl * rho_al * 4) .* section_area # x4 because 4g relief

    # Bending moment and stress at inboard edge of panel
    M = ((reverse(cumsum(reverse(loads .* y_c)) - (reverse(y_vtx[1:end-1]) .* cumsum(reverse(loads))))))

    # Add tip ballast moment
    M = M - Wt .* (half_span .- TLmesh.y_vtx[1:end-1]) * 2 # x2 because 4g relief but halfwing

    # Area moment of inertia, stresses (/!\ assuming that web and spar cap have the same thicknesses)
    h = (beam_height - 2 * thickness)
    I = copy(h)
    @. I = 1/12 * (web_thickness * h^3 + 
        sparcap_width * (beam_height^3 - h^3))
    sigma = (M .* beam_height) ./ I ./ 2

    # Range
    Wfuel = W - W0 - Wwing - Wt
    
    return xcg, sigma, Wfuel, Wwing
end

#### ---- Performance ----
function performance(D, Wfuel)
    R = - eta_prop / SFC * W / D * log(1 - Wfuel/W)
    return R
end

#### ---- All-at-once Solution ----
function solve_aao()
    @assert keys(global_variables)[1] == :R

    variables   = mergevar((; xcg=global_variables.xcg, R=global_variables.R), aero_local, struc_local)
    constraints = mergevar(aero_output, struc_output)
    idx         = indexbyname(variables)
    idg         = indexbyname(constraints)
    k           = upper(variables) - lower(variables)
    b           = lower(variables)

    function aao(g, x)
        # unscale
        x .*= k
        x .+= b

        # Aero
        D, load_4g, Cl, Cm, Cm_4g, q, Cm_al, Cm_al_4g = aero(x[idx.alpha], x[idx.delta_e], x[idx.twist], x[idx.xcg])
        g[idg.qpos] = -q/W
        @. g[idg.Cl] = Cl / 1.45 - 1
        g[idg.Cm_al] = Cm_al
        g[idg.Cm_al_4g] = Cm_al_4g
        g[idg.load_4g] = (2 * W - sum(load_4g))/2/W
        g[idg.Cm] = Cm
        g[idg.Cm_4g] = Cm_4g

        a1 = half_span / (pi * W) * ((load_4g .* sin.(  TLmesh.theta_c)) ⋅ TLmesh.theta_panel_size)
        a3 = half_span / (pi * W) * ((load_4g .* sin.(3*TLmesh.theta_c)) ⋅ TLmesh.theta_panel_size)
        
        # Struct
        load = 4 * W / half_span * (a1.*sin.(TLmesh.theta_c) + a3.*sin.(3*TLmesh.theta_c))
        xcg, sigma, Wfuel, Wwing = structures(x[idx.thickness], x[idx.Wt], load)
        @. g[idg.sigma] = abs(sigma) / sigma_failure - 1.
        g[idg.xcg] = (x[idx.xcg] - xcg)
        g[idg.Wfuel] = Wfuel / W - 1.
        g[idg.Wwing] = Wwing / W - 1.
        
        # Performance
        R = performance(D, Wfuel)
        x .-= b
        x ./= k

        # Objective
        return -R

    end

    Ng = len(constraints)
    Nx = len(variables)

    x0 = ini_scaled(variables)  # starting point
    g = zeros(len(constraints))  # starting point

    lx = zeros(Nx) # lower bounds on x
    ux = ones(Nx) # upper bounds on x
    lg = lower(constraints)
    ug = upper(constraints) # upper bounds on g
    ug[idg.xcg] = 0.
    options = SNOW.Options(derivatives=SNOW.CentralFD())

    xopt, fopt, info = minimize(aao, x0, Ng, lx, ux, lg, ug, options)

    # print result
    v = (upper(variables) - lower(variables)) .* xopt + lower(variables)
    v = unpack(v, idx)
    for k = keys(v)
        println("$k: $(v[k])")
    end
    c = unpack(g, idg)
    for k = keys(c)
        println("$k: $(c[k])")
    end

    return v
end

#### ---- Solution ----
@consts begin
    tailless_optimum = (;
        D = 44.535857314079685, 
        a1 = 0.8725964469247693, 
        a3 = -0.20726629666356344, 
        xcg = 3.912023450325287, 
        R = 5741.697478177579, 
        twist = [10.63646819832547, 9.98583555676163, 8.578137289289218, 7.984914143100919, 7.637829494329576, 7.420075796963346, 7.300820719288684, 7.336947685654138, 7.494792863212517, 7.071963950475457, 6.599451422214532, 6.078195888154806, 5.509562108922545, 4.8947157866769935, 4.2345417629166455, 3.528711592533222, 2.786857572652277, 1.9885910036632382, 1.144753612539409, -0.012915839725520684], 
        alpha = 8.156226315114909, 
        delta_e = -11.699951299009436, 
        thickness = [0.003952060271571667, 0.0035224933357481144, 0.0031001273620913563, 0.0026882960282519535, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665, 0.0026041666666666665], 
        Wt = 55.74934541169471)
    Zopt = NamedTuple(Pair(k,(tailless_optimum[k].-global_variables[k].lb)./(global_variables[k].ub-global_variables[k].lb)) for k=keys(global_variables))
    zopt = vcat(Zopt...)
end

#### ---- Collaborative Optimization ----
struct Tailless <: AbstractProblem end
BayesianCollaborativeOptimization.discipline_names(::Tailless) = (:struc, :aero)
BayesianCollaborativeOptimization.indexmap(::Tailless) = tailless_idz
BayesianCollaborativeOptimization.number_shared_variables(::Tailless) = 5
BayesianCollaborativeOptimization.objective_opt(::Tailless) = Zopt.R

function BayesianCollaborativeOptimization.subspace(::Tailless, ::Val{:aero}, z::AbstractArray, filename::String, 
    ipoptions::Dict{Any,Any} = Dict{Any,Any}("print_level"=>2, "file_print_level"=>5, "tol"=>1e-8, "output_file"=>"tmp.txt","linear_solver"=>"mumps"))
    """
    z is a NamedTuple of global variables
    """
    variables = mergevar((; xcg=aero_global.xcg, D=aero_global.D), aero_local)
    idx = indexbyname(variables)
    idz = indexbyname(aero_global)
    idg = indexbyname(aero_output)
    k = upper(variables) - lower(variables) 
    b = lower(variables)

    # unscale z
    kz = upper(aero_global) - lower(aero_global) 
    bz = lower(aero_global)

    function fun(g, x)
        g.= 0.
        # unscale
        x .*= k
        x .+= b

        # Aero
        D, load_4g, Cl, Cm, Cm_4g, q, Cm_al, Cm_al_4g = aero(x[idx.alpha], x[idx.delta_e], x[idx.twist], x[idx.xcg])

        g[idg.qpos] = -q/W
        @. g[idg.Cl] = (Cl / 1.45 - 1) / 1000.
        g[idg.Cm_al] = Cm_al
        g[idg.Cm_al_4g] = Cm_al_4g
        g[idg.load_4g] = (2 * W - sum(load_4g))/2/W
        g[idg.Cm] = Cm
        g[idg.Cm_4g] = Cm_4g

        # Compute Loads ROM
        a1 = half_span / (pi * W) * ((load_4g .* sin.(  TLmesh.theta_c)) ⋅ TLmesh.theta_panel_size)
        a3 = half_span / (pi * W) * ((load_4g .* sin.(3*TLmesh.theta_c)) ⋅ TLmesh.theta_panel_size)
        
        # rescale
        x .-= b
        x ./= k

        # objective function
        g[idg.D]  =  (D-bz[idz.D]) / kz[idz.D] - x[idx.D]
        g[idg.a1] = (a1-bz[idz.a1]) / kz[idz.a1]
        g[idg.a3] = (a3-bz[idz.a3]) / kz[idz.a3]
        f = ((x[idx.D]  - z[idz.D] )^2 + (g[idg.a1] - z[idz.a1] )^2+
             (g[idg.a3] - z[idz.a3])^2 + (x[idx.xcg] - z[idz.xcg])^2)
        return f
    end
    Ng = len(aero_output)
    Nx = len(variables)

    x0 = ini_scaled(variables)  # starting point
    x0[idx.xcg] = z[idz.xcg]
    x0[idx.D]   = z[idz.D]
    
    ipoptions["tol"] = 1e-8
    ipoptions["max_iter"] = 150
    ipoptions["output_file"] = filename
    # ipoptions["linear_solver"] = "ma97"
    options = SNOW.Options(derivatives=ForwardAD(), solver=SNOPT())#IPOPT(ipoptions))
    
    g = zeros(Ng)  # starting point
    lx = zeros(Nx) # lower bounds on x
    ux = ones(Nx) # upper bounds on x
    lg = lower(aero_output)
    ug = upper(aero_output) # upper bounds on g
    xopt, fopt, info = minimize(fun, x0, Ng, lx, ux, lg, ug, options)
    
    # Compute z star
    zs          = copy(z)
    fun(g, xopt)
    zs[idz.xcg] = xopt[idx.xcg]
    zs[idz.D]   = xopt[idx.D]
    zs[idz.a1]  = g[idg.a1]
    zs[idz.a3]  = g[idg.a3]
    
    viol = sum(max(0., lg[i]-g[i])+ max(0., g[i]-ug[i]) for i=1:Ng)
    # @assert (viol < 1e-6) "$viol"
    return zs
end

function BayesianCollaborativeOptimization.subspace(::Tailless, ::Val{:struc}, z::AbstractArray, filename::String,
    ipoptions::Dict{Any,Any} = Dict{Any,Any}("print_level"=>2, "file_print_level"=>5, "tol"=>1e-8, "output_file"=>"tmp.txt","linear_solver"=>"mumps"))

    variables = mergevar((; a1=struc_global.a1, a3=struc_global.a3, D=struc_global.D, R=struc_global.R), struc_local)
    idx = indexbyname(variables)
    idz = indexbyname(global_variables)
    idg = indexbyname(struc_output)
    k = upper(variables) - lower(variables) 
    b = lower(variables)

    kz = upper(global_variables) - lower(global_variables) 
    bz = lower(global_variables)
    
    function fun(g, x)
        # unscale
        x .*= k
        x .+= b

        # Struct
        load = 4 * W / half_span * (x[idx.a1].*sin.(TLmesh.theta_c) + x[idx.a3].*sin.(3*TLmesh.theta_c))
        xcg, sigma, Wfuel, Wwing = structures(x[idx.thickness], x[idx.Wt], load)
        @. g[idg.sigma] = abs(sigma) / sigma_failure - 1.
        g[idg.Wfuel] = Wfuel / W - 1.
        g[idg.Wwing] = Wwing / W - 1.
        
        # Performance
        R = performance(x[idx.D], Wfuel)

        # rescale
        x .-= b
        x ./= k
        g[idg.xcg] = (xcg-bz[idz.xcg]) / kz[idz.xcg]
        g[idg.R] = (R-bz[idz.R]) / kz[idz.R] - x[idx.R]

        return ((x[idx.D] - z[idz.D])^2 + (x[idx.a1] - z[idz.a1])^2+
                (x[idx.a3] - z[idz.a3])^2+(g[idg.xcg] - z[idz.xcg])^2+(x[idx.R] - z[idz.R])^2)
    end
    Ng = len(struc_output)
    Nx = len(variables)

    x0 = ini_scaled(variables)  # starting point
    x0[idx.D]  = z[idz.D]
    x0[idx.a1] = z[idz.a1]
    x0[idx.a3] = z[idz.a3]
    x0[idx.R]  = z[idz.R]
    
    ipoptions["tol"] = 1e-8
    ipoptions["max_iter"] = 500
    ipoptions["output_file"] = filename
    options = SNOW.Options(derivatives=ForwardAD(), solver=IPOPT(ipoptions))
    lx = zeros(Nx) # lower bounds on x
    ux = ones(Nx) # upper bounds on x
    lg = lower(struc_output)
    ug = upper(struc_output) # upper bounds on g
    xopt, fopt, info, out = minimize(fun, x0, Ng, lx, ux, lg, ug, options)

    # pack zs
    g = zeros(Ng) 
    fun(g, xopt)

    zs          = copy(z)
    zs[idz.D]   = xopt[idx.D]
    zs[idz.a1]  = xopt[idx.a1]
    zs[idz.a3]  = xopt[idx.a3]
    zs[idz.R]   = xopt[idx.R]
    zs[idz.xcg] = g[idg.xcg]
    
    viol = sum(max(0., lg[i]-g[i])+ max(0., g[i]-ug[i]) for i=1:Ng)
    # @assert (viol < 1e-6) "$viol"
    return zs
end

function solve_co()
    cotol = 1e-4
    ipoptions=Dict("print_level"=>2, "tol"=>1e-6, "max_iter"=>500)
    idz = indexbyname(global_variables)
    prob = Tailless()
    aero = Val(:aero)
    struc = Val(:struc)

    function cofun(g, df, dg, z)
        # Constraints
        zStar_aero = subspace(prob, aero, z,ipoptions)
        @. dg[1,:] = z-zStar_aero
        zStar_struc = subspace(prob, struc, z,ipoptions)
        @. dg[2,:] = z-zStar_struc
        g[1] = (dg[1,:] ⋅ dg[1,:])/2
        g[2] = (dg[2,:] ⋅ dg[2,:])/2

        # Objective
        df .= 0
        df[idz.R] = -1.
        return -z[idz.R]
    end

    Ng = 2
    Nz = len(global_variables)
    z0 = ini_scaled(global_variables)  # starting point
    lz = zeros(Nz) # lower bounds on z
    uz = ones(Nz) # upper bounds on z
    lg = [-Inf, -Inf]
    ug = [cotol, cotol] # upper bounds on g
    g = zeros(Ng)  # starting point

    co_options = Dict("tol"=>1e-4, "max_iter"=>50)
    options = SNOW.Options(derivatives=SNOW.UserDeriv(), solver=IPOPT(co_options))
    xopt, fopt, info = minimize(cofun, z0, Ng, lz, uz, lg, ug, options)

    # print result
    v = (upper(global_variables) - lower(global_variables)) .* xopt + lower(global_variables)
    v = unpack(v, idz)
    for k = keys(v)
        println("$k: $(v[k])")
    end
    return xopt
end
