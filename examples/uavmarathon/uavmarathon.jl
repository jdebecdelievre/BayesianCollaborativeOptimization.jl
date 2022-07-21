using Pkg
Pkg.activate("/home/adgboost/.julia/dev/BayesianCollaborativeOptimization/")
using LinearAlgebra
using JLD2
using BayesianCollaborativeOptimization
using Parameters
using Snopt
using SNOW
# Adapted in July 2022 from a Matlab code written by Prof. Ilan Kroo

### Analysis function ###
function PropulsionModel(U, Volts)
    
    # Using a fit to the propeller data from the UIUC database and the motor constants published by HobbyKing,
    # this code estimates the motor and propeller performance at the RPM for which motor and prop torque match.
    # Note that one could also match the torques using constrained optimization. 
    # Sample calling: [Mbatt_rqd,FlightTime,RPM,T,Q,Pin,I,etap,etam,eta] = PropulsionModel(14,0.6)
    # Inputs: U (m/sec), M_battery (kg)
    # Outputs: Mbatt_rqd (kg), FlightTime (hr), RPM (rev/min), T (N), Q (N m), Pin (W), I (A), etap, etam, eta (efficiency of prop, motor, total)

    # Note that with a reasonable efficiency the propulsion system can supply 2
    # N of thrust.  This means that the airplane might weigh 20-40N or a mass
    # of 2-4kg. To fly 100km may require .6 kg of batteries.  Perhaps we should reduce M_fixed. We'll see.

    # Constants
    rho=1.225
    radius=0.1143
    D = 2*radius
    Range = 42000 # in kg/m^3 and m
    I0=1.0
    R=0.07
    Kv_RPM=950
    SpNRG=200 # in A,Ω,RPM/V,V,Wh/kg
    # Graupner 9x5 Slim CAM propeller data (similar to Aeronaut 9x5 folding prop)
    # Quadratic fit to UIUC data at about 4000 RPM (coefficients not very sensitive to RPM at fixed J)
    # Motor data from Hobby King: Turnigy D2836/9 950KV Brushless Outrunner 7.4V - 14.8V Max current: 23.2A
    # No load current: 1A Max power: 243W Internal resistance: 0.070 ohm Mass: 70g (including connectors)

    # Propeller torque from fit:
    # Q  = rho * 4/pi^3 * radius^3 * (0.0363*omega^2*radius^2 + 0.0147*U*omega*radius*pi - 0.0953*U^2*pi^2)
    #   = ap + bp * omega + cp * omega^2  with:
    k = rho * 4/pi^3 * radius^3
    ap = -0.0953*k*U^2*pi^2
    bp = 0.0147*k*U*radius*pi
    cp = 0.0363*k*radius^2

    # Motor constants from input values of Kv_RPM, I0, R, Volts
    # Torque may be written: Qm = am + bm*omega with:
    Kv = Kv_RPM * 2*pi/60 # rad/sec/Volt
    am = Volts/(R*Kv)-I0/Kv
    bm = -1/(R*Kv^2)

    # Solve for omega:
    omega = (-(bp-bm) + sqrt((bp-bm)^2 - 4*cp*(ap-am)))/(2*cp)
    RPM = omega * 60/(2*pi)
    n = RPM/60

    # Motor performance:
    I = Volts/R - omega/(R*Kv)
    Qm = am + bm*omega
    Pin = I*Volts
    etam = Qm*omega / Pin

    # Prop performance:
    J_UI = U/(n*D)
    CT_UI = 0.090 - 0.0735*J_UI - 0.1141*J_UI^2
    T = CT_UI * rho * n^2 * D^4
    Q = Qm
    Pout = T*U
    etap = Pout/(Q*omega)

    # Overall performance:
    eta = Pout/Pin

    # Battery Mass in kg, Pin in W, Flight time in sec, SpNRG in Wh/kg
    FlightTime = Range/U/3600  # hr
    Mbattery_rqd = Pin * FlightTime / SpNRG # Rqd battery mass (kg)

    return Mbattery_rqd,FlightTime,RPM,T,Q,Pin,I,etap,etam,eta, J_UI
end
##
function WingModel(U,S,b,M_battery)

        # e.g. [D, CL, LD, M_wing, M_total] = WingModel(13,1.2,10,0.6)
        # For debugging:
        # [c,b,Re_wing,StressToMaxStress,deflToSemispan,D,CL,LD,CDp_wing,CDp_fuse,CDp_total,CDi,CD_total,M_fixed,M_foam,M_spar,M_wing,M_total] = WingModel(15,0.35,50,.2

        # Mass/Weight Model
        g = 9.8 
        FR_fuse = 6
        rho_foam = 2.5/2.2 * 3.281^3 # 40 kg/m^3
        rho_carbon = 1380 # kg/m^3 based on Dragon Plate Braided Tubes w/50# resin.
        rho_atm = 1.225 # kg/m^3
        t_c = .12
        taper = 0.75
        E = 2.344e+11  # N/m^2
        MaxStress = 4.413e+9 # N/m^2
        M_motor = 0.7 # kg
        M_prop = 0.1 # kg
        M_fixed = 3 + M_motor + M_prop + M_battery # kg
        lift = M_fixed*g
        
        AR = b^2 / S
        c = S/b
        croot = c * 2/(1+taper) # S = b*(croot+ctip)/2 = b*croot*(1+taper)/2_
        Sref = S
        Dia_spar = 0.5 * c * t_c # 50# of section thickness at root (taper>0.5)
        r_spar = Dia_spar/2
        t_spar = 0.045*2.54/100 # m
        
        # Compute required spar gauge based on stress or deflection
        # Uniform load: Mroot = qL^2/2  with L = b/2 q = lift/span  
        # and qL the lift on the semispan (lift/2).
        # The tip deflection  is d = q L^4 / (8EI) or d/L = q L^3 / 8E
        # For the braided tubes: E=34 Msi (2.344e+11 N/m^2) sigmamax=640ksi (4.413e+9 N/m^2)
        # Include bending only from fixed items.
        # rootmoment = qL^2/2 = lift/b (b/2)^2 / 2 = lift*b/8
        q = lift / b
        rootmoment = lift*b/8
        I = pi*r_spar^3*t_spar
        StressToMaxStress = rootmoment*r_spar/I / MaxStress # Should be less than .17 (4g with 1.5 factor)
        treq_stress = t_spar*StressToMaxStress/0.17
        deflect = lift * b^4 / (64*E*I)  # 
        DeflToSemispan = deflect / (b/2) # should be < .25 at 3+ g's, so at 1 g should be < 0.0
        treq_defl = t_spar*DeflToSemispan/0.07
        treq_spar = max(max(t_spar,treq_stress),treq_defl)
        M_foam = rho_foam * S * c*t_c*.44 # Should include taper
        M_spar = Dia_spar * pi * treq_spar * b * rho_carbon
        M_wing = M_foam + M_spar
        M_total = M_fixed + M_wing
        
        # Aero model
        Dia_fuse = 0.1 # m
        CLmax = 1.0
        nu = 1.46E-5
        e = 0.8
        Re_wing = U*c/nu
        Cf_wing = 0.455/(log10(Re_wing))^2.58 # ≈ 0.074/Re^0.2  -- fully turbulent
        CDp_wing = (1+2*t_c)*Cf_wing*2.04 * 1.3 # Include markups for t/c and CL (via e) and tails
        len_fuse = Dia_fuse*FR_fuse
        Re_fuse = U*len_fuse/nu
        Cf_fuse = 0.455/(log10(Re_fuse))^2.58 # ≈ 0.074/Re^0.2  -- fully turbulent
        Swet_fuse = pi*Dia_fuse*len_fuse
        CDp_fuse = Cf_fuse * Swet_fuse/Sref * 1.22 # See PASS for body form factor. FF = 1.22 at FR=6
        CDp_total = CDp_wing + CDp_fuse
        
        CL = 2*M_total*g / (S*rho_atm*U^2)  
        CDp_wing = CDp_wing + 0.1*max(0, CL-CLmax)^2  # Drag penalty near stall
        CDi = CL^2/(pi*AR*e)
        CD_total = CDp_wing+CDp_fuse+CDi
        LD = CL/CD_total
        D = M_total*g / LD
        return D, CL, LD, M_wing, M_total
end

    # x0 = [U,S,AR]; lb = [5,0.1,2]; ub = [40,5,30];

## BCO API formating
@consts begin 
    V = (;
    # Global
    V     = Var(lb=5., ub=15., group=(:wing,:prop)),
    D     = Var(lb=1., ub=6., group=(:wing,:prop)),
    Mbat  = Var(lb=0.1, ub=5., group=(:wing,:prop)),
    # Wing
    S     = Var(lb=0.1, ub=1., group=:wing),
    b     = Var(lb=0.1, ub=5., group=:wing),
    # Prop
    volts = Var(lb=0.1, ub=9., group=:prop)
    )
    lb = lower(V)
    ub = upper(V)

    wing_variables = subset(V, :wing)
    prop_variables = subset(V, :prop)
    disciplines = (:wing, :prop)
    
    idz = (; wing=[1,2,3], prop=[1,2,3])
    idz_d = NamedTuple{disciplines}(map(d->indexbyname(subset(V, d)), disciplines))
    idx = indexbyname(V)
    
    opt = (; V=13.7133, D=2.1525, Mbat=0.2434158)
    zopt = ([opt.V, opt.D, opt.Mbat] - lb[1:3]) ./ (ub[1:3] - lb[1:3])
end

struct UAVmarathon <: AbstractProblem end
BayesianCollaborativeOptimization.discipline_names(::UAVmarathon) = disciplines
BayesianCollaborativeOptimization.indexmap(::UAVmarathon) = idz
BayesianCollaborativeOptimization.number_shared_variables(::UAVmarathon) = 3
BayesianCollaborativeOptimization.objective_opt(::UAVmarathon) = zopt[1]


function BayesianCollaborativeOptimization.subspace(::UAVmarathon, ::Val{:wing}, z0::AbstractArray, filename::String)
    function fun(g,z)
        v = unscale_unpack(z, idz_d.wing, wing_variables)
        # D, CL, LD, M_wing, M_total = WingModel(13.7133, 0.397, 2.7574, 0.2434158) 
        D, CL, LD, M_wing, M_total = WingModel(v.V, v.S, v.b, v.Mbat) 
        g[1] = D - v.D
        return (z[1:3]-z0)⋅(z[1:3]-z0)
    end
    Nx = 5
    lx = zeros(Nx)
    ux = ones(Nx)
    Ng = 1
    lg = [-Inf]
    ug = [0.]
    ipoptions=Dict{Any,Any}("print_level"=>3)
    # options = SNOW.Options(derivatives=SNOW.CentralFD(), solver=SNOW.IPOPT(ipoptions))
    options = SNOW.Options(derivatives=SNOW.CentralFD(), solver=SNOW.SNOPT())
    x0  = zeros(Nx) .+ 1/2
    xopt, fopt, info = SNOW.minimize(fun, x0, Ng, lx, ux, lg, ug, options)
    return xopt[1:3]
end

function BayesianCollaborativeOptimization.subspace(::UAVmarathon, ::Val{:prop}, z0::AbstractArray, filename::String)
    function fun(g,z)
        v = unscale_unpack(z, idz_d.prop, prop_variables)
        Mbattery_rqd,FlightTime,RPM,T,Q,Pin,I,etap,etam,eta, J_UI = PropulsionModel(v.V, v.volts)
        g[1] = (v.D - T)
        g[2] = (Mbattery_rqd - v.Mbat)
        g[3] = J_UI
        return (z[1:3]-z0)⋅(z[1:3]-z0)
    end
    Nx = 4
    lx = zeros(Nx)
    ux = ones(Nx)
    Ng = 3
    lg = [-Inf, -Inf, 0.1]
    ug = [0., 0., 0.8]
    ipoptions=Dict{Any,Any}("print_level"=>3)
    options = SNOW.Options(derivatives=SNOW.CentralFD(), solver=SNOW.SNOPT())
    x0  = zeros(Nx) .+ 1/2
    xopt, fopt, info = SNOW.minimize(fun, x0, Ng, lx, ux, lg, ug, options)
    return xopt[1:3]
end

##
for i=1:20
    options = SolveOptions(n_ite=15, ini_samples=1, warm_start_sampler=i, savedir="xpu$i/bco")
    solver = BCO(UAVmarathon(), training_tol=1e-3, N_epochs=300_000, stepsize=100., 
                            dropprob=0.02, αlr=0.97, nlayers=20, tol=1e-3)
    solve(solver, options, terminal_print=false)
    
    ## 
    options = SolveOptions(n_ite=150, ini_samples=1, warm_start_sampler=i, savedir="xpu$i/sqp")
    solver = SQP(UAVmarathon(), λ=1.,tol=1e-6)
    solve(solver, options, terminal_print=false)
    
    ##
    options = SolveOptions(n_ite=150, ini_samples=1, warm_start_sampler=i, savedir="xpu$i/admm")
    solver = ADMM(UAVmarathon(), ρ=.1)
    solve(solver, options, terminal_print=false)
end
# ddir = (; prop="xpu/prop.jld2", wing="xpu/wing.jld2")
# data = map(load_data, ddir);
# obj, Z, fsb, sqJ = load("xpu/obj.jld2","obj","Z","fsb","sqJ");