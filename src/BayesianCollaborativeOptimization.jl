module BayesianCollaborativeOptimization

using LinearAlgebra
using StaticArrays
using StatsBase
using Random
using Sobol

using JLD2
using CSV

using Evolutionary
using NLopt # Possible improvement: write my own quasi-Newton for unconstrained EIc maximization

using HouseholderNets
using Parameters
using TimerOutputs
using Printf
using OptimUtils

include("problem.jl")
export AbstractProblem, indexmap, objective, discipline_names, indexmap, subspace, number_shared_variables, objective_lowerbound, objective_upperbound, objective_opt

include("utils.jl")
export get_metrics, datacheck

include("solve.jl")
export solve, load_data, save_data, trim_data!, SolveOptions

import SNOW # Possible improvement: replace by Ipopt direct wrapper to avoid SNOW dependencies
include("baselines/sqp.jl")
export SQP

include("baselines/admm.jl")
export ADMM

include("bco.jl")
export BCO

include("train_hnet.jl")
export learn_feasible_set, save_ensemble, load_ensemble

include("maximize_ei.jl")
export maximize_ei, eic, eic_cache

end