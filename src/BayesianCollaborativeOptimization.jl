module BayesianCollaborativeOptimization

using BenchmarkTools, LinearAlgebra, Plots
using StaticArrays
using FiniteDiff
using Random
using StatsBase
using Printf
using DataFrames
using JLD2
using Sobol
using Plots
ENV["GKSwstype"] = "100"
using Evolutionary
using HouseholderNets
using CSV
using Parameters
using NLopt
using TimerOutputs

include("utils.jl")
export get_metrics, datacheck

include("var.jl")
export Var, ini, lower, upper, varnames, len, index, indexbyname, indexbygroup, len, mergevar, ini_scaled, get_scaled
export unscale_unpack, unpack, getvar!, scale, subset

include("problem.jl")
export AbstractProblem, indexmap, objective, discipline_names, indexmap, subspace, number_shared_variables

include("solve.jl")
export solve, load_data, save_data, trim_data!, SolveOptions

include("sqp.jl")
export SQP

include("admm.jl")
export ADMM

include("bco.jl")
export BCO

include("train_hnet.jl")
export learn_feasible_set, save_ensemble, load_ensemble

include("maximize_ei.jl")
export maximize_ei, eic, eic_cache

end