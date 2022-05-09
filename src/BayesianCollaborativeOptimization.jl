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
using Evolutionary
using HouseholderNets
using CSV

include("var.jl")
export Var, ini, lower, upper, varnames, len, index, indexbyname, indexbygroup, len, mergevar, ini_scaled, get_scaled
export unscale_unpack, unpack, getvar!
include("train_hnet.jl")
export learn_feasible_set, save_ensemble, load_ensemble
include("bco.jl")
export bco, load_data, save_data, trim_data!
include("minimize_ei.jl")
export minimize_ei
end