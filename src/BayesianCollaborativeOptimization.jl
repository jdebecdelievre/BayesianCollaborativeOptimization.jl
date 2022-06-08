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

include("var.jl")
export Var, ini, lower, upper, varnames, len, index, indexbyname, indexbygroup, len, mergevar, ini_scaled, get_scaled
export unscale_unpack, unpack, getvar!, scale
include("train_hnet.jl")
export learn_feasible_set, save_ensemble, load_ensemble
include("bco.jl")
export bco, load_data, save_data, trim_data!, BCOoptions
include("minimize_ei.jl")
export minimize_ei
include("nlopt_ei.jl")
export maximize_ei
end