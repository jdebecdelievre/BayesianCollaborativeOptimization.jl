using BayesianCollaborativeOptimization
using LinearAlgebra
using JLD2
using LaTeXStrings
using Statistics
using Parameters
##
@consts begin
    idz_twins = (; A=[1,2], B=[1,2])
    cA = [0.5, 0.25]
    cB = [0.5, 0.75]
    opt = [sqrt(-0.25^2+.4^2)+0.5, 0.5]
end

struct TwinDisks <: AbstractProblem end # singleton type
BayesianCollaborativeOptimization.discipline_names(::TwinDisks) = (:A, :B)
BayesianCollaborativeOptimization.indexmap(::TwinDisks) = idz_twins
BayesianCollaborativeOptimization.number_shared_variables(::TwinDisks) = 2
BayesianCollaborativeOptimization.subspace(::TwinDisks, ::Val{:A}, z::AbstractArray,s::String) = z - ([(z[1]-cA[1]), (z[2]-cA[2])] / norm([(z[1]-cA[1]), (z[2]-cA[2])])).*max(0, sqrt((z[1]-cA[1])^2 +(z[2]-cA[2])^2)-.4)
BayesianCollaborativeOptimization.subspace(::TwinDisks, ::Val{:B}, z::AbstractArray,s::String) = z - ([(z[1]-cB[1]), (z[2]-cB[2])] / norm([(z[1]-cB[1]), (z[2]-cB[2])])).*max(0, sqrt((z[1]-cB[1])^2 +(z[2]-cB[2])^2)-.4)
