abstract type AbstractProblem end

"""
Default objective function to maximize.
"""
function objective(::AbstractProblem, z::AbstractArray, grad=nothing)
    if typeof(grad) <: AbstractArray
        grad .= 0.
        grad[1] = 1.
    end
    return z[1]
end
objective_lowerbound(::AbstractProblem) = 0.
objective_upperbound(::AbstractProblem) = 1.
objective_opt(::AbstractProblem) = [1.]

"""
Methods that must be overridden by each concrete subtype
"""
discipline_names(::AbstractProblem) = throw("unimplemented")
number_shared_variables(::AbstractProblem) = throw("unimplemented")
indexmap(::AbstractProblem) = throw("unimplemented")
subspace(pb::AbstractProblem, dis::Val{T} where T, z::AbstractArray, filename::String) = throw("subspace not implemented for discipline $dis of $pb.")

