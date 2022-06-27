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

"""
Methods that must be overridden by each concrete subtype
"""
discipline_names(::AbstractProblem) = throw("unimplemented")
number_shared_variables(::AbstractProblem) = throw("unimplemented")
indexmap(::AbstractProblem) = throw("unimplemented")
subspace(::AbstractProblem, ::Val{T} where T, z::AbstractArray, filename::String) = throw("unimplemented")

