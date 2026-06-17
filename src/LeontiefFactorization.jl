struct LeontiefFactorization{F} <: AbstractMatrixEntry
    factorization::F
    col_indices::DataFrame
    row_indices::DataFrame
    row_lookup::Dict{NamedTuple, Int}
    col_lookup::Dict{NamedTuple, Int}
end

function LeontiefFactorization(factorization::F, col_indices::DataFrame, row_indices::DataFrame) where {F}
    row_lookup = Dict(NamedTuple(row) => i for (i, row) in enumerate(eachrow(row_indices)))
    col_lookup = Dict(NamedTuple(row) => i for (i, row) in enumerate(eachrow(col_indices)))
    return LeontiefFactorization{F}(factorization, col_indices, row_indices, row_lookup, col_lookup)
end

function calculate_leontief_factorization(a::MatrixEntry)
    I_minus_A = I - a.data
    return LeontiefFactorization(lu(I_minus_A), a.col_indices, a.row_indices)
end

function Base.getproperty(m::LeontiefFactorization, sym::Symbol)
    if sym === :data
        n = size(m.row_indices, 1)
        I_mat = Matrix{Float64}(I, n, n)
        return m.factorization \ I_mat
    else
        return getfield(m, sym)
    end
end
