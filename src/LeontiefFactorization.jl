"""
    LeontiefFactorization{F} <: AbstractMatrixEntry

LU factorization of `I - A` with labeled row/column indices. Constructed via
`calculate_leontief_factorization(a::MatrixEntry)` or the 3-argument
constructor `(factorization, col_indices, row_indices)`.

!!! warning "The cached inverse is shared state"
    `getproperty(lf, :data)` materializes the full Leontief inverse on first
    access and caches it; every subsequent access returns *the same matrix
    object*. Treat `lf.data` as READ-ONLY — mutating it in place corrupts the
    result of all later accesses. Likewise treat `row_indices`/`col_indices`
    as immutable: modifying them in place silently invalidates the cached
    inverse and the dimension checks performed at construction time.
"""
struct LeontiefFactorization{F} <: AbstractMatrixEntry
    factorization::F
    col_indices::DataFrame
    row_indices::DataFrame
    row_lookup::Dict{NamedTuple, Int}
    col_lookup::Dict{NamedTuple, Int}
    inverse_cache::Base.RefValue{Union{Nothing, Matrix{Float64}}}
end

function LeontiefFactorization(factorization::F, col_indices::DataFrame, row_indices::DataFrame) where {F}
    row_lookup = _build_lookup(row_indices, "row")
    col_lookup = _build_lookup(col_indices, "column")
    return LeontiefFactorization{F}(
        factorization,
        col_indices,
        row_indices,
        row_lookup,
        col_lookup,
        Ref{Union{Nothing, Matrix{Float64}}}(nothing),
    )
end

function calculate_leontief_factorization(a::MatrixEntry)
    I_minus_A = I - a.data
    return LeontiefFactorization(lu(I_minus_A), a.col_indices, a.row_indices)
end

"""
    solve_leontief(factorization::LeontiefFactorization, final_demand::AbstractVecOrMat{<:Number})

Solve the Leontief system for a numeric final-demand vector or matrix.
"""
function solve_leontief(
        factorization::LeontiefFactorization,
        final_demand::AbstractVecOrMat{<:Number}
    )
    return factorization.factorization \ final_demand
end

"""
    sum_rows(x::AbstractMatrix{<:Number})

Return the sum of each row of a numeric matrix as a vector.
"""
sum_rows(x::AbstractMatrix{<:Number}) = vec(sum(x; dims = 2))
sum_rows(x::MatrixEntry) = sum_rows(x.data)
sum_rows(x::LeontiefFactorization) = sum_rows(x.data)

"""
    sum_cols(x::AbstractMatrix{<:Number})

Return the sum of each column of a numeric matrix as a vector.
"""
sum_cols(x::AbstractMatrix{<:Number}) = vec(sum(x; dims = 1))
sum_cols(x::MatrixEntry) = sum_cols(x.data)
sum_cols(x::LeontiefFactorization) = sum_cols(x.data)

function Base.getproperty(m::LeontiefFactorization, sym::Symbol)
    if sym === :data
        # Materializing F\\I costs O(n^3): compute once and serve the cached
        # matrix on every later access.
        cache = getfield(m, :inverse_cache)
        if isnothing(cache[])
            n = size(getfield(m, :row_indices), 1)
            cache[] = getfield(m, :factorization) \ Matrix{Float64}(I, n, n)
        end
        return cache[]
    else
        return getfield(m, sym)
    end
end
