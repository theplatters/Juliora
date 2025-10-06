

struct LeontiefFactorization <:AbstractMatrixEntry
    factorization::LU{Float64, Matrix{Float64}, Vector{Int64}}
    col_indices::DataFrame
    row_indices::DataFrame
end

function calculate_leontief_factorization(a::MatrixEntry)
    I_minus_A = I - a.data
    return LeontiefFactorization(lu(I_minus_A), a.col_indices, a.row_indices)
end