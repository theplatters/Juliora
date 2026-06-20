"""
    aggregate(mrio::MRIO, cols; dims::Int = 1, agg_func = sum)

Aggregate the components of an MRIO database by specified column names (e.g. `[:Sector]`).

# Arguments
- `mrio::MRIO`: The MRIO database to aggregate.
- `cols`: A Symbol or Vector{Symbol} specifying the grouping columns in the indices.
- `dims::Int=1`: Dimension along which to aggregate (1 for rows, 2 for columns).
- `agg_func`: Aggregation function to apply (default is `sum`).

# Returns
A new `MRIO` instance with aggregated components.
"""
function aggregate(mrio::MRIO, cols; dims::Int = 1, agg_func = sum)
    if dims == 1
        # Aggregating rows of Z/T
        T_agg = aggregate(groupby(mrio.Z, cols; dims = 1), agg_func)
        
        # VA columns correspond to sectors
        VA_agg = aggregate(groupby(mrio.VA, cols; dims = 2), agg_func)
        
        # FD/Y rows correspond to sectors
        Y_agg = aggregate(groupby(mrio.Y, cols; dims = 1), agg_func)
        
        # Determine total output of the columns of T_agg (which are not aggregated)
        x_cols = mrio.X.data
        
        # New row dimension is aggregated, so aggregate the output vector for rows
        x_agg = aggregate(groupby(mrio.X, cols), agg_func).data
        
    elseif dims == 2
        # Aggregating columns of Z/T
        T_agg = aggregate(groupby(mrio.Z, cols; dims = 2), agg_func)
        
        # VA columns correspond to sectors
        VA_agg = aggregate(groupby(mrio.VA, cols; dims = 2), agg_func)
        
        # FD/Y columns are final demand categories. Only aggregate if cols exist in col_indices.
        if all(c -> string(c) in names(mrio.Y.col_indices), cols)
            Y_agg = aggregate(groupby(mrio.Y, cols; dims = 2), agg_func)
        else
            Y_agg = mrio.Y
        end
        
        # Determine total output of the columns of T_agg (which are aggregated)
        x_cols = aggregate(groupby(mrio.X, cols), agg_func).data
        
        # Row dimension is not aggregated
        x_agg = mrio.X.data
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end
    
    # Calculate new technical coefficients A_agg = T_agg ./ x_cols'
    x_cols_safe = replace(x_cols, 0.0 => 1.0)
    A_agg = MatrixEntry(T_agg.data ./ x_cols_safe', T_agg.col_indices, T_agg.row_indices)
    
    # Calculate new Leontief factorization L_agg
    if size(A_agg.data, 1) == size(A_agg.data, 2)
        L_agg = calculate_leontief_factorization(A_agg)
    else
        L_agg = LeontiefFactorization(lu(Matrix{Float64}(I, 1, 1)), A_agg.col_indices, A_agg.row_indices)
    end
    
    # Aggregate Environmental Extension (env)
    # The columns of F and A correspond to sectors/countries.
    # Group env.F columns using cols only when aggregating the column dimension (dims == 2)
    F_agg = if dims == 2
        aggregate(groupby(mrio.env.F, cols; dims = 2), agg_func)
    else
        mrio.env.F
    end
    
    # Calculate new intensities env_A_agg = F_agg ./ x_cols_safe'
    env_A_agg = MatrixEntry(F_agg.data ./ x_cols_safe', F_agg.col_indices, F_agg.row_indices)
    env_agg = EnvironmentalExtension(F_agg, env_A_agg)
    
    return MRIO(
        A_agg,
        T_agg,
        VA_agg,
        Y_agg,
        L_agg,
        SeriesEntry(x_agg, T_agg.row_indices),
        env_agg
    )
end

function string_to_func(func::AbstractString)
    f = lowercase(func)
    if f == "sum" || f == "base.sum"
        return sum
    elseif f == "mean" || f == "statistics.mean"
        return Statistics.mean
    elseif f == "median" || f == "statistics.median"
        return Statistics.median
    elseif f == "std" || f == "statistics.std"
        return Statistics.std
    elseif f == "var" || f == "statistics.var"
        return Statistics.var
    elseif f == "min" || f == "minimum" || f == "base.minimum"
        return minimum
    elseif f == "max" || f == "maximum" || f == "base.maximum"
        return maximum
    else
        throw(ArgumentError("Unknown aggregation function string: $func"))
    end
end

function aggregate(gm::GroupedMatrixEntry, func::AbstractString)
    return aggregate(gm, string_to_func(func))
end

function aggregate(gs::GroupedSeriesEntry, func::AbstractString)
    return aggregate(gs, string_to_func(func))
end
