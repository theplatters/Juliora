"""
    aggregate(mrio::MRIO, by::DataFrame; dims::Int=1, agg_func=sum)::Eora
Aggregate the components of an Eora database by specified dimensions.

# Arguments
- `eora::Eora`: The Eora database to aggregate.
- `by::DataFrame`: A DataFrame specifying the grouping variables.
- `dims::Int=1`: Dimension along which to aggregate (1 for rows, 2 for columns).
- `agg_func`: Aggregation function to apply (default is `sum`).

# Returns
A new `MRIO` instance with aggregated components.

# Example
```julia
ag = groupby(eora.A, [:Sector], dims=2)
eora_agg = aggregate(eora, ag; dims=2)
```
"""
function aggregate(mrio::MRIO, by::DataFrame; dims::Int = 1, agg_func = sum)
    T_agg = aggregate(eora.T, by; dims = dims, agg_func = agg_func)
    V_agg = aggregate(eora.V, by; dims = dims, agg_func = agg_func)
    Y_agg = aggregate(eora.Y, by; dims = dims, agg_func = agg_func)
    L_agg = aggregate(eora.L, by; dims = dims, agg_func = agg_func)

    x_agg = vec(sum(T_agg.data, dims = 2) + sum(Y_agg.data, dims = 2))
    F_agg = eora.env.F ./ replace(eora.env.x, 0.0 => 1.0) .* replace(x_agg, 0.0 => 1.0)

    env_agg = EnvironmentalExtension(F_agg, x_agg, A_agg.row_indices)

    return MRIO(A_agg, T_agg, V_agg, Y_agg, L_agg, SeriesEntry(x_agg, A_agg.row_indices), env_agg)
end
