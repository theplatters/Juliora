module Juliora

using Tidier
using TidierFiles
import TidierData
using LinearAlgebra
using DataFrames
using CSV
using Statistics
using Tables


export Eora, Gloria, MRIO, parse_gloria, parse_gloria_sut, save_gloria_cache, load_gloria_cache, EnvironmentalExtension, groupby, aggregate, filter_rows, filter_cols, drop, drop!
export MatrixEntry, SeriesEntry
export solve_leontief, sum_rows, sum_cols
export filter_matrix, to_long_dataframe, from_long_dataframe, groupby_matrix, sum_by_country, sum_by_sector, add_calculated_column, pivot_matrix_to_wide, matrix_summary, country_summary, environmental_impact, induced_production
export countries, country, sectors, sector, stressors, stressor
export @filter_rows, @filter_cols, @mutate_rows, @mutate_cols, @select_rows, @select_cols, @rename_rows, @rename_cols, @slice_rows, @slice_cols
export update_row_indices, update_col_indices


function safe_dataframe(df)
    if df isa DataFrame
        return df
    end
    colnames = propertynames(df)
    cols = Any[]
    for colname in colnames
        col = getproperty(df, colname)
        if col isa AbstractArray
            push!(cols, collect(col))
        else
            push!(cols, [col])
        end
    end
    return DataFrame(cols, collect(colnames))
end

export safe_dataframe

"""
    _partial_key_match(full_key::NamedTuple, partial_key::NamedTuple)

Internal helper: true when every field of `partial_key` exists in `full_key`
with an `isequal` value. `isequal` semantics make `missing` and `NaN`
metadata values match themselves instead of throwing or silently missing.
"""
function _partial_key_match(full_key::NamedTuple, partial_key::NamedTuple)
    return all(k -> haskey(full_key, k) && isequal(full_key[k], partial_key[k]), keys(partial_key))
end

"""
    _build_lookup(indices::DataFrame, what::String)

Internal helper: build a `Dict{NamedTuple,Int}` row-label lookup for an
index frame. When two rows share the same full label, the later row shadows
the earlier one in the lookup (last wins); a single `@warn` naming an
example duplicate key is emitted since the first duplicate becomes
unreachable through label indexing.
"""
function _build_lookup(indices::DataFrame, what::String)
    lookup = Dict{NamedTuple, Int}()
    n_duplicates = 0
    example = nothing
    for (i, row) in enumerate(eachrow(indices))
        key = NamedTuple(row)
        if haskey(lookup, key)
            n_duplicates += 1
            if isnothing(example)
                example = (key = key, first_seen_at_row = lookup[key], duplicate_at_row = i)
            end
        end
        # Last wins (matches the previous Dict-comprehension behavior).
        lookup[key] = i
    end
    if n_duplicates > 0
        @warn "duplicate $what labels collapse in lookups (later rows shadow earlier ones)" n_duplicates example
    end
    return lookup
end

include("seriesentry.jl")
include("matrixentry.jl")
include("LeontiefFactorization.jl")
include("environmental_extension.jl")
include("mrio.jl")
include("parsers/parsers.jl")
using .Parser: parse_gloria, parse_gloria_sut
using .Parser: save_gloria_cache, load_gloria_cache
include("aggregation.jl")
include("analysis.jl")
include("tidier_integrations.jl")

# R helper functions
function make_named_tuple(keys::Union{String, Vector{String}}, values::Union{Vector, Any})
    keys_vec = keys isa Vector ? Symbol.(keys) : [Symbol(keys)]
    values_vec = values isa Vector ? values : [values]
    if length(keys_vec) != length(values_vec)
        throw(ArgumentError("make_named_tuple: got $(length(keys_vec)) keys but $(length(values_vec)) values; keys and values must have the same length"))
    end
    return NamedTuple(keys_vec .=> values_vec)
end

function make_named_tuple_vector(keys_list::Vector, values_list::Vector)
    if length(keys_list) != length(values_list)
        throw(ArgumentError("make_named_tuple_vector: got $(length(keys_list)) key entries but $(length(values_list)) value entries; both lists must have the same length"))
    end
    return [make_named_tuple(keys_list[i], values_list[i]) for i in 1:length(keys_list)]
end

export make_named_tuple, make_named_tuple_vector

end # module Juliora
