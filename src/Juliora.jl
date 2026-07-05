module Juliora

using Tidier
using TidierFiles
using LinearAlgebra
using DataFrames
using CSV
using Statistics


export Eora, Gloria, MRIO, parse_gloria, parse_gloria_sut, EnvironmentalExtension, groupby, aggregate, filter_rows, filter_cols, drop, drop!
export MatrixEntry, SeriesEntry
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

include("seriesentry.jl")
include("matrixentry.jl")
include("LeontiefFactorization.jl")
include("environmental_extension.jl")
include("mrio.jl")
include("parsers/parsers.jl")
using .Parser: parse_gloria, parse_gloria_sut
include("aggregation.jl")
include("analysis.jl")
include("tidier_integrations.jl")

# R helper functions
function make_named_tuple(keys::Union{String, Vector{String}}, values::Union{Vector, Any})
    keys_vec = keys isa Vector ? Symbol.(keys) : [Symbol(keys)]
    values_vec = values isa Vector ? values : [values]
    return NamedTuple(keys_vec .=> values_vec)
end

function make_named_tuple_vector(keys_list::Vector, values_list::Vector)
    return [make_named_tuple(keys_list[i], values_list[i]) for i in 1:length(keys_list)]
end

export make_named_tuple, make_named_tuple_vector

end # module Juliora
