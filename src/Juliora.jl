module Juliora

using Tidier
using TidierFiles
using LinearAlgebra
using DataFrames
using CSV


export Eora, MRIO, parse_gloria, parse_gloria_sut, EnvironmentalExtension, groupby, aggregate, filter_rows, filter_cols, drop, drop!
export MatrixEntry, SeriesEntry
export filter_matrix, to_long_dataframe, from_long_dataframe, groupby_matrix, sum_by_country, sum_by_sector, add_calculated_column, pivot_matrix_to_wide, matrix_summary, country_summary

include("seriesentry.jl")
include("matrixentry.jl")
include("LeontiefFactorization.jl")
include("environmental_extension.jl")
include("mrio.jl")
include("parsers/parsers.jl")
using .Parser: parse_gloria, parse_gloria_sut
include("aggregation.jl")
include("analysis.jl")


end # module Juliora
