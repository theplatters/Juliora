module Juliora

using Tidier
using TidierFiles
using LinearAlgebra
using DataFrames
using CSV


export Eora, EnvironmentalExtension, groupby, aggregate, filter_rows, filter_cols, drop, drop!
export MatrixEntry, Matrixentry, SeriesEntry
export filter_matrix, to_long_dataframe, from_long_dataframe, groupby_matrix, sum_by_country, sum_by_sector, add_calculated_column, pivot_matrix_to_wide, matrix_summary, country_summary

include("parsers/parsers.jl")
include("seriesentry.jl")
include("matrixentry.jl")
const Matrixentry = MatrixEntry
include("LeontiefFactorization.jl")
include("environmental_extension.jl")
include("eora.jl")
include("aggregation.jl")
include("analysis.jl")


end # module Juliora
