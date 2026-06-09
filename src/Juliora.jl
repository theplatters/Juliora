module Juliora

using Tidier
using TidierFiles
using LinearAlgebra
using DataFrames
using CSV


export Eora, EnvironmentalExtension, groupby, aggregate, filter_rows, filter_cols, drop, drop!

include("parsers/parsers.jl")
include("seriesentry.jl")
include("matrixentry.jl")
include("LeontiefFactorization.jl")
include("environmental_extension.jl")
include("eora.jl")
include("aggregation.jl")


end # module Juliora
