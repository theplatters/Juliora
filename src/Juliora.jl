module Juliora

using Tidier
using TidierFiles
using LinearAlgebra
using DataFrames
using CSV


include("seriesentry.jl")
include("matrixentry.jl")
include("LeontiefFactorization.jl")
include("environmental_extension.jl")
include("eora.jl")
include("aggregation.jl")


end # module Juliora

