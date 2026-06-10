using Test
using Juliora
import Juliora as IO
using DataFrames
using LinearAlgebra
using Tidier

@testset "Juliora.jl" begin
    @testset "MatrixEntry Tests" begin
        include("test_matrixentry.jl")
    end
    
    @testset "SeriesEntry Tests" begin
        include("test_seriesentry.jl")
    end
    
    @testset "Boolean Indexing Tests" begin
        include("test_boolean_indexing.jl")
    end
    
    @testset "Filtering Tests" begin
        include("test_filtering.jl")
    end
    
    @testset "Environmental Extension Tests" begin
        include("test_environmental_extension.jl")
    end
    
    @testset "Eora Database Tests" begin
        include("test_eora.jl")
    end
    
    @testset "Analysis Functions Tests" begin
        include("test_analysis_functions.jl")
    end

    @testset "Gloria Parser Tests" begin
        include("test_gloria.jl")
    end
end