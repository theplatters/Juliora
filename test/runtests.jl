using Test
using Juliora
using DataFrames
using LinearAlgebra

# Include all test modules
include("test_matrixentry.jl")
include("test_seriesentry.jl") 
include("test_boolean_indexing.jl")
include("test_filtering.jl")
include("test_tidier_integration.jl")
include("test_environmental_extension.jl")
include("test_eora.jl")
include("test_analysis_functions.jl")

@testset "Juliora.jl" begin
    @testset "MatrixEntry Tests" begin
        test_matrixentry()
    end
    
    @testset "SeriesEntry Tests" begin
        test_seriesentry()
    end
    
    @testset "Boolean Indexing Tests" begin
        test_boolean_indexing()
    end
    
    @testset "Filtering Tests" begin
        test_filtering()
    end
    
    @testset "TidierData Integration Tests" begin
        test_tidier_integration()
    end
    
    @testset "Environmental Extension Tests" begin
        test_environmental_extension()
    end
    
    @testset "Eora Database Tests" begin
        test_eora()
    end
    
    @testset "Analysis Functions Tests" begin
        test_analysis_functions()
    end
end