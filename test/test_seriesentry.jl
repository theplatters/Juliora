function test_seriesentry()
    @testset "SeriesEntry Constructor" begin
        # Test basic construction
        data = [100.0, 200.0, 150.0, 300.0]
        indices = DataFrame(
            Country = ["USA", "CHN", "DEU", "JPN"],
            Sector = ["Agriculture", "Manufacturing", "Services", "Mining"],
            GDP = [20000.0, 14000.0, 4000.0, 5000.0]
        )
        
        series = SeriesEntry(data, indices)
        
        @test length(series.data) == 4
        @test series.data == data
        @test nrow(series.col_indices) == 4
        @test series.col_indices.Country == ["USA", "CHN", "DEU", "JPN"]
        @test series.col_indices.Sector == ["Agriculture", "Manufacturing", "Services", "Mining"]
        @test series.col_indices.GDP == [20000.0, 14000.0, 4000.0, 5000.0]
    end
    
    @testset "SeriesEntry Edge Cases" begin
        # Test empty series
        empty_data = Float64[]
        empty_indices = DataFrame(Country=String[], Value=Float64[])
        empty_series = SeriesEntry(empty_data, empty_indices)
        
        @test length(empty_series.data) == 0
        @test nrow(empty_series.col_indices) == 0
        
        # Test single element series
        single_data = [42.0]
        single_indices = DataFrame(Country=["USA"], Metric=["GDP"])
        single_series = SeriesEntry(single_data, single_indices)
        
        @test length(single_series.data) == 1
        @test single_series.data[1] == 42.0
        @test single_series.col_indices.Country[1] == "USA"
        
        # Test series with missing values
        data_with_missing = [1.0, 2.0, 3.0]
        indices_with_missing = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Population = [330, missing, 83]  # Missing value for CHN
        )
        series_missing = SeriesEntry(data_with_missing, indices_with_missing)
        
        @test length(series_missing.data) == 3
        @test ismissing(series_missing.col_indices.Population[2])
    end
    
    @testset "SeriesEntry Data Types" begin
        # Test with different numeric types
        int_data = [1.0, 2.0, 3.0]  # Will be Float64
        indices = DataFrame(ID=[1, 2, 3], Name=["A", "B", "C"])
        series = SeriesEntry(int_data, indices)
        
        @test series.data isa Vector{Float64}
        @test all(x -> x isa Float64 for x in series.data)
        
        # Test indices with mixed types
        mixed_indices = DataFrame(
            IntCol = [1, 2, 3],
            StringCol = ["X", "Y", "Z"],
            BoolCol = [true, false, true],
            FloatCol = [1.1, 2.2, 3.3]
        )
        mixed_series = SeriesEntry([10.0, 20.0, 30.0], mixed_indices)
        
        @test mixed_series.col_indices.IntCol isa Vector{Int}
        @test mixed_series.col_indices.StringCol isa Vector{String}
        @test mixed_series.col_indices.BoolCol isa Vector{Bool}
        @test mixed_series.col_indices.FloatCol isa Vector{Float64}
    end
    
    @testset "SeriesEntry Large Data" begin
        # Test with larger dataset
        n = 1000
        large_data = rand(n)
        large_indices = DataFrame(
            ID = 1:n,
            Group = repeat(["A", "B", "C", "D"], n÷4),
            Value = rand(n)
        )
        
        large_series = SeriesEntry(large_data, large_indices)
        
        @test length(large_series.data) == n
        @test nrow(large_series.col_indices) == n
        @test length(unique(large_series.col_indices.Group)) == 4
        
        # Test that we can access all elements
        @test large_series.data[1] isa Float64
        @test large_series.data[end] isa Float64
        @test large_series.col_indices.ID[1] == 1
        @test large_series.col_indices.ID[end] == n
    end
    
    @testset "SeriesEntry Memory Efficiency" begin
        # Test that SeriesEntry doesn't copy data unnecessarily
        original_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        indices = DataFrame(ID=1:5, Name=["A", "B", "C", "D", "E"])
        
        series = SeriesEntry(original_data, indices)
        
        # Modify original data and check if series data is affected
        # (This tests if data is shared or copied)
        original_data[1] = 999.0
        @test series.data[1] == 999.0  # Should be affected if sharing memory
        
        # Test with fresh copy to avoid shared reference issues
        fresh_data = copy([1.0, 2.0, 3.0])
        fresh_indices = DataFrame(X=[1, 2, 3])
        fresh_series = SeriesEntry(fresh_data, fresh_indices)
        
        @test fresh_series.data == [1.0, 2.0, 3.0]
    end
end