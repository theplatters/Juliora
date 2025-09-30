function test_matrixentry()
    @testset "MatrixEntry Constructor" begin
        # Test basic construction
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Sector = ["Agriculture", "Manufacturing", "Services"]
        )
        col_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Sector = ["Agriculture", "Manufacturing", "Services"]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        @test size(matrix_entry.data) == (3, 3)
        @test matrix_entry.data == data
        @test nrow(matrix_entry.row_indices) == 3
        @test nrow(matrix_entry.col_indices) == 3
        @test length(matrix_entry.row_lookup) == 3
        @test length(matrix_entry.col_lookup) == 3
        
        # Test dimension mismatch error
        wrong_data = [1.0 2.0; 3.0 4.0]  # 2x2 but indices are 3x3
        @test_throws AssertionError MatrixEntry(wrong_data, col_df, row_df)
        
        # Test lookup dictionary functionality
        usa_agr_key = (Country="USA", Sector="Agriculture")
        @test haskey(matrix_entry.row_lookup, usa_agr_key)
        @test matrix_entry.row_lookup[usa_agr_key] == 1
        
        chn_man_key = (Country="CHN", Sector="Manufacturing")
        @test haskey(matrix_entry.col_lookup, chn_man_key)
        @test matrix_entry.col_lookup[chn_man_key] == 2
    end
    
    @testset "MatrixEntry Indexing with NamedTuples" begin
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Sector = ["Agr", "Man", "Ser"]
        )
        col_df = DataFrame(
            Country = ["USA", "CHN"],
            Sector = ["Agr", "Man"]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test valid indexing
        @test matrix_entry[(Country="USA", Sector="Agr"), (Country="USA", Sector="Agr")] == 1.0
        @test matrix_entry[(Country="CHN", Sector="Man"), (Country="CHN", Sector="Man")] == 4.0
        @test matrix_entry[(Country="DEU", Sector="Ser"), (Country="CHN", Sector="Man")] == 6.0
        
        # Test invalid row key
        @test_throws BoundsError matrix_entry[(Country="JPN", Sector="Agr"), (Country="USA", Sector="Agr")]
        
        # Test invalid column key
        @test_throws BoundsError matrix_entry[(Country="USA", Sector="Agr"), (Country="JPN", Sector="Agr")]
        
        # Test missing sector
        @test_throws BoundsError matrix_entry[(Country="USA", Sector="Tech"), (Country="USA", Sector="Agr")]
    end
    
    @testset "MatrixEntry Edge Cases" begin
        # Test with single row/column
        single_data = reshape([42.0], 1, 1)
        single_row = DataFrame(Country=["USA"], Sector=["Total"])
        single_col = DataFrame(Country=["USA"], Sector=["Total"])
        
        single_matrix = MatrixEntry(single_data, single_col, single_row)
        @test single_matrix[(Country="USA", Sector="Total"), (Country="USA", Sector="Total")] == 42.0
        
        # Test with different column types
        mixed_data = [1.0 2.0; 3.0 4.0]
        mixed_row = DataFrame(
            ID = [1, 2],
            Name = ["A", "B"],
            Active = [true, false]
        )
        mixed_col = DataFrame(
            ID = [10, 20],
            Type = ["X", "Y"]
        )
        
        mixed_matrix = MatrixEntry(mixed_data, mixed_col, mixed_row)
        @test mixed_matrix[(ID=1, Name="A", Active=true), (ID=10, Type="X")] == 1.0
        @test mixed_matrix[(ID=2, Name="B", Active=false), (ID=20, Type="Y")] == 4.0
    end
    
    @testset "MatrixEntry Type Stability" begin
        data = [1.0 2.0; 3.0 4.0]
        row_df = DataFrame(A=[1, 2], B=["X", "Y"])
        col_df = DataFrame(C=[10, 20], D=["P", "Q"])
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test that returned values are Float64
        val = matrix_entry[(A=1, B="X"), (C=10, D="P")]
        @test val isa Float64
        @test val == 1.0
        
        # Test that lookup dictionaries have correct types
        @test matrix_entry.row_lookup isa Dict{NamedTuple, Int}
        @test matrix_entry.col_lookup isa Dict{NamedTuple, Int}
    end
end