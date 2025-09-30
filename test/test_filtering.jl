function test_filtering()
    @testset "Filter Rows Function" begin
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU", "JPN"],
            Region = ["NA", "Asia", "EU", "Asia"],
            GDP = [20000, 14000, 4000, 5000],
            Developed = [true, false, true, true]
        )
        col_df = DataFrame(
            Sector = ["Agr", "Man", "Ser"],
            Share = [0.1, 0.3, 0.6]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test simple country filtering
        usa_filtered = filter_rows(matrix_entry, row -> row.Country == "USA")
        @test size(usa_filtered.data) == (1, 3)
        @test usa_filtered.data == [1.0 2.0 3.0]
        @test usa_filtered.row_indices.Country == ["USA"]
        
        # Test region filtering
        asia_filtered = filter_rows(matrix_entry, row -> row.Region == "Asia")
        @test size(asia_filtered.data) == (2, 3)
        @test asia_filtered.data == [4.0 5.0 6.0; 10.0 11.0 12.0]
        @test asia_filtered.row_indices.Country == ["CHN", "JPN"]
        
        # Test multiple condition filtering
        rich_developed = filter_rows(matrix_entry, row -> row.Developed && row.GDP > 10000)
        @test size(rich_developed.data) == (1, 3)  # Only USA
        @test rich_developed.row_indices.Country == ["USA"]
        
        # Test filtering with list membership
        selected_countries = filter_rows(matrix_entry, row -> row.Country in ["USA", "DEU"])
        @test size(selected_countries.data) == (2, 3)
        @test selected_countries.row_indices.Country == ["USA", "DEU"]
        
        # Test filtering that returns empty
        no_match = filter_rows(matrix_entry, row -> row.Country == "NONEXISTENT")
        @test size(no_match.data) == (0, 3)
        @test nrow(no_match.row_indices) == 0
        
        # Test filtering all rows
        all_rows = filter_rows(matrix_entry, row -> true)
        @test size(all_rows.data) == (4, 3)
        @test all_rows.data == data
    end
    
    @testset "Filter Columns Function" begin
        data = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0]
        row_df = DataFrame(Country = ["USA", "CHN", "DEU"])
        col_df = DataFrame(
            Sector = ["Agr", "Man", "Ser", "Min"],
            Primary = [true, false, true, true],
            Share = [0.05, 0.30, 0.60, 0.05],
            Tradable = [true, true, false, true]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test simple sector filtering
        agr_filtered = filter_cols(matrix_entry, col -> col.Sector == "Agr")
        @test size(agr_filtered.data) == (3, 1)
        @test agr_filtered.data == [1.0; 5.0; 9.0]
        @test agr_filtered.col_indices.Sector == ["Agr"]
        
        # Test primary sectors filtering
        primary_filtered = filter_cols(matrix_entry, col -> col.Primary)
        @test size(primary_filtered.data) == (3, 3)  # Agr, Ser, Min
        @test primary_filtered.col_indices.Sector == ["Agr", "Ser", "Min"]
        
        # Test share-based filtering
        major_sectors = filter_cols(matrix_entry, col -> col.Share > 0.1)
        @test size(major_sectors.data) == (3, 2)  # Man, Ser
        @test major_sectors.col_indices.Sector == ["Man", "Ser"]
        
        # Test complex filtering
        tradable_primary = filter_cols(matrix_entry, col -> col.Tradable && col.Primary)
        @test size(tradable_primary.data) == (3, 2)  # Agr, Min
        @test tradable_primary.col_indices.Sector == ["Agr", "Min"]
        
        # Test no match
        no_match = filter_cols(matrix_entry, col -> col.Share > 1.0)
        @test size(no_match.data) == (3, 0)
        @test nrow(no_match.col_indices) == 0
    end
    
    @testset "Filter Matrix Function" begin
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU", "JPN"],
            Developed = [true, false, true, true]
        )
        col_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            EU = [false, false, true]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test filtering both dimensions
        developed_to_eu = filter_matrix(
            matrix_entry,
            row -> row.Developed,
            col -> col.EU
        )
        @test size(developed_to_eu.data) == (3, 1)  # USA, DEU, JPN → DEU
        @test developed_to_eu.data == [3.0; 6.0; 9.0]
        @test developed_to_eu.row_indices.Country == ["USA", "DEU", "JPN"]
        @test developed_to_eu.col_indices.Country == ["DEU"]
        
        # Test specific country-to-country flows
        usa_to_china = filter_matrix(
            matrix_entry,
            row -> row.Country == "USA",
            col -> col.Country == "CHN"
        )
        @test size(usa_to_china.data) == (1, 1)
        @test usa_to_china.data[1, 1] == 2.0
        
        # Test same condition for both dimensions
        same_countries = filter_matrix(
            matrix_entry,
            row -> row.Country in ["USA", "DEU"],
            col -> col.Country in ["USA", "DEU"]
        )
        @test size(same_countries.data) == (2, 2)
        @test same_countries.row_indices.Country == ["USA", "DEU"]
        @test same_countries.col_indices.Country == ["USA", "DEU"]
        
        # Test empty result
        impossible = filter_matrix(
            matrix_entry,
            row -> row.Country == "NONE",
            col -> col.Country == "NONE"
        )
        @test size(impossible.data) == (0, 0)
    end
    
    @testset "Filtering Edge Cases" begin
        # Test with single row/column
        single_data = reshape([42.0], 1, 1)
        single_row = DataFrame(ID=[1], Name=["A"])
        single_col = DataFrame(ID=[1], Type=["X"])
        
        single_matrix = MatrixEntry(single_data, single_col, single_row)
        
        # Filter that matches
        match_filter = filter_rows(single_matrix, row -> row.ID == 1)
        @test size(match_filter.data) == (1, 1)
        @test match_filter.data[1, 1] == 42.0
        
        # Filter that doesn't match
        no_match_filter = filter_rows(single_matrix, row -> row.ID == 999)
        @test size(no_match_filter.data) == (0, 1)
        
        # Test with missing values in indices
        data_missing = [1.0 2.0; 3.0 4.0]
        row_missing = DataFrame(
            Country = ["USA", "CHN"],
            Population = [330, missing]
        )
        col_df = DataFrame(Sector = ["A", "B"])
        
        matrix_missing = MatrixEntry(data_missing, col_df, row_missing)
        
        # Filter handling missing values
        non_missing_pop = filter_rows(matrix_missing, row -> !ismissing(row.Population))
        @test size(non_missing_pop.data) == (1, 2)
        @test non_missing_pop.row_indices.Country == ["USA"]
        
        # Filter for missing values
        missing_pop = filter_rows(matrix_missing, row -> ismissing(row.Population))
        @test size(missing_pop.data) == (1, 2)
        @test missing_pop.row_indices.Country == ["CHN"]
    end
    
    @testset "Filtering Preserves Structure" begin
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Code = ["US", "CN", "DE"]
        )
        col_df = DataFrame(
            Sector = ["Goods", "Services"],
            ID = [1, 2]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test that filtered result maintains lookup functionality
        filtered = filter_rows(matrix_entry, row -> row.Country != "CHN")
        
        @test filtered isa MatrixEntry
        @test haskey(filtered.row_lookup, (Country="USA", Code="US"))
        @test haskey(filtered.row_lookup, (Country="DEU", Code="DE"))
        @test !haskey(filtered.row_lookup, (Country="CHN", Code="CN"))
        
        # Test that indexing works on filtered result
        @test filtered[(Country="USA", Code="US"), (Sector="Goods", ID=1)] == 1.0
        @test filtered[(Country="DEU", Code="DE"), (Sector="Services", ID=2)] == 6.0
        
        # Test error for removed keys
        @test_throws BoundsError filtered[(Country="CHN", Code="CN"), (Sector="Goods", ID=1)]
    end
end