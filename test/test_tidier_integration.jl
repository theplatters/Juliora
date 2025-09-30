function test_tidier_integration()
    @testset "to_long_dataframe Function" begin
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Region = ["NA", "Asia", "EU"]
        )
        col_df = DataFrame(
            Sector = ["Goods", "Services"],
            Type = ["Tradable", "NonTradable"]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test basic conversion
        long_df = to_long_dataframe(matrix_entry)
        
        @test size(long_df) == (6, 5)  # 3×2 = 6 rows, 2+2+1 = 5 columns
        @test names(long_df) == ["row_Country", "row_Region", "col_Sector", "col_Type", "value"]
        
        # Check data values are correct
        expected_values = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]  # Column-major order
        @test long_df.value == expected_values
        
        # Check row expansion is correct
        @test long_df.row_Country == ["USA", "CHN", "DEU", "USA", "CHN", "DEU"]
        @test long_df.row_Region == ["NA", "Asia", "EU", "NA", "Asia", "EU"]
        
        # Check column expansion is correct
        @test long_df.col_Sector == ["Goods", "Goods", "Goods", "Services", "Services", "Services"]
        @test long_df.col_Type == ["Tradable", "Tradable", "Tradable", "NonTradable", "NonTradable", "NonTradable"]
        
        # Test custom value name
        custom_long = to_long_dataframe(matrix_entry; value_name="trade_flow")
        @test "trade_flow" in names(custom_long)
        @test !("value" in names(custom_long))
        @test custom_long.trade_flow == expected_values
    end
    
    @testset "from_long_dataframe Function" begin
        # Create a long dataframe manually
        long_df = DataFrame(
            row_Country = ["USA", "CHN", "USA", "CHN"],
            row_Sector = ["Agr", "Agr", "Man", "Man"],
            col_Country = ["USA", "USA", "CHN", "CHN"],
            col_Sector = ["Agr", "Agr", "Man", "Man"],
            value = [1.0, 2.0, 3.0, 4.0]
        )
        
        # Convert back to MatrixEntry
        reconstructed = from_long_dataframe(long_df)
        
        @test reconstructed isa MatrixEntry
        @test size(reconstructed.data) == (2, 2)
        @test reconstructed.data == [1.0 3.0; 2.0 4.0]
        
        # Check indices were reconstructed correctly
        @test reconstructed.row_indices.Country == ["USA", "CHN"]
        @test reconstructed.row_indices.Sector == ["Agr", "Agr"]  # Note: might have duplicates
        @test reconstructed.col_indices.Country == ["USA", "CHN"]
        @test reconstructed.col_indices.Sector == ["Agr", "Man"]
        
        # Test with custom column names
        custom_long = DataFrame(
            origin_Country = ["USA", "CHN"],
            origin_Type = ["Dev", "Dev"],
            dest_Country = ["USA", "CHN"],
            dest_Type = ["Dev", "Dev"],
            flow = [10.0, 20.0]
        )
        
        custom_reconstructed = from_long_dataframe(
            custom_long;
            value_col="flow",
            row_prefix="origin_",
            col_prefix="dest_"
        )
        
        @test size(custom_reconstructed.data) == (2, 2)
        @test custom_reconstructed.row_indices.Country == ["USA", "CHN"]
        @test custom_reconstructed.col_indices.Country == ["USA", "CHN"]
    end
    
    @testset "Round-trip Conversion" begin
        # Test that converting to long and back preserves data
        original_data = [1.0 2.0 3.0; 4.0 5.0 6.0]
        row_df = DataFrame(
            Country = ["USA", "CHN"],
            GDP = [20000, 14000]
        )
        col_df = DataFrame(
            Sector = ["Agr", "Man", "Ser"],
            Share = [0.1, 0.3, 0.6]
        )
        
        original = MatrixEntry(original_data, col_df, row_df)
        
        # Round trip
        long_form = to_long_dataframe(original)
        reconstructed = from_long_dataframe(long_form)
        
        @test size(reconstructed.data) == size(original.data)
        @test reconstructed.data ≈ original.data
        @test reconstructed.row_indices.Country == original.row_indices.Country
        @test reconstructed.col_indices.Sector == original.col_indices.Sector
        
        # Test that lookups work
        @test reconstructed[(Country="USA", GDP=20000), (Sector="Man", Share=0.3)] ≈ 2.0
        @test reconstructed[(Country="CHN", GDP=14000), (Sector="Ser", Share=0.6)] ≈ 6.0
    end
    
    @testset "Pipe Operator" begin
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        row_df = DataFrame(Country = ["USA", "CHN", "DEU"])
        col_df = DataFrame(Sector = ["Goods", "Services"])
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test piping to a simple function
        result = matrix_entry |> df -> nrow(df)
        @test result == 6  # 3×2 = 6 rows in long form
        
        # Test piping to TidierData-style operations
        using Tidier
        
        # Test filtering through pipe
        filtered_result = matrix_entry |> df -> @chain df begin
            @filter(value > 2.0)
            @select(row_Country, value)
        end
        
        @test nrow(filtered_result) == 4  # Values 3,4,5,6 are > 2
        @test names(filtered_result) == ["row_Country", "value"]
        @test all(filtered_result.value .> 2.0)
        
        # Test aggregation through pipe
        country_sums = matrix_entry |> df -> @chain df begin
            @group_by(row_Country)
            @summarize(total = sum(value))
            @arrange(desc(total))
        end
        
        @test nrow(country_sums) == 3
        @test "total" in names(country_sums)
        @test country_sums.total == [11.0, 7.0, 3.0]  # DEU: 5+6, CHN: 3+4, USA: 1+2
    end
    
    @testset "Complex TidierData Operations" begin
        # Create larger test dataset
        data = rand(4, 3) * 100  # Scale up for meaningful tests
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU", "JPN"],
            Region = ["NA", "Asia", "EU", "Asia"],
            Developed = [true, false, true, true]
        )
        col_df = DataFrame(
            Sector = ["Agr", "Man", "Ser"],
            Primary = [true, false, false]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test complex aggregation
        using Tidier
        
        regional_analysis = matrix_entry |> df -> @chain df begin
            @group_by(row_Region, col_Sector)
            @summarize(
                avg_flow = mean(value),
                max_flow = maximum(value),
                count = length(value)
            )
            @arrange(row_Region, desc(avg_flow))
        end
        
        @test nrow(regional_analysis) == 6  # 2 regions × 3 sectors
        @test names(regional_analysis) == ["row_Region", "col_Sector", "avg_flow", "max_flow", "count"]
        
        # Test pivot-like operations
        country_sector_matrix = matrix_entry |> df -> @chain df begin
            @group_by(row_Country, col_Sector)
            @summarize(total_flow = sum(value))
            @pivot_wider(names_from = col_Sector, values_from = total_flow)
        end
        
        @test nrow(country_sector_matrix) == 4  # 4 countries
        @test "Agr" in names(country_sector_matrix)
        @test "Man" in names(country_sector_matrix)
        @test "Ser" in names(country_sector_matrix)
    end
    
    @testset "Integration with Boolean Indexing" begin
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Developed = [true, false, true]
        )
        col_df = DataFrame(
            Sector = ["Agr", "Man", "Ser"],
            Primary = [true, false, false]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Combine boolean indexing with TidierData operations
        using Tidier
        
        # First filter with boolean indexing, then use TidierData
        developed_only = matrix_entry[matrix_entry.row_indices.Developed, :]
        
        developed_analysis = developed_only |> df -> @chain df begin
            @group_by(col_Sector)
            @summarize(avg_value = mean(value))
            @arrange(desc(avg_value))
        end
        
        @test nrow(developed_analysis) == 3
        @test developed_analysis.col_Sector == ["Ser", "Man", "Agr"]  # Descending order
        
        # Test the reverse: TidierData then boolean indexing
        high_value_data = matrix_entry |> df -> @chain df begin
            @filter(value > 5.0)
        end
        
        # This returns a DataFrame, so we need to convert back for boolean indexing
        reconstructed = from_long_dataframe(high_value_data)
        final_filter = reconstructed[:, reconstructed.col_indices.Primary]
        
        @test size(final_filter.data)[2] == 1  # Only primary sector (Agr)
    end
end