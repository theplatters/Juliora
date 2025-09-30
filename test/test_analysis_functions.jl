function test_analysis_functions()
    @testset "sum_by_country Function" begin
        # Create test data with known structure
        data = [10.0 20.0 30.0; 40.0 50.0 60.0; 70.0 80.0 90.0; 100.0 110.0 120.0]
        row_df = DataFrame(
            CountryCode = ["USA", "USA", "CHN", "CHN"],
            Industry = ["Agr", "Man", "Agr", "Man"],
            Sector = ["Primary", "Secondary", "Primary", "Secondary"]
        )
        col_df = DataFrame(
            CountryCode = ["USA", "CHN", "DEU"],
            Industry = ["Agr", "Man", "Ser"],
            Sector = ["Primary", "Secondary", "Tertiary"]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test sum by rows (exports by country)
        row_sums = sum_by_country(matrix_entry; dimension=:rows)
        @test nrow(row_sums) == 2  # USA, CHN
        @test "row_CountryCode" in names(row_sums)
        @test "value" in names(row_sums)
        
        # Check actual values - USA rows sum to 50+70=120, CHN rows sum to 130+150=280
        usa_total = row_sums[row_sums.row_CountryCode .== "USA", :value][1]
        chn_total = row_sums[row_sums.row_CountryCode .== "CHN", :value][1]
        @test usa_total == 60.0  # 10+20+30
        @test chn_total == 390.0  # 40+50+60+70+80+90+100+110+120
        
        # Test sum by columns (imports by country)
        col_sums = sum_by_country(matrix_entry; dimension=:cols)
        @test nrow(col_sums) == 3  # USA, CHN, DEU
        @test "col_CountryCode" in names(col_sums)
        
        # Test bilateral country flows
        bilateral = sum_by_country(matrix_entry; dimension=:both)
        @test nrow(bilateral) == 6  # 2 row countries × 3 col countries
        @test "row_CountryCode" in names(bilateral)
        @test "col_CountryCode" in names(bilateral)
        @test "value" in names(bilateral)
    end
    
    @testset "sum_by_sector Function" begin
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
        row_df = DataFrame(
            CountryCode = ["USA", "USA", "CHN", "CHN"],
            Sector = ["Agr", "Man", "Agr", "Man"]
        )
        col_df = DataFrame(
            CountryCode = ["USA", "CHN"],
            Sector = ["Agr", "Man"]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test sum by row sectors
        row_sector_sums = sum_by_sector(matrix_entry; dimension=:rows)
        @test nrow(row_sector_sums) == 2  # Agr, Man
        @test "row_Sector" in names(row_sector_sums)
        
        # Check values: Agr rows (1,2 + 5,6) = 14, Man rows (3,4 + 7,8) = 22
        agr_total = row_sector_sums[row_sector_sums.row_Sector .== "Agr", :value][1]
        man_total = row_sector_sums[row_sector_sums.row_Sector .== "Man", :value][1]
        @test agr_total == 14.0
        @test man_total == 22.0
        
        # Test sum by column sectors
        col_sector_sums = sum_by_sector(matrix_entry; dimension=:cols)
        @test nrow(col_sector_sums) == 2  # Agr, Man
        @test "col_Sector" in names(col_sector_sums)
        
        # Test bilateral sector flows
        sector_bilateral = sum_by_sector(matrix_entry; dimension=:both)
        @test nrow(sector_bilateral) == 4  # 2 row sectors × 2 col sectors
        @test "row_Sector" in names(sector_bilateral)
        @test "col_Sector" in names(sector_bilateral)
    end
    
    @testset "groupby_matrix Function" begin
        data = rand(6, 4) * 100
        row_df = DataFrame(
            CountryCode = repeat(["USA", "CHN", "DEU"], 2),
            Sector = repeat(["Agr", "Man"], 3),
            Developed = repeat([true, false, true], 2)
        )
        col_df = DataFrame(
            CountryCode = repeat(["USA", "CHN"], 2),
            Sector = repeat(["Goods", "Services"], 2)
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test grouping by single column (rows)
        country_groups = groupby_matrix(matrix_entry, :CountryCode; rows=true)
        @test nrow(country_groups) == 3  # USA, CHN, DEU
        @test "row_CountryCode" in names(country_groups)
        @test "value" in names(country_groups)
        
        # Test grouping by multiple columns (rows)
        multi_groups = groupby_matrix(matrix_entry, :CountryCode, :Sector; rows=true)
        @test nrow(multi_groups) == 6  # 3 countries × 2 sectors
        @test "row_CountryCode" in names(multi_groups)
        @test "row_Sector" in names(multi_groups)
        
        # Test grouping columns
        col_groups = groupby_matrix(matrix_entry, :Sector; rows=false)
        @test nrow(col_groups) == 2  # Goods, Services
        @test "col_Sector" in names(col_groups)
        
        # Test different aggregation functions
        max_groups = groupby_matrix(matrix_entry, :CountryCode; agg_func=maximum, rows=true)
        @test nrow(max_groups) == 3
        @test all(max_groups.value .>= 0)  # All should be positive
        
        mean_groups = groupby_matrix(matrix_entry, :CountryCode; agg_func=mean, rows=true)
        @test nrow(mean_groups) == 3
        @test all(mean_groups.value .>= 0)
        
        # Test custom value name
        custom_groups = groupby_matrix(matrix_entry, :CountryCode; value_name="custom_value")
        @test "custom_value" in names(custom_groups)
        @test !("value" in names(custom_groups))
    end
    
    @testset "matrix_summary Function" begin
        # Create test data with known properties
        data = [1.0 0.0 3.0; 0.0 5.0 6.0; 7.0 8.0 0.0]  # Has zeros and known values
        row_df = DataFrame(Country = ["A", "B", "C"])
        col_df = DataFrame(Sector = ["X", "Y", "Z"])
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        summary_stats = matrix_summary(matrix_entry)
        
        @test nrow(summary_stats) == 1
        @test "total" in names(summary_stats)
        @test "mean" in names(summary_stats)
        @test "median" in names(summary_stats)
        @test "std" in names(summary_stats)
        @test "min_val" in names(summary_stats)
        @test "max_val" in names(summary_stats)
        @test "n_nonzero" in names(summary_stats)
        @test "n_total" in names(summary_stats)
        
        # Check calculated values
        @test summary_stats.total[1] == 30.0  # Sum of all elements
        @test summary_stats.mean[1] ≈ 30.0/9  # Mean
        @test summary_stats.min_val[1] == 0.0
        @test summary_stats.max_val[1] == 8.0
        @test summary_stats.n_nonzero[1] == 6  # Count of non-zero elements
        @test summary_stats.n_total[1] == 9    # Total elements
        
        # Test with all zeros
        zero_data = zeros(2, 2)
        zero_matrix = MatrixEntry(zero_data, DataFrame(A=[1,2]), DataFrame(B=[1,2]))
        zero_summary = matrix_summary(zero_matrix)
        
        @test zero_summary.total[1] == 0.0
        @test zero_summary.n_nonzero[1] == 0
        @test zero_summary.n_total[1] == 4
    end
    
    @testset "country_summary Function" begin
        # Create bilateral trade data
        data = [10.0 5.0 8.0; 12.0 15.0 20.0; 3.0 7.0 25.0]
        row_df = DataFrame(CountryCode = ["USA", "CHN", "DEU"])
        col_df = DataFrame(CountryCode = ["USA", "CHN", "DEU"])
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        country_flows = country_summary(matrix_entry)
        
        @test nrow(country_flows) == 9  # 3×3 country pairs
        @test "row_CountryCode" in names(country_flows)
        @test "col_CountryCode" in names(country_flows)
        @test "total_flow" in names(country_flows)
        @test "mean_flow" in names(country_flows)
        @test "n_sectors" in names(country_flows)
        
        # Check that it's sorted by total_flow descending
        @test country_flows.total_flow[1] >= country_flows.total_flow[2]
        @test country_flows.total_flow[2] >= country_flows.total_flow[3]
        
        # Test largest flow (should be DEU->DEU = 25.0)
        largest_flow = country_flows[1, :]
        @test largest_flow.total_flow == 25.0
        @test largest_flow.row_CountryCode == "DEU"
        @test largest_flow.col_CountryCode == "DEU"
        @test largest_flow.n_sectors == 1  # Only one entry per country pair in this simple case
        
        # Test with multi-sector data
        multi_data = rand(6, 6) * 100  # 6 sectors (2 per country)
        multi_row_df = DataFrame(
            CountryCode = repeat(["USA", "CHN", "DEU"], 2),
            Sector = repeat(["Agr", "Man"], 3)
        )
        multi_col_df = DataFrame(
            CountryCode = repeat(["USA", "CHN", "DEU"], 2),
            Sector = repeat(["Agr", "Man"], 3)
        )
        
        multi_matrix = MatrixEntry(multi_data, multi_col_df, multi_row_df)
        multi_summary = country_summary(multi_matrix)
        
        @test nrow(multi_summary) == 9  # Still 3×3 countries
        # Each country pair should have 4 sectors (2×2)
        @test all(multi_summary.n_sectors .== 4)
    end
    
    @testset "pivot_matrix_to_wide Function" begin
        # Create test data
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Region = ["NA", "Asia", "EU"]
        )
        col_df = DataFrame(
            Sector = ["Agr", "Man"],
            Type = ["Primary", "Secondary"]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test pivot by country and sector
        wide_df = pivot_matrix_to_wide(matrix_entry, [:Country], :Sector)
        
        @test nrow(wide_df) == 3  # 3 countries
        @test "row_Country" in names(wide_df)
        @test "Agr" in names(wide_df)
        @test "Man" in names(wide_df)
        
        # Check that pivoting worked correctly
        usa_row = wide_df[wide_df.row_Country .== "USA", :]
        @test usa_row.Agr[1] == 1.0  # USA-Agr value
        @test usa_row.Man[1] == 2.0  # USA-Man value
        
        # Test pivot with multiple row variables
        multi_wide = pivot_matrix_to_wide(matrix_entry, [:Country, :Region], :Sector)
        @test "row_Country" in names(multi_wide)
        @test "row_Region" in names(multi_wide)
        @test "Agr" in names(multi_wide)
        @test "Man" in names(multi_wide)
        
        # Test with custom value name
        custom_wide = pivot_matrix_to_wide(matrix_entry, [:Country], :Sector, "trade_value")
        @test "Agr" in names(custom_wide)
        @test "Man" in names(custom_wide)
    end
    
    @testset "add_calculated_column Function" begin
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        row_df = DataFrame(
            CountryCode = ["USA", "CHN", "DEU"],
            GDP = [20000, 14000, 4000]
        )
        col_df = DataFrame(
            Sector = ["Agr", "Man"],
            Share = [0.1, 0.3]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test adding calculated column to rows
        with_region = add_calculated_column(
            matrix_entry, 
            :Region,
            row -> row.CountryCode in ["USA"] ? "NA" : (row.CountryCode in ["CHN"] ? "Asia" : "EU")
        )
        
        @test "Region" in names(with_region.row_indices)
        @test with_region.row_indices.Region == ["NA", "Asia", "EU"]
        @test size(with_region.data) == size(matrix_entry.data)  # Data unchanged
        
        # Test adding calculated column to columns
        with_importance = add_calculated_column(
            matrix_entry,
            :Important,
            col -> col.Share > 0.2,
            to_rows=false
        )
        
        @test "Important" in names(with_importance.col_indices)
        @test with_importance.col_indices.Important == [false, true]  # Agr: 0.1 < 0.2, Man: 0.3 > 0.2
        @test size(with_importance.data) == size(matrix_entry.data)
        
        # Test that lookups still work with new columns
        usa_key = (CountryCode="USA", GDP=20000, Region="NA")
        agr_key = (Sector="Agr", Share=0.1)
        @test with_region[usa_key, agr_key] == 1.0
        
        # Test complex calculation
        with_gdp_class = add_calculated_column(
            matrix_entry,
            :GDPClass,
            row -> row.GDP > 15000 ? "High" : (row.GDP > 5000 ? "Medium" : "Low")
        )
        
        @test with_gdp_class.row_indices.GDPClass == ["High", "Medium", "Low"]
    end
end