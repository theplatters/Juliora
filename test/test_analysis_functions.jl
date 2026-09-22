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

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)

    # Test sum by rows (exports by country)
    row_sums = sum_by_country(matrix_entry; dimension = :rows)
    @test nrow(row_sums) == 2  # USA, CHN
    @test "row_CountryCode" in names(row_sums)
    @test "value" in names(row_sums)

    # Check actual values
    usa_total = row_sums[row_sums.row_CountryCode .== "USA", :value][1]
    chn_total = row_sums[row_sums.row_CountryCode .== "CHN", :value][1]
    @test usa_total == 210.0  # (10+20+30) + (40+50+60)
    @test chn_total == 570.0  # (70+80+90) + (100+110+120)

    # Test sum by columns (imports by country)
    col_sums = sum_by_country(matrix_entry; dimension = :cols)
    @test nrow(col_sums) == 3  # USA, CHN, DEU
    @test "col_CountryCode" in names(col_sums)

    # Test bilateral country flows
    bilateral = sum_by_country(matrix_entry; dimension = :both)
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

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)

    # Test sum by row sectors
    row_sector_sums = sum_by_sector(matrix_entry; dimension = :rows)
    @test nrow(row_sector_sums) == 2  # Agr, Man
    @test "row_Sector" in names(row_sector_sums)

    # Check values: Agr rows (1,2 + 5,6) = 14, Man rows (3,4 + 7,8) = 22
    agr_total = row_sector_sums[row_sector_sums.row_Sector .== "Agr", :value][1]
    man_total = row_sector_sums[row_sector_sums.row_Sector .== "Man", :value][1]
    @test agr_total == 14.0
    @test man_total == 22.0

    # Test sum by column sectors
    col_sector_sums = sum_by_sector(matrix_entry; dimension = :cols)
    @test nrow(col_sector_sums) == 2  # Agr, Man
    @test "col_Sector" in names(col_sector_sums)

    # Test bilateral sector flows
    sector_bilateral = sum_by_sector(matrix_entry; dimension = :both)
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

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)

    # Test grouping by single column (rows)
    country_groups = groupby_matrix(matrix_entry, :CountryCode; rows = true)
    @test nrow(country_groups) == 3  # USA, CHN, DEU
    @test "row_CountryCode" in names(country_groups)
    @test "value" in names(country_groups)

    # Test grouping by multiple columns (rows)
    multi_groups = groupby_matrix(matrix_entry, :CountryCode, :Sector; rows = true)
    @test nrow(multi_groups) == 6  # 3 countries × 2 sectors
    @test "row_CountryCode" in names(multi_groups)
    @test "row_Sector" in names(multi_groups)

    # Test grouping columns
    col_groups = groupby_matrix(matrix_entry, :Sector; rows = false)
    @test nrow(col_groups) == 2  # Goods, Services
    @test "col_Sector" in names(col_groups)

    # Test different aggregation functions
    max_groups = groupby_matrix(matrix_entry, :CountryCode; agg_func = maximum, rows = true)
    @test nrow(max_groups) == 3
    @test all(max_groups.value .>= 0)  # All should be positive

    mean_groups = groupby_matrix(matrix_entry, :CountryCode; agg_func = mean, rows = true)
    @test nrow(mean_groups) == 3
    @test all(mean_groups.value .>= 0)

    # Test custom value name
    custom_groups = groupby_matrix(matrix_entry, :CountryCode; value_name = "custom_value")
    @test "custom_value" in names(custom_groups)
    @test !("value" in names(custom_groups))
end

@testset "matrix_summary Function" begin
    # Create test data with known properties
    data = [1.0 0.0 3.0; 0.0 5.0 6.0; 7.0 8.0 0.0]  # Has zeros and known values
    row_df = DataFrame(Country = ["A", "B", "C"])
    col_df = DataFrame(Sector = ["X", "Y", "Z"])

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)
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
    @test summary_stats.mean[1] ≈ 30.0 / 9  # Mean
    @test summary_stats.min_val[1] == 0.0
    @test summary_stats.max_val[1] == 8.0
    @test summary_stats.n_nonzero[1] == 6  # Count of non-zero elements
    @test summary_stats.n_total[1] == 9    # Total elements

    # Test with all zeros
    zero_data = zeros(2, 2)
    zero_matrix = IO.MatrixEntry(zero_data, DataFrame(A = [1, 2]), DataFrame(B = [1, 2]))
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

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)
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

    multi_matrix = IO.MatrixEntry(multi_data, multi_col_df, multi_row_df)
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

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)

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

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)

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
        to_rows = false
    )

    @test "Important" in names(with_importance.col_indices)
    @test with_importance.col_indices.Important == [false, true]  # Agr: 0.1 < 0.2, Man: 0.3 > 0.2
    @test size(with_importance.data) == size(matrix_entry.data)

    # Test that lookups still work with new columns
    usa_key = (CountryCode = "USA", GDP = 20000, Region = "NA")
    agr_key = (Sector = "Agr", Share = 0.1)
    @test with_region[usa_key, agr_key] == 1.0

    # Test complex calculation
    with_gdp_class = add_calculated_column(
        matrix_entry,
        :GDPClass,
        row -> row.GDP > 15000 ? "High" : (row.GDP > 5000 ? "Medium" : "Low")
    )

    @test with_gdp_class.row_indices.GDPClass == ["High", "Medium", "Low"]
end

@testset "Convenience Query Functions" begin
    # Create mock data
    f_data = [1.0 2.0; 3.0 4.0; 5.0 6.0] # 3 stressors x 2 sectors
    sector_indices = DataFrame(
        CountryCode = ["USA", "CHN"],
        Industry = ["Agr", "Man"],
        Sector = ["Primary", "Secondary"]
    )
    stressor_indices = DataFrame(
        Stressor = ["CO2", "Water", "Land"],
        Source = ["Fossil", "Fresh", "Arable"]
    )
    x_output = [10.0, 20.0]

    # 1. Test DataFrame methods
    @test countries(sector_indices) == ["USA", "CHN"]
    @test sectors(sector_indices) == ["Primary", "Secondary"]
    @test stressors(stressor_indices) == ["CO2", "Water", "Land"]

    # Test fallback
    @test countries(DataFrame(A = [1])) == String[]
    @test sectors(DataFrame(A = [1])) == String[]
    @test stressors(DataFrame(A = [1])) == String[]

    # 2. Test MatrixEntry methods
    f_matrix = IO.MatrixEntry(f_data, sector_indices, stressor_indices)
    a_matrix = IO.MatrixEntry(f_data ./ x_output', sector_indices, stressor_indices)

    @test countries(f_matrix) == ["USA", "CHN"]
    @test sectors(f_matrix) == ["Primary", "Secondary"]
    @test stressors(f_matrix) == ["CO2", "Water", "Land"]

    # Test singular aliases
    @test country(f_matrix) == ["USA", "CHN"]
    @test sector(f_matrix) == ["Primary", "Secondary"]
    @test stressor(f_matrix) == ["CO2", "Water", "Land"]

    # 3. Test SeriesEntry methods
    s_entry = SeriesEntry(x_output, sector_indices)
    @test countries(s_entry) == ["USA", "CHN"]
    @test sectors(s_entry) == ["Primary", "Secondary"]
    @test country(s_entry) == ["USA", "CHN"]
    @test sector(s_entry) == ["Primary", "Secondary"]

    # 4. Test EnvironmentalExtension methods
    env_ext = EnvironmentalExtension(f_matrix, a_matrix)
    @test countries(env_ext) == ["USA", "CHN"]
    @test sectors(env_ext) == ["Primary", "Secondary"]
    @test stressors(env_ext) == ["CO2", "Water", "Land"]
    @test country(env_ext) == ["USA", "CHN"]
    @test sector(env_ext) == ["Primary", "Secondary"]
    @test stressor(env_ext) == ["CO2", "Water", "Land"]

    # 5. Test MRIO methods
    z_data = [10.0 2.0; 3.0 15.0]
    y_data = [5.0 1.0; 2.0 8.0]
    va_data = [2.0 3.0; 1.0 1.0]

    z_matrix = IO.MatrixEntry(z_data, sector_indices, sector_indices)
    y_matrix = IO.MatrixEntry(y_data, sector_indices, sector_indices)

    va_col_indices = DataFrame(Category = ["Compensation", "Taxes"])
    va_matrix = IO.MatrixEntry(va_data, sector_indices, va_col_indices)

    mrio = MRIO(Z = z_matrix, Y = y_matrix, VA = va_matrix)

    @test countries(mrio) == ["USA", "CHN"]
    @test sectors(mrio) == ["Primary", "Secondary"]

    # Attach env_ext to mrio and test stressors
    mrio_with_env = MRIO(
        mrio.A,
        mrio.T,
        mrio.VA,
        mrio.FD,
        mrio.L,
        mrio.X,
        env_ext
    )
    @test stressors(mrio_with_env) == ["CO2", "Water", "Land"]
    @test stressor(mrio_with_env) == ["CO2", "Water", "Land"]
end


function _wp2_square_mrio(; row_countries = ["USA", "USA", "CHN", "CHN"], col_countries = ["USA", "CHN"])
    sec_idx = DataFrame(CountryCode = row_countries, Sector = ["Agr", "Man", "Agr", "Man"])
    z = IO.MatrixEntry(
        [10.0 2.0 3.0 1.0; 1.0 15.0 2.0 4.0; 4.0 1.0 12.0 3.0; 2.0 3.0 1.0 18.0],
        sec_idx,
        sec_idx
    )
    y = IO.MatrixEntry(
        [5.0 1.0; 2.0 8.0; 1.0 3.0; 4.0 6.0],
        DataFrame(CountryCode = col_countries),
        sec_idx
    )
    va = IO.MatrixEntry(
        [2.0 3.0 1.0 4.0],
        sec_idx,
        DataFrame(Category = ["Compensation"])
    )
    return MRIO(Z = z, Y = y, VA = va)
end

@testset "induced_production Country Matching" begin
    m = _wp2_square_mrio()

    full = induced_production(m)
    @test nrow(full) == 4

    usa_only = induced_production(m; consumer_countries = ["USA"])
    @test nrow(usa_only) == 4
    @test usa_only.InducedProduction != full.InducedProduction

    # Nothing matches -> ArgumentError (both roles)
    @test_throws ArgumentError induced_production(m; consumer_countries = ["XX"])
    @test_throws ArgumentError induced_production(m; producer_countries = ["XX"])
    @test_throws ArgumentError induced_production(m, ["XX"], ["USA"])
    @test_throws ArgumentError induced_production(m, ["USA"], ["XX"])

    # Partial match -> warn, result equals the matched subset
    partial = @test_logs (:warn,) induced_production(m; consumer_countries = ["USA", "XX"])
    @test partial.InducedProduction ≈ usa_only.InducedProduction
    partial_p = @test_logs (:warn,) induced_production(m; producer_countries = ["USA", "XX"])
    @test partial_p.CountryCode == ["USA", "USA"]

    # Positional overload accepts integer codes (standardized via string.())
    m_num = _wp2_square_mrio(row_countries = ["1", "1", "2", "2"], col_countries = ["1", "2"])
    by_int = induced_production(m_num, [1], [2])
    by_str = induced_production(m_num; consumer_countries = ["1"], producer_countries = ["2"])
    @test by_int.InducedProduction ≈ by_str.InducedProduction

    # Non-square system without Leontief factorization -> clear ArgumentError
    z_ns = IO.MatrixEntry(
        [10.0 2.0 3.0; 4.0 5.0 6.0],
        DataFrame(CountryCode = ["USA", "CHN", "DEU"]),
        DataFrame(CountryCode = ["USA", "CHN"], Sector = ["Agr", "Man"])
    )
    m_ns = MRIO(
        Z = z_ns,
        Y = IO.MatrixEntry(reshape([5.0, 6.0], 2, 1), DataFrame(Category = ["HH"]), z_ns.row_indices),
        VA = IO.MatrixEntry([2.0 3.0 1.0], z_ns.col_indices, DataFrame(Category = ["C"]))
    )
    @test_throws ArgumentError induced_production(m_ns)

    # Country-labeled (non-Code) fixtures work
    sec_country = DataFrame(Country = ["USA", "USA", "CHN", "CHN"], Sector = ["Agr", "Man", "Agr", "Man"])
    m_country = MRIO(
        Z = IO.MatrixEntry(m.T.data, sec_country, sec_country),
        Y = IO.MatrixEntry(m.FD.data, DataFrame(Country = ["USA", "CHN"]), sec_country),
        VA = IO.MatrixEntry(m.VA.data, sec_country, DataFrame(Category = ["Compensation"]))
    )
    df_country = induced_production(m_country; consumer_countries = ["USA"])
    @test nrow(df_country) == 4
    @test df_country.InducedProduction ≈ usa_only.InducedProduction
end

@testset "from_long_dataframe Robustness" begin
    data = [1.0 2.0; 3.0 4.0]
    row_df = DataFrame(CountryCode = ["USA", "CHN"], Sector = ["Agr", "Man"])
    col_df = DataFrame(CountryCode = ["USA", "CHN"], Sector = ["Agr", "Man"])
    m = IO.MatrixEntry(data, col_df, row_df)

    # Round-trip fidelity
    rt = from_long_dataframe(to_long_dataframe(m))
    @test rt.data == data
    @test rt.row_indices.CountryCode == ["USA", "CHN"]
    @test rt.col_indices.CountryCode == ["USA", "CHN"]

    # Duplicates are summed (with a warning), not last-wins
    df = to_long_dataframe(m)
    dup_df = vcat(df, df[1:1, :])
    summed = @test_logs (:warn,) from_long_dataframe(dup_df)
    @test summed.data[1, 1] == 2.0
    @test summed.data[2, 2] == 4.0

    # Missing value column -> clear ArgumentError
    @test_throws ArgumentError from_long_dataframe(df; value_col = "nope")
end

@testset "Dual-Naming Country/Sector Support" begin
    data = [10.0 20.0 30.0; 40.0 50.0 60.0; 70.0 80.0 90.0; 100.0 110.0 120.0]
    row_df = DataFrame(
        Country = ["USA", "USA", "CHN", "CHN"],
        Industry = ["Agr", "Man", "Agr", "Man"],
        Sector = ["Primary", "Secondary", "Primary", "Secondary"]
    )
    col_df = DataFrame(
        Country = ["USA", "CHN", "DEU"],
        Industry = ["Agr", "Man", "Ser"],
        Sector = ["Primary", "Secondary", "Tertiary"]
    )
    country_only_row = DataFrames.select(row_df, :Country)
    country_only_col = DataFrames.select(col_df, :Country)
    m_country = IO.MatrixEntry(data, country_only_col, country_only_row)

    rows = sum_by_country(m_country; dimension = :rows)
    @test nrow(rows) == 2
    @test rows[rows.row_Country .== "USA", :value][1] == 210.0
    @test rows[rows.row_Country .== "CHN", :value][1] == 570.0
    cols = sum_by_country(m_country; dimension = :cols)
    @test nrow(cols) == 3
    both = sum_by_country(m_country; dimension = :both)
    @test nrow(both) == 6
    cs = country_summary(m_country)
    @test nrow(cs) == 6
    @test "row_Country" in names(cs)

    # Industry-only fallback for sectors
    industry_row = DataFrames.select(row_df, :Industry)
    industry_col = DataFrame(Industry = ["Agr", "Man"])
    m_ind = IO.MatrixEntry([1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0], industry_col, industry_row)
    sec_rows = sum_by_sector(m_ind; dimension = :rows)
    @test sec_rows[sec_rows.row_Industry .== "Agr", :value][1] == 14.0
    sec_both = sum_by_sector(m_ind; dimension = :both)
    @test nrow(sec_both) == 4

    # Neither naming present -> clear ArgumentError
    m_bare = IO.MatrixEntry([1.0 2.0; 3.0 4.0], DataFrame(X = [1, 2]), DataFrame(X = [1, 2]))
    @test_throws ArgumentError sum_by_country(m_bare)
    @test_throws ArgumentError sum_by_sector(m_bare)
    @test_throws ArgumentError country_summary(m_bare)
end

@testset "pivot_matrix_to_wide Duplicate Keys" begin
    data = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
    row_df = DataFrame(Country = ["USA", "USA", "CHN", "CHN"], Sector = ["Agr", "Man", "Agr", "Man"])
    col_df = DataFrame(Sector = ["Agr", "Man"])
    m = IO.MatrixEntry(data, col_df, row_df)

    # Full keys are unique -> works
    wide = pivot_matrix_to_wide(m, [:Country, :Sector], :Sector)
    @test nrow(wide) == 4

    # Subset keys collide -> informative ArgumentError
    err = try
        pivot_matrix_to_wide(m, [:Country], :Sector)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("aggregate", sprint(showerror, err))
end

@testset "matrix_summary Edge Cases" begin
    empty_m = IO.MatrixEntry(reshape(Float64[], 0, 0), DataFrame(A = String[]), DataFrame(B = String[]))
    @test_throws ArgumentError matrix_summary(empty_m)

    # Single element: std is NaN by definition (kept as-is)
    single_m = IO.MatrixEntry(reshape([5.0], 1, 1), DataFrame(A = ["x"]), DataFrame(B = ["y"]))
    s = matrix_summary(single_m)
    @test s.total[1] == 5.0
    @test isnan(s.std[1])
end
