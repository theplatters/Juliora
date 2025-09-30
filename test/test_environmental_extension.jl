function test_environmental_extension()
    @testset "EnvironmentalExtension Constructor (Mock Data)" begin
        # Create mock environmental data since we don't have actual files
        f_data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]  # 3 stressors × 3 sectors
        sector_indices = DataFrame(
            CountryCode = ["USA", "CHN", "DEU"],
            Industry = ["Agriculture", "Manufacturing", "Services"],
            Sector = ["Primary", "Secondary", "Tertiary"]
        )
        stressor_indices = DataFrame(
            Stressor = ["CO2", "Water", "Land"],
            Source = ["Fossil", "Fresh", "Arable"]
        )
        x_output = [10.0, 20.0, 30.0]  # Total output by sector
        
        # Create environmental extension manually
        f_matrix = MatrixEntry(f_data, sector_indices, stressor_indices)
        a_matrix = MatrixEntry(f_data ./ x_output', sector_indices, stressor_indices)
        
        env_ext = EnvironmentalExtension(a_matrix, f_matrix)
        
        @test env_ext isa EnvironmentalExtension
        @test env_ext.A isa MatrixEntry
        @test env_ext.F isa MatrixEntry
        
        # Test that dimensions are correct
        @test size(env_ext.F.data) == (3, 3)
        @test size(env_ext.A.data) == (3, 3)
        
        # Test that intensity matrix is correctly calculated
        expected_intensities = [1.0/10.0 2.0/20.0 3.0/30.0; 4.0/10.0 5.0/20.0 6.0/30.0; 7.0/10.0 8.0/20.0 9.0/30.0]
        @test env_ext.A.data ≈ expected_intensities
        
        # Test that direct impacts are preserved
        @test env_ext.F.data == f_data
        
        # Test indices are correctly set
        @test env_ext.F.col_indices.CountryCode == ["USA", "CHN", "DEU"]
        @test env_ext.F.row_indices.Stressor == ["CO2", "Water", "Land"]
        @test env_ext.A.col_indices.CountryCode == ["USA", "CHN", "DEU"]
        @test env_ext.A.row_indices.Stressor == ["CO2", "Water", "Land"]
    end
    
    @testset "EnvironmentalExtension Indexing" begin
        # Create test data
        f_data = [100.0 200.0; 150.0 250.0]  # 2 stressors × 2 sectors
        sector_indices = DataFrame(
            CountryCode = ["USA", "CHN"],
            Industry = ["Agr", "Man"],
            Sector = ["Primary", "Secondary"]
        )
        stressor_indices = DataFrame(
            Stressor = ["CO2", "Water"],
            Source = ["Fossil", "Fresh"]
        )
        x_output = [1000.0, 2000.0]
        
        f_matrix = MatrixEntry(f_data, sector_indices, stressor_indices)
        a_matrix = MatrixEntry(f_data ./ x_output', sector_indices, stressor_indices)
        env_ext = EnvironmentalExtension(a_matrix, f_matrix)
        
        # Test direct access to environmental impacts
        co2_usa = env_ext.F[(Stressor="CO2", Source="Fossil"), (CountryCode="USA", Industry="Agr", Sector="Primary")]
        @test co2_usa == 100.0
        
        water_china = env_ext.F[(Stressor="Water", Source="Fresh"), (CountryCode="CHN", Industry="Man", Sector="Secondary")]
        @test water_china == 250.0
        
        # Test access to intensities
        co2_intensity_usa = env_ext.A[(Stressor="CO2", Source="Fossil"), (CountryCode="USA", Industry="Agr", Sector="Primary")]
        @test co2_intensity_usa ≈ 100.0 / 1000.0
        
        water_intensity_china = env_ext.A[(Stressor="Water", Source="Fresh"), (CountryCode="CHN", Industry="Man", Sector="Secondary")]
        @test water_intensity_china ≈ 250.0 / 2000.0
    end
    
    @testset "EnvironmentalExtension Filtering" begin
        # Create larger test dataset
        f_data = rand(4, 6) * 1000  # 4 stressors × 6 sectors
        sector_indices = DataFrame(
            CountryCode = repeat(["USA", "CHN", "DEU"], 2),
            Industry = repeat(["Agr", "Man"], 3),
            Sector = ["Primary", "Secondary", "Primary", "Secondary", "Primary", "Secondary"]
        )
        stressor_indices = DataFrame(
            Stressor = ["CO2", "CH4", "Water", "Land"],
            Source = ["Fossil", "Agr", "Fresh", "Arable"],
            Type = ["Air", "Air", "Water", "Land"]
        )
        x_output = rand(6) * 10000
        
        f_matrix = MatrixEntry(f_data, sector_indices, stressor_indices)
        a_matrix = MatrixEntry(f_data ./ x_output', sector_indices, stressor_indices)
        env_ext = EnvironmentalExtension(a_matrix, f_matrix)
        
        # Filter for CO2 emissions only
        co2_impacts = filter_rows(env_ext.F, row -> row.Stressor == "CO2")
        @test size(co2_impacts.data) == (1, 6)
        @test co2_impacts.row_indices.Stressor == ["CO2"]
        
        # Filter for air pollutants
        air_pollutants = filter_rows(env_ext.A, row -> row.Type == "Air")
        @test size(air_pollutants.data) == (2, 6)  # CO2 and CH4
        @test air_pollutants.row_indices.Stressor == ["CO2", "CH4"]
        
        # Filter for USA impacts
        usa_impacts = filter_cols(env_ext.F, col -> col.CountryCode == "USA")
        @test size(usa_impacts.data) == (4, 2)  # All stressors × 2 USA sectors
        @test all(usa_impacts.col_indices.CountryCode .== "USA")
        
        # Filter for primary sectors
        primary_impacts = filter_cols(env_ext.A, col -> col.Sector == "Primary")
        @test size(primary_impacts.data) == (4, 3)  # All stressors × 3 primary sectors
        @test all(primary_impacts.col_indices.Sector .== "Primary")
        
        # Combined filtering
        co2_usa_primary = filter_matrix(
            env_ext.F,
            row -> row.Stressor == "CO2",
            col -> col.CountryCode == "USA" && col.Sector == "Primary"
        )
        @test size(co2_usa_primary.data) == (1, 1)
        @test co2_usa_primary.row_indices.Stressor == ["CO2"]
        @test co2_usa_primary.col_indices.CountryCode == ["USA"]
    end
    
    @testset "EnvironmentalExtension Analysis" begin
        # Create test data with known values for analysis
        f_data = [10.0 20.0 30.0; 40.0 50.0 60.0; 70.0 80.0 90.0]
        sector_indices = DataFrame(
            CountryCode = ["USA", "CHN", "DEU"],
            Industry = ["Agr", "Man", "Ser"],
            Sector = ["Primary", "Secondary", "Tertiary"]
        )
        stressor_indices = DataFrame(
            Stressor = ["CO2", "Water", "Land"],
            Source = ["Fossil", "Fresh", "Arable"]
        )
        x_output = [100.0, 200.0, 300.0]
        
        f_matrix = MatrixEntry(f_data, sector_indices, stressor_indices)
        a_matrix = MatrixEntry(f_data ./ x_output', sector_indices, stressor_indices)
        env_ext = EnvironmentalExtension(a_matrix, f_matrix)
        
        # Test total impacts by stressor
        using Tidier
        
        stressor_totals = env_ext.F |> df -> @chain df begin
            @group_by(row_Stressor)
            @summarize(total_impact = sum(value))
            @arrange(desc(total_impact))
        end
        
        @test nrow(stressor_totals) == 3
        @test stressor_totals.total_impact == [240.0, 150.0, 60.0]  # Land, Water, CO2
        @test stressor_totals.row_Stressor == ["Land", "Water", "CO2"]
        
        # Test impacts by country
        country_totals = env_ext.F |> df -> @chain df begin
            @group_by(col_CountryCode)
            @summarize(total_impact = sum(value))
            @arrange(col_CountryCode)
        end
        
        @test nrow(country_totals) == 3
        @test country_totals.total_impact == [120.0, 150.0, 180.0]  # USA, CHN, DEU
        @test country_totals.col_CountryCode == ["CHN", "DEU", "USA"]  # Alphabetical order
        
        # Test intensity analysis
        max_intensities = env_ext.A |> df -> @chain df begin
            @group_by(row_Stressor)
            @summarize(max_intensity = maximum(value))
            @arrange(desc(max_intensity))
        end
        
        @test nrow(max_intensities) == 3
        # Check that intensities are correctly calculated
        expected_max = [90.0/300.0, 60.0/300.0, 30.0/300.0]  # Land, Water, CO2 max intensities
        @test max_intensities.max_intensity ≈ expected_max
    end
    
    @testset "EnvironmentalExtension Edge Cases" begin
        # Test with zero output (should handle division by zero)
        f_data = [1.0 2.0; 3.0 4.0]
        sector_indices = DataFrame(CountryCode = ["USA", "CHN"], Sector = ["A", "B"])
        stressor_indices = DataFrame(Stressor = ["CO2", "Water"], Source = ["F", "W"])
        x_output = [0.0, 10.0]  # Zero output for first sector
        
        # The constructor should handle this gracefully (usually by replacing 0 with 1 or Inf)
        f_matrix = MatrixEntry(f_data, sector_indices, stressor_indices)
        
        # Manual calculation with zero handling
        safe_x = replace(x_output, 0.0 => 1.0)  # Replace zeros to avoid division by zero
        a_matrix = MatrixEntry(f_data ./ safe_x', sector_indices, stressor_indices)
        
        env_ext = EnvironmentalExtension(a_matrix, f_matrix)
        
        @test size(env_ext.F.data) == (2, 2)
        @test size(env_ext.A.data) == (2, 2)
        @test !any(isinf.(env_ext.A.data))  # No infinite values
        @test !any(isnan.(env_ext.A.data))  # No NaN values
        
        # Test with all zero impacts
        zero_f_data = zeros(2, 2)
        zero_f_matrix = MatrixEntry(zero_f_data, sector_indices, stressor_indices)
        zero_a_matrix = MatrixEntry(zero_f_data ./ safe_x', sector_indices, stressor_indices)
        zero_env_ext = EnvironmentalExtension(zero_a_matrix, zero_f_matrix)
        
        @test all(zero_env_ext.F.data .== 0.0)
        @test all(zero_env_ext.A.data .== 0.0)
        
        # Test with single stressor/sector
        single_f = reshape([42.0], 1, 1)
        single_sector = DataFrame(CountryCode = ["USA"], Sector = ["Total"])
        single_stressor = DataFrame(Stressor = ["CO2"], Source = ["All"])
        single_x = [100.0]
        
        single_f_matrix = MatrixEntry(single_f, single_sector, single_stressor)
        single_a_matrix = MatrixEntry(single_f ./ single_x', single_sector, single_stressor)
        single_env_ext = EnvironmentalExtension(single_a_matrix, single_f_matrix)
        
        @test size(single_env_ext.F.data) == (1, 1)
        @test single_env_ext.F.data[1, 1] == 42.0
        @test single_env_ext.A.data[1, 1] ≈ 0.42
    end
end