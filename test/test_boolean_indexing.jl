function test_boolean_indexing()
    @testset "Boolean Indexing Setup" begin
        # Create test data
        data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU", "JPN"],
            Sector = ["Agr", "Man", "Ser", "Min"],
            Developed = [true, false, true, true]
        )
        col_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Sector = ["Agr", "Man", "Ser"],
            EU = [false, false, true]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        @testset "Boolean Indexing Both Dimensions" begin
            # Test filtering both rows and columns
            developed_mask = matrix_entry.row_indices.Developed
            eu_mask = matrix_entry.col_indices.EU
            
            filtered = matrix_entry[developed_mask, eu_mask]
            
            @test size(filtered.data) == (3, 1)  # USA, DEU, JPN rows × DEU column
            @test filtered.data == [3.0; 6.0; 9.0]
            @test filtered.row_indices.Country == ["USA", "DEU", "JPN"]
            @test filtered.col_indices.Country == ["DEU"]
            
            # Test with all false mask
            no_rows = [false, false, false, false]
            no_cols = [false, false, false]
            empty_filtered = matrix_entry[no_rows, no_cols]
            
            @test size(empty_filtered.data) == (0, 0)
            @test nrow(empty_filtered.row_indices) == 0
            @test nrow(empty_filtered.col_indices) == 0
            
            # Test with all true mask
            all_rows = [true, true, true, true]
            all_cols = [true, true, true]
            full_filtered = matrix_entry[all_rows, all_cols]
            
            @test size(full_filtered.data) == (4, 3)
            @test full_filtered.data == data
        end
        
        @testset "Boolean Indexing Dimension Mismatch" begin
            # Test wrong length masks
            short_row_mask = [true, false]  # Should be length 4
            @test_throws AssertionError matrix_entry[short_row_mask, [true, true, true]]
            
            long_col_mask = [true, false, true, false, true]  # Should be length 3
            @test_throws AssertionError matrix_entry[[true, true, true, true], long_col_mask]
        end
    end
    
    @testset "Boolean Indexing Rows Only" begin
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Type = ["Developed", "Developing", "Developed"]
        )
        col_df = DataFrame(
            Sector = ["Agr", "Man"],
            Primary = [true, false]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Filter only developed countries (keep all columns)
        developed_mask = matrix_entry.row_indices.Type .== "Developed"
        developed_only = matrix_entry[developed_mask, :]
        
        @test size(developed_only.data) == (2, 2)
        @test developed_only.data == [1.0 2.0; 5.0 6.0]
        @test developed_only.row_indices.Country == ["USA", "DEU"]
        @test developed_only.col_indices.Sector == ["Agr", "Man"]  # All columns preserved
        
        # Test single row selection
        usa_only = matrix_entry[[true, false, false], :]
        @test size(usa_only.data) == (1, 2)
        @test usa_only.data == [1.0 2.0]
        @test usa_only.row_indices.Country == ["USA"]
        
        # Test wrong mask length
        @test_throws AssertionError matrix_entry[[true, false], :]  # Should be length 3
    end
    
    @testset "Boolean Indexing Columns Only" begin
        data = [1.0 2.0 3.0; 4.0 5.0 6.0]
        row_df = DataFrame(Country = ["USA", "CHN"])
        col_df = DataFrame(
            Sector = ["Agr", "Man", "Ser"],
            Services = [false, false, true]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Filter only service sectors (keep all rows)
        service_mask = matrix_entry.col_indices.Services
        services_only = matrix_entry[:, service_mask]
        
        @test size(services_only.data) == (2, 1)
        @test services_only.data == [3.0; 6.0]
        @test services_only.row_indices.Country == ["USA", "CHN"]  # All rows preserved
        @test services_only.col_indices.Sector == ["Ser"]
        
        # Test multiple column selection
        non_agr_mask = matrix_entry.col_indices.Sector .!= "Agr"
        non_agr = matrix_entry[:, non_agr_mask]
        
        @test size(non_agr.data) == (2, 2)
        @test non_agr.data == [2.0 3.0; 5.0 6.0]
        @test non_agr.col_indices.Sector == ["Man", "Ser"]
        
        # Test wrong mask length
        @test_throws AssertionError matrix_entry[:, [true, false]]  # Should be length 3
    end
    
    @testset "Boolean Indexing Complex Conditions" begin
        data = rand(5, 4)
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU", "JPN", "BRA"],
            GDP = [20000, 14000, 4000, 5000, 2000],
            Developed = [true, false, true, true, false]
        )
        col_df = DataFrame(
            Sector = ["Agr", "Man", "Ser", "Min"],
            Share = [0.1, 0.3, 0.5, 0.1],
            Important = [false, true, true, false]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Complex row condition: developed countries with GDP > 10000
        rich_developed = (matrix_entry.row_indices.Developed) .& (matrix_entry.row_indices.GDP .> 10000)
        
        # Complex column condition: important sectors with share > 0.2
        major_important = (matrix_entry.col_indices.Important) .& (matrix_entry.col_indices.Share .> 0.2)
        
        filtered = matrix_entry[rich_developed, major_important]
        
        # Should select USA, JPN (rich developed) × Man, Ser (major important)
        @test size(filtered.data) == (2, 2)
        @test filtered.row_indices.Country == ["USA", "JPN"]
        @test filtered.col_indices.Sector == ["Man", "Ser"]
        
        # Test combining with negation
        not_developed = .!matrix_entry.row_indices.Developed
        developing_filtered = matrix_entry[not_developed, :]
        
        @test size(developing_filtered.data) == (2, 4)  # CHN, BRA
        @test developing_filtered.row_indices.Country == ["CHN", "BRA"]
    end
    
    @testset "Boolean Indexing Preserves Structure" begin
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        row_df = DataFrame(
            Country = ["USA", "CHN", "DEU"],
            Region = ["NA", "Asia", "EU"]
        )
        col_df = DataFrame(
            Sector = ["Goods", "Services"],
            Tradable = [true, false]
        )
        
        matrix_entry = MatrixEntry(data, col_df, row_df)
        
        # Test that filtered MatrixEntry maintains all structure
        asian_mask = matrix_entry.row_indices.Region .== "Asia"
        asian_filtered = matrix_entry[asian_mask, :]
        
        @test asian_filtered isa MatrixEntry
        @test haskey(asian_filtered.row_lookup, (Country="CHN", Region="Asia"))
        @test haskey(asian_filtered.col_lookup, (Sector="Goods", Tradable=true))
        @test haskey(asian_filtered.col_lookup, (Sector="Services", Tradable=false))
        
        # Test that lookups work correctly
        @test asian_filtered[(Country="CHN", Region="Asia"), (Sector="Goods", Tradable=true)] == 3.0
        @test asian_filtered[(Country="CHN", Region="Asia"), (Sector="Services", Tradable=false)] == 4.0
    end
end