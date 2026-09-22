# Helper function for Eora filtering
function filter_eora(
        eora::MRIO;
        row_countries = nothing, col_countries = nothing,
        row_industries = nothing, col_industries = nothing,
        row_regions = nothing, col_regions = nothing,
        final_demand_countries = nothing,
        row_filter = nothing, col_filter = nothing
    )

    # Start with all indices
    row_mask = trues(size(eora.Z.data, 1))
    col_mask = trues(size(eora.Z.data, 2))
    fd_mask = trues(size(eora.Y.data, 2))

    # Apply country filters
    if row_countries !== nothing
        row_mask .&= [r.CountryCode in row_countries for r in eachrow(eora.Z.row_indices)]
    end
    if col_countries !== nothing
        col_mask .&= [c.CountryCode in col_countries for c in eachrow(eora.Z.col_indices)]
    end

    # Apply industry filters
    if row_industries !== nothing
        row_mask .&= [r.Industry in row_industries for r in eachrow(eora.Z.row_indices)]
    end
    if col_industries !== nothing
        col_mask .&= [c.Industry in col_industries for c in eachrow(eora.Z.col_indices)]
    end

    # Apply region filters
    if row_regions !== nothing
        row_mask .&= [r.Region in row_regions for r in eachrow(eora.Z.row_indices)]
    end
    if col_regions !== nothing
        col_mask .&= [c.Region in col_regions for c in eachrow(eora.Z.col_indices)]
    end

    # Apply custom filters
    if row_filter !== nothing
        row_mask .&= [row_filter(r) for r in eachrow(eora.Z.row_indices)]
    end
    if col_filter !== nothing
        col_mask .&= [col_filter(c) for c in eachrow(eora.Z.col_indices)]
    end

    # Apply final demand filter
    if final_demand_countries !== nothing
        fd_mask .&= [fd.CountryCode in final_demand_countries for fd in eachrow(eora.Y.col_indices)]
    end

    # Create filtered matrices
    filtered_Z = IO.MatrixEntry(
        eora.Z.data[row_mask, col_mask],
        eora.Z.col_indices[col_mask, :],
        eora.Z.row_indices[row_mask, :]
    )

    filtered_Y = IO.MatrixEntry(
        eora.Y.data[row_mask, fd_mask],
        eora.Y.col_indices[fd_mask, :],
        eora.Y.row_indices[row_mask, :]
    )

    filtered_VA = IO.MatrixEntry(
        eora.VA.data[:, col_mask],
        eora.VA.col_indices[col_mask, :],
        eora.VA.row_indices
    )

    return MRIO(Z = filtered_Z, Y = filtered_Y, VA = filtered_VA)
end

# Analysis functions
function calculate_total_output(eora::MRIO)
    intermediate_demand = sum(eora.Z.data, dims = 2)
    final_demand = sum(eora.Y.data, dims = 2)
    total_output = vec(intermediate_demand + final_demand)

    return SeriesEntry(total_output, eora.Z.row_indices)
end

function calculate_technical_coefficients(eora::MRIO)
    total_output = calculate_total_output(eora)
    # Avoid division by zero
    X_safe = max.(total_output.data, 1.0e-10)
    A_data = eora.Z.data ./ X_safe'  # Broadcasting division

    return IO.MatrixEntry(A_data, eora.Z.col_indices, eora.Z.row_indices)
end

function calculate_leontief_inverse(eora::MRIO)
    A = calculate_technical_coefficients(eora)
    n = size(A.data, 1)
    I_matrix = Matrix{Float64}(LinearAlgebra.I, n, n)
    L_data = inv(I_matrix - A.data)

    return IO.MatrixEntry(L_data, eora.Z.col_indices, eora.Z.row_indices)
end

function calculate_multipliers(eora::MRIO)
    L = calculate_leontief_inverse(eora)
    multipliers = vec(sum(L.data, dims = 1))

    return SeriesEntry(multipliers, eora.Z.col_indices)
end

@testset "Eora MRIO Tests" begin

    @testset "Eora Structure Construction" begin
        # Create minimal test data that mimics Eora structure
        # Eora has countries and industries organized in a specific way

        # Create country data
        countries = ["USA", "CHN", "AUS", "DEU", "JPN"]
        n_countries = length(countries)

        # Create industry data (simplified version of Eora's 26 industries)
        industries = ["Agriculture", "Mining", "Manufacturing", "Construction", "Services"]
        n_industries = length(industries)

        # Create total dimensions
        n_total = n_countries * n_industries  # 25 total sectors

        # Create Z matrix (intermediate demand) - square matrix
        Z_data = rand(n_total, n_total) * 1000  # Scale to reasonable economic values

        # Create row indices (supplying sectors)
        row_df = DataFrame()
        row_df.CountryCode = repeat(countries, n_industries)
        row_df.Industry = repeat(industries, outer = n_countries)
        row_df.Sector = [row_df.CountryCode[i] * "_" * row_df.Industry[i] for i in 1:nrow(row_df)]

        # Create column indices (demanding sectors) - same structure as rows for Z matrix
        col_df = copy(row_df)

        # Create final demand matrix Y
        # Final demand typically has countries as columns, not sectors
        Y_data = rand(n_total, n_countries) * 500
        Y_col_df = DataFrame(CountryCode = countries, DemandType = repeat(["FinalDemand"], n_countries))

        # Create value added vector (one value per sector)
        VA_data = rand(1, n_total) * 200
        VA_row_df = DataFrame(ValueAddedType = ["TotalValueAdded"])

        # Test Eora construction
        eora = MRIO(
            Z = IO.MatrixEntry(Z_data, col_df, row_df),
            Y = IO.MatrixEntry(Y_data, Y_col_df, row_df),
            VA = IO.MatrixEntry(VA_data, row_df, VA_row_df)
        )

        @test isa(eora, MRIO)
        @test isa(eora.Z, IO.MatrixEntry)
        @test isa(eora.Y, IO.MatrixEntry)
        @test isa(eora.VA, IO.MatrixEntry)

        # Test dimensions are consistent
        @test size(eora.Z.data, 1) == size(eora.Z.data, 2) == n_total  # Z is square
        @test size(eora.Y.data, 1) == n_total  # Y rows match Z
        @test size(eora.Y.data, 2) == n_countries  # Y columns are countries
        @test size(eora.VA.data, 1) == 1  # VA is single row
        @test size(eora.VA.data, 2) == n_total  # VA columns match sectors

        # Test that country codes are preserved
        @test Set(eora.Z.row_indices.CountryCode) == Set(countries)
        @test Set(eora.Z.col_indices.CountryCode) == Set(countries)
        @test Set(eora.Y.col_indices.CountryCode) == Set(countries)

        # Test that industries are preserved
        @test Set(eora.Z.row_indices.Industry) == Set(industries)
        @test Set(eora.Z.col_indices.Industry) == Set(industries)
        @test Set(eora.Y.row_indices.Industry) == Set(industries)
    end

    @testset "Matrix Relationships" begin
        # Create simple 2x2 country, 2 industry test case
        countries = ["USA", "CHN"]
        industries = ["Agr", "Man"]

        # Create indices
        sectors = [(c, i) for c in countries, i in industries]
        flat_sectors = vec(sectors)

        row_df = DataFrame(
            CountryCode = [s[1] for s in flat_sectors],
            Industry = [s[2] for s in flat_sectors],
            Sector = [s[1] * "_" * s[2] for s in flat_sectors]
        )
        col_df = copy(row_df)

        # Create matrices with known relationships
        Z_data = [
            10.0 5.0 3.0 2.0;   # USA_Agr supplies to all sectors
            8.0 15.0 4.0 6.0;   # USA_Man supplies to all sectors
            2.0 3.0 20.0 8.0;   # CHN_Agr supplies to all sectors
            4.0 7.0 12.0 25.0
        ]  # CHN_Man supplies to all sectors

        Y_data = [
            100.0 80.0;   # Final demand by country
            120.0 90.0;
            60.0 150.0;
            80.0 110.0
        ]

        Y_col_df = DataFrame(CountryCode = countries, DemandType = ["FinalDemand", "FinalDemand"])

        VA_data = [50.0 60.0 40.0 45.0]  # Value added by sector
        VA_row_df = DataFrame(ValueAddedType = ["TotalValueAdded"])

        eora = MRIO(
            Z = IO.MatrixEntry(Z_data, col_df, row_df),
            Y = IO.MatrixEntry(Y_data, Y_col_df, row_df),
            VA = IO.MatrixEntry(VA_data, row_df, VA_row_df)
        )

        # Test basic accounting identity: X = Z * 1 + Y * 1 (total output)
        Z_row_sums = sum(eora.Z.data, dims = 2)  # Intermediate demand
        Y_row_sums = sum(eora.Y.data, dims = 2)  # Final demand
        total_output = Z_row_sums + Y_row_sums

        @test size(total_output) == (4, 1)  # 4 sectors, 1 column
        @test all(total_output .> 0)  # All outputs should be positive

        # Test that value added makes sense (should be positive)
        @test all(eora.VA.data .> 0)
        @test size(eora.VA.data) == (1, 4)  # 1 VA type, 4 sectors

        # Test matrix dimensions consistency
        @test size(eora.Z.data, 1) == size(eora.Y.data, 1)  # Same sectors
        @test size(eora.Z.data, 2) == size(eora.VA.data, 2)  # Same sectors
    end

    @testset "Indexing and Access" begin
        # Create test Eora with known values
        countries = ["USA", "CHN"]
        industries = ["Agr", "Man"]

        row_df = DataFrame(
            CountryCode = repeat(countries, inner = 2),
            Industry = repeat(industries, outer = 2)
        )
        col_df = copy(row_df)

        Z_data = [
            1.0 2.0 3.0 4.0;
            5.0 6.0 7.0 8.0;
            9.0 10.0 11.0 12.0;
            13.0 14.0 15.0 16.0
        ]

        Y_data = [100.0 200.0; 300.0 400.0; 500.0 600.0; 700.0 800.0]
        Y_col_df = DataFrame(CountryCode = countries)

        VA_data = [10.0 20.0 30.0 40.0]
        VA_row_df = DataFrame(ValueAddedType = ["TotalValueAdded"])

        eora = MRIO(
            Z = IO.MatrixEntry(Z_data, col_df, row_df),
            Y = IO.MatrixEntry(Y_data, Y_col_df, row_df),
            VA = IO.MatrixEntry(VA_data, row_df, VA_row_df)
        )

        # Test Z matrix indexing
        usa_agr = (CountryCode = "USA", Industry = "Agr")
        usa_man = (CountryCode = "USA", Industry = "Man")
        chn_agr = (CountryCode = "CHN", Industry = "Agr")
        chn_man = (CountryCode = "CHN", Industry = "Man")

        @test eora.Z[usa_agr, usa_agr] == 1.0
        @test eora.Z[usa_agr, usa_man] == 2.0
        @test eora.Z[usa_agr, chn_agr] == 3.0
        @test eora.Z[usa_agr, chn_man] == 4.0

        # Test Y matrix indexing
        usa_country = (CountryCode = "USA",)
        chn_country = (CountryCode = "CHN",)

        @test eora.Y[usa_agr, usa_country] == 100.0
        @test eora.Y[usa_agr, chn_country] == 200.0
        @test eora.Y[chn_man, usa_country] == 700.0
        @test eora.Y[chn_man, chn_country] == 800.0

        # Test VA indexing
        va_type = (ValueAddedType = "TotalValueAdded",)
        @test eora.VA[va_type, usa_agr] == 10.0
        @test eora.VA[va_type, usa_man] == 20.0
        @test eora.VA[va_type, chn_agr] == 30.0
        @test eora.VA[va_type, chn_man] == 40.0

        # Test boolean indexing
        usa_sectors = eora.Z.row_indices.CountryCode .== "USA"
        usa_z_data = eora.Z[usa_sectors, :]
        @test size(usa_z_data.data, 1) == 2  # 2 USA sectors
        @test size(usa_z_data.data, 2) == 4  # All 4 sectors as columns

        # Test country-level aggregation
        usa_final_demand = eora.Y[:, eora.Y.col_indices.CountryCode .== "USA"]
        @test size(usa_final_demand.data) == (4, 1)  # All sectors, USA demand only
        @test usa_final_demand.data[1, 1] == 100.0  # USA_Agr -> USA demand
    end

    @testset "Filtering Operations" begin
        countries = ["USA", "CHN", "DEU", "JPN"]
        industries = ["Agr", "Man", "Ser"]

        # Create larger test case
        n_sectors = length(countries) * length(industries)  # 12 sectors

        row_df = DataFrame(
            CountryCode = repeat(countries, length(industries)),
            Industry = repeat(industries, outer = length(countries)),
            Region = repeat(["NA", "Asia", "EU", "Asia"], length(industries)),
            Developed = repeat([true, false, true, true], length(industries))
        )
        col_df = copy(row_df)

        Z_data = rand(n_sectors, n_sectors) * 100
        Y_data = rand(n_sectors, length(countries)) * 50
        Y_col_df = DataFrame(CountryCode = countries, Region = ["NA", "Asia", "EU", "Asia"])

        VA_data = rand(1, n_sectors) * 20
        VA_row_df = DataFrame(ValueAddedType = ["TotalValueAdded"])

        eora = MRIO(
            Z = IO.MatrixEntry(Z_data, col_df, row_df),
            Y = IO.MatrixEntry(Y_data, Y_col_df, row_df),
            VA = IO.MatrixEntry(VA_data, row_df, VA_row_df)
        )

        # Test filtering by country
        usa_eora = filter_eora(eora, row_countries = ["USA"], col_countries = ["USA", "CHN"])
        @test Set(usa_eora.Z.row_indices.CountryCode) == Set(["USA"])
        @test Set(usa_eora.Z.col_indices.CountryCode) == Set(["USA", "CHN"])
        @test size(usa_eora.Z.data, 1) == 3  # 3 USA industries
        @test size(usa_eora.Z.data, 2) == 6  # 3 USA + 3 CHN industries

        # Test filtering by industry
        agr_eora = filter_eora(eora, row_industries = ["Agr"], col_industries = ["Agr", "Man"])
        @test Set(agr_eora.Z.row_indices.Industry) == Set(["Agr"])
        @test Set(agr_eora.Z.col_indices.Industry) == Set(["Agr", "Man"])
        @test size(agr_eora.Z.data, 1) == 4  # 4 countries with Agr
        @test size(agr_eora.Z.data, 2) == 8  # 4 countries × 2 industries

        # Test filtering by region
        asia_eora = filter_eora(eora, row_regions = ["Asia"])
        asia_countries = unique(row_df[row_df.Region .== "Asia", :CountryCode])
        @test Set(asia_eora.Z.row_indices.CountryCode) == Set(asia_countries)

        # Test filtering final demand
        eu_demand = filter_eora(eora, final_demand_countries = ["DEU"])
        @test size(eu_demand.Y.data, 2) == 1  # Only DEU final demand
        @test eu_demand.Y.col_indices.CountryCode == ["DEU"]

        # Test complex filtering
        developed_man = filter_eora(
            eora,
            row_industries = ["Man"],
            row_filter = row -> row.Developed == true
        )
        @test all(developed_man.Z.row_indices.Developed .== true)
        @test all(developed_man.Z.row_indices.Industry .== "Man")
    end

    @testset "Analysis Functions" begin
        # Create simple test case for analysis
        countries = ["USA", "CHN"]
        industries = ["Agr", "Man"]

        row_df = DataFrame(
            CountryCode = repeat(countries, inner = 2),
            Industry = repeat(industries, outer = 2)
        )
        col_df = copy(row_df)

        # Create matrices with known structure for testing
        Z_data = [
            0.1 0.2 0.05 0.1;   # Low intermediate use within Agr
            0.3 0.4 0.15 0.2;   # Higher intermediate use for Man
            0.1 0.1 0.2 0.15;   # Cross-country flows
            0.2 0.15 0.25 0.3
        ]

        Y_data = [50.0 30.0; 80.0 60.0; 40.0 70.0; 90.0 100.0]
        Y_col_df = DataFrame(CountryCode = countries)

        VA_data = [20.0 25.0 18.0 22.0]
        VA_row_df = DataFrame(ValueAddedType = ["TotalValueAdded"])

        eora = MRIO(
            Z = IO.MatrixEntry(Z_data, col_df, row_df),
            Y = IO.MatrixEntry(Y_data, Y_col_df, row_df),
            VA = IO.MatrixEntry(VA_data, row_df, VA_row_df)
        )

        # Test total output calculation
        total_output = calculate_total_output(eora)
        @test isa(total_output, SeriesEntry)
        @test length(total_output.data) == 4  # 4 sectors
        @test all(total_output.data .> 0)  # All outputs positive

        # Verify accounting identity: X = Z*1 + Y*1
        expected_output = vec(sum(Z_data, dims = 2) + sum(Y_data, dims = 2))
        @test total_output.data ≈ expected_output rtol = 1.0e-10

        # Test technical coefficients calculation
        tech_coeffs = calculate_technical_coefficients(eora)
        @test isa(tech_coeffs, IO.MatrixEntry)
        @test size(tech_coeffs.data) == size(Z_data)

        # Technical coefficients should be Z / X (column-wise division)
        X_diag = Diagonal(total_output.data)
        expected_A = Z_data / X_diag
        @test tech_coeffs.data ≈ expected_A rtol = 1.0e-10

        # Test Leontief inverse calculation
        leontief = calculate_leontief_inverse(eora)
        @test isa(leontief, IO.MatrixEntry)
        @test size(leontief.data) == size(Z_data)

        # Leontief inverse should be (I - A)^(-1)
        I_matrix = Matrix{Float64}(LinearAlgebra.I, 4, 4)
        expected_L = inv(I_matrix - tech_coeffs.data)
        @test leontief.data ≈ expected_L rtol = 1.0e-10

        # Test multiplier calculation
        multipliers = calculate_multipliers(eora)
        @test isa(multipliers, SeriesEntry)
        @test length(multipliers.data) == 4
        @test all(multipliers.data .>= 1.0)  # Multipliers should be >= 1

        # Multipliers should be column sums of Leontief inverse
        expected_mult = vec(sum(leontief.data, dims = 1))
        @test multipliers.data ≈ expected_mult rtol = 1.0e-10
    end
end


@testset "MRIO Keyword Constructor Honesty" begin
    sec_idx = DataFrame(CountryCode = ["USA", "CHN"], Sector = ["Agr", "Man"])
    fd_idx = DataFrame(CountryCode = ["USA", "CHN"])
    va_idx = DataFrame(Category = ["Compensation"])

    @testset "square system has real L/X and no env" begin
        z = IO.MatrixEntry([10.0 2.0; 3.0 15.0], sec_idx, sec_idx)
        y = IO.MatrixEntry([5.0 1.0; 2.0 8.0], fd_idx, sec_idx)
        va = IO.MatrixEntry([2.0 3.0], sec_idx, va_idx)
        m = MRIO(Z = z, Y = y, VA = va)
        @test m.env === nothing
        @test m.L isa Juliora.LeontiefFactorization
        @test m.X.data ≈ [18.0, 28.0]
        @test stressors(m) == String[]
        @test_throws ArgumentError environmental_impact(m, [1.0, 2.0])
        @test_throws ArgumentError environmental_impact(m, SeriesEntry([1.0, 2.0], sec_idx))
        @test_throws ArgumentError environmental_impact(m, [1.0 2.0; 3.0 4.0])
        # induced_production works on square systems with real L
        df = induced_production(m)
        @test nrow(df) == 2
    end

    @testset "non-square system has no L and no env" begin
        z = IO.MatrixEntry([10.0 2.0 3.0; 4.0 5.0 6.0], DataFrame(CountryCode = ["USA", "CHN", "DEU"]), sec_idx)
        y = IO.MatrixEntry(reshape([5.0, 6.0], 2, 1), DataFrame(Category = ["HH"]), sec_idx)
        va = IO.MatrixEntry([2.0 3.0 1.0], DataFrame(CountryCode = ["USA", "CHN", "DEU"]), va_idx)
        m = MRIO(Z = z, Y = y, VA = va)
        @test m.env === nothing
        @test m.L === nothing
        @test m.X.data == zeros(3)
        @test m.A.data == z.data  # zero output guard keeps raw flows
        @test_throws ArgumentError environmental_impact(m, [1.0, 2.0, 3.0])
        @test_throws ArgumentError induced_production(m)
    end

    @testset "cross-field validation" begin
        z = IO.MatrixEntry([10.0 2.0; 3.0 15.0], sec_idx, sec_idx)
        y_bad = IO.MatrixEntry([5.0 1.0; 2.0 8.0; 1.0 1.0], fd_idx, DataFrame(CountryCode = ["USA", "CHN", "DEU"]))
        va = IO.MatrixEntry([2.0 3.0], sec_idx, va_idx)
        @test_throws DimensionMismatch MRIO(Z = z, Y = y_bad, VA = va)
        y = IO.MatrixEntry([5.0 1.0; 2.0 8.0], fd_idx, sec_idx)
        va_bad = IO.MatrixEntry([2.0 3.0 1.0], DataFrame(CountryCode = ["USA", "CHN", "DEU"]), va_idx)
        @test_throws DimensionMismatch MRIO(Z = z, Y = y, VA = va_bad)
    end

    @testset "zero-output sectors warn but keep raw flows" begin
        z = IO.MatrixEntry([10.0 2.0; 3.0 15.0], sec_idx, sec_idx)
        a = @test_logs (:warn,) IO.calculate_technical_coefficients(z, [10.0, 0.0])
        @test a.data[:, 2] == [2.0, 15.0]
        @test_logs IO.calculate_technical_coefficients(z, [10.0, 20.0])
    end

    @testset "Eora path validation" begin
        @test_throws ArgumentError Eora("/tmp/opencode/wp2-nonexistent-dir/")
        mktempdir() do dir
            @test_throws ArgumentError Eora(dir)
        end
    end
end

@testset "Eora File Loading Layout" begin
    function write_eora_fixture(dir; va_cols = 2, q_cols = 2)
        mkpath(dir)
        open(joinpath(dir, "T.txt"), "w") do io
            write(io, "1.0,2.0\n3.0,4.0\n")
        end
        open(joinpath(dir, "VA.txt"), "w") do io
            write(io, join(fill("10.0", va_cols), ",") * "\n")
        end
        open(joinpath(dir, "FD.txt"), "w") do io
            write(io, "5.0,6.0\n7.0,8.0\n")
        end
        open(joinpath(dir, "labels_T.txt"), "w") do io
            write(io, "1\tUSA\tAgr\tPrimary\n2\tCHN\tMan\tSecondary\n")
        end
        open(joinpath(dir, "labels_VA.txt"), "w") do io
            write(io, "1\tCompensation\n")
        end
        open(joinpath(dir, "labels_FD.txt"), "w") do io
            write(io, "1\tUSA\tHH\tC1\n2\tCHN\tHH\tC2\n")
        end
        open(joinpath(dir, "Q.txt"), "w") do io
            write(io, join(fill("0.5", q_cols), ",") * "\n")
        end
        open(joinpath(dir, "labels_Q.txt"), "w") do io
            write(io, "CO2\tFossil\n")
        end
    end

    mktempdir() do dir
        write_eora_fixture(dir)
        # Works with and without trailing slash (joinpath)
        e1 = Eora(dir)
        e2 = Eora(dir * "/")
        @test e1 isa MRIO
        @test size(e1.T.data) == (2, 2)
        @test size(e1.VA.data) == (1, 2)
        @test size(e1.FD.data) == (2, 2)
        @test size(e1.env.F.data) == (1, 2)
        @test e2.T.data == e1.T.data
    end

    mktempdir() do dir
        write_eora_fixture(dir; va_cols = 3)
        @test_throws DimensionMismatch Eora(dir)
    end

    mktempdir() do dir
        write_eora_fixture(dir; q_cols = 3)
        @test_throws DimensionMismatch Eora(dir)
    end
end
