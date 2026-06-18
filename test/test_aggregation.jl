using Test
using Juliora
import Juliora as IO
using DataFrames

@testset "MatrixEntry Groupby and Aggregation" begin
    # Mock data: 4 sectors, 2 countries
    data = [
        1.0 2.0 3.0 4.0;
        5.0 6.0 7.0 8.0;
        9.0 10.0 11.0 12.0;
        13.0 14.0 15.0 16.0
    ]
    row_df = DataFrame(
        Country = ["USA", "USA", "CHN", "CHN"],
        Sector = ["Agriculture", "Manufacturing", "Agriculture", "Manufacturing"]
    )
    col_df = DataFrame(
        Country = ["USA", "USA", "CHN", "CHN"],
        Sector = ["Agriculture", "Manufacturing", "Agriculture", "Manufacturing"]
    )

    m = IO.MatrixEntry(data, col_df, row_df)

    # 1. Test groupby and aggregate rows (dims=1) by Country
    g_rows = IO.groupby(m, :Country; dims = 1)
    @test g_rows isa IO.GroupedMatrixEntry
    @test g_rows.dims == 1

    m_agg_rows = IO.aggregate(g_rows)
    @test size(m_agg_rows.data) == (2, 4)
    # Group 1 (USA): rows 1 & 2 -> 1+5=6, 2+6=8, 3+7=10, 4+8=12
    # Group 2 (CHN): rows 3 & 4 -> 9+13=22, 10+14=24, 11+15=26, 12+16=28
    @test m_agg_rows.data[1, :] == [6.0, 8.0, 10.0, 12.0]
    @test m_agg_rows.data[2, :] == [22.0, 24.0, 26.0, 28.0]
    @test m_agg_rows.row_indices.Country == ["USA", "CHN"]

    # 2. Test groupby and aggregate columns (dims=2) by Sector
    g_cols = IO.groupby(m, :Sector; dims = 2)
    @test g_cols isa IO.GroupedMatrixEntry
    @test g_cols.dims == 2

    m_agg_cols = IO.aggregate(g_cols)
    @test size(m_agg_cols.data) == (4, 2)
    # Group 1 (Agriculture): cols 1 & 3 -> 1+3=4, 5+7=12, 9+11=20, 13+15=28
    # Group 2 (Manufacturing): cols 2 & 4 -> 2+4=6, 6+8=14, 10+12=22, 14+16=30
    @test m_agg_cols.data[:, 1] == [4.0, 12.0, 20.0, 28.0]
    @test m_agg_cols.data[:, 2] == [6.0, 14.0, 22.0, 30.0]
    @test m_agg_cols.col_indices.Sector == ["Agriculture", "Manufacturing"]
end

@testset "MRIO Aggregation" begin
    # Setup simple MRIO matrices
    sector_indices = DataFrame(
        CountryCode = ["USA", "USA", "CHN", "CHN"],
        Sector = ["Primary", "Secondary", "Primary", "Secondary"]
    )

    z_data = [
        10.0 2.0 3.0 1.0;
        1.0 15.0 2.0 4.0;
        4.0 1.0 12.0 3.0;
        2.0 3.0 1.0 18.0
    ]
    y_data = [
        5.0 1.0;
        2.0 8.0;
        1.0 3.0;
        4.0 6.0
    ]
    va_data = [
        2.0 3.0 1.0 4.0;
        1.0 1.0 2.0 2.0
    ]

    z_matrix = IO.MatrixEntry(z_data, sector_indices, sector_indices)
    y_matrix = IO.MatrixEntry(y_data, DataFrame(CountryCode = ["USA", "CHN"]), sector_indices)

    va_col_indices = DataFrame(Category = ["Compensation", "Taxes"])
    va_matrix = IO.MatrixEntry(va_data, sector_indices, va_col_indices)

    # Mock environmental data for MRIO env field
    f_data = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]
    f_indices = DataFrame(Stressor = ["CO2", "Water"], Source = ["Fossil", "Fresh"])
    f_matrix = IO.MatrixEntry(f_data, sector_indices, f_indices)
    a_matrix = IO.MatrixEntry(f_data ./ [10.0, 20.0, 30.0, 40.0]', sector_indices, f_indices)
    env = EnvironmentalExtension(f_matrix, a_matrix)

    mrio = MRIO(
        z_matrix, # A
        z_matrix, # T
        va_matrix, # VA
        y_matrix, # FD
        IO.calculate_leontief_factorization(z_matrix), # L
        SeriesEntry([10.0, 20.0, 30.0, 40.0], sector_indices), # X
        env # env
    )

    # Aggregate by Sector (dims=1)
    mrio_agg_sector = IO.aggregate(mrio, :Sector; dims = 1)
    @test mrio_agg_sector isa MRIO

    # The row dimension of T should now be 2 (Primary and Secondary)
    @test size(mrio_agg_sector.T.data, 1) == 2
    @test mrio_agg_sector.T.row_indices.Sector == ["Primary", "Secondary"]

    # Aggregate by Country (dims=1)
    mrio_agg_country = IO.aggregate(mrio, :CountryCode; dims = 1)
    @test mrio_agg_country isa MRIO
    @test size(mrio_agg_country.T.data, 1) == 2
    @test mrio_agg_country.T.row_indices.CountryCode == ["USA", "CHN"]
end
