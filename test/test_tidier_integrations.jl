using Test
using Juliora
using DataFrames
using Tidier
using LinearAlgebra

@testset "Tidier.jl Integration Tests" begin
    # Create test data
    data = [10.0 20.0 30.0; 40.0 50.0 60.0; 70.0 80.0 90.0]
    row_df = DataFrame(
        Country = ["USA", "CHN", "DEU"],
        Region = ["NA", "Asia", "EU"],
        GDP = [20000, 14000, 4000],
        Developed = [true, false, true]
    )
    col_df = DataFrame(
        Sector = ["Agr", "Man", "Ser"],
        Type = ["Primary", "Secondary", "Tertiary"]
    )

    me = MatrixEntry(data, col_df, row_df)

    # Create SeriesEntry test data
    se_data = [1.5, 2.5, 3.5]
    se = SeriesEntry(se_data, col_df)

    # Create LeontiefFactorization test data
    # Matrix A has to have eigenvalues with magnitude < 1 for Leontief to make physical sense
    # but mathematically any invertible (I-A) works.
    a_data = [0.1 0.2 0.0; 0.3 0.1 0.2; 0.0 0.1 0.1]
    a_me = MatrixEntry(a_data, col_df, row_df)
    lf = Juliora.calculate_leontief_factorization(a_me)

    @testset "Helper Functions (update_row_indices & update_col_indices)" begin
        new_row_df = copy(row_df)
        new_row_df.NewCol .= "Test"

        me2 = update_row_indices(me, new_row_df)
        @test "NewCol" in names(me2.row_indices)
        @test me2.data === me.data # verify zero-copy behavior (reused array pointer)

        new_col_df = copy(col_df)
        new_col_df.NewCol .= "Test"

        me3 = update_col_indices(me, new_col_df)
        @test "NewCol" in names(me3.col_indices)
        @test me3.data === me.data # verify zero-copy

        se2 = update_col_indices(se, new_col_df)
        @test "NewCol" in names(se2.col_indices)
        @test se2.data === se.data

        lf2 = update_row_indices(lf, new_row_df)
        @test "NewCol" in names(lf2.row_indices)
        @test lf2.factorization === lf.factorization

        lf3 = update_col_indices(lf, new_col_df)
        @test "NewCol" in names(lf3.col_indices)
        @test lf3.factorization === lf.factorization
    end

    @testset "@filter_rows and @filter_cols macros" begin
        # MatrixEntry row filtering
        f_row = @filter_rows(me, Developed == true, GDP > 5000)
        @test size(f_row.data) == (1, 3)
        @test f_row.row_indices.Country == ["USA"]
        @test f_row.data == [10.0 20.0 30.0]

        # MatrixEntry column filtering
        f_col = @filter_cols(me, Type in ["Primary", "Tertiary"])
        @test size(f_col.data) == (3, 2)
        @test f_col.col_indices.Sector == ["Agr", "Ser"]

        # SeriesEntry filtering (Series only has columns)
        se_f = @filter_cols(se, Type == "Secondary")
        @test length(se_f) == 1
        @test se_f.col_indices.Sector == ["Man"]
        @test se_f.data == [2.5]
    end

    @testset "@mutate_rows and @mutate_cols macros" begin
        # Mutate rows
        me_m = @mutate_rows(me, GDP_Double = GDP * 2, Country_Lower = lowercase(Country))
        @test "GDP_Double" in names(me_m.row_indices)
        @test "Country_Lower" in names(me_m.row_indices)
        @test me_m.row_indices.GDP_Double == [40000, 28000, 8000]
        @test me_m.row_indices.Country_Lower == ["usa", "chn", "deu"]
        @test me_m.data === me.data # zero-copy check

        # Mutate columns
        me_mc = @mutate_cols(me, Sector_Upper = uppercase(Sector))
        @test "Sector_Upper" in names(me_mc.col_indices)
        @test me_mc.col_indices.Sector_Upper == ["AGR", "MAN", "SER"]

        # SeriesEntry mutate
        se_m = @mutate_cols(se, Sector_Upper = uppercase(Sector))
        @test "Sector_Upper" in names(se_m.col_indices)
        @test se_m.col_indices.Sector_Upper == ["AGR", "MAN", "SER"]
    end

    @testset "@select_rows and @select_cols macros" begin
        # Select rows
        me_s = @select_rows(me, Country, GDP)
        @test names(me_s.row_indices) == ["Country", "GDP"]
        @test size(me_s.data) == size(me.data)

        # Select columns
        me_sc = @select_cols(me, Sector)
        @test names(me_sc.col_indices) == ["Sector"]

        # SeriesEntry select
        se_s = @select_cols(se, Sector)
        @test names(se_s.col_indices) == ["Sector"]
    end

    @testset "@rename_rows and @rename_cols macros" begin
        # Rename rows
        me_r = @rename_rows(me, Nation = Country, Income = GDP)
        @test "Nation" in names(me_r.row_indices)
        @test "Income" in names(me_r.row_indices)
        @test !("Country" in names(me_r.row_indices))

        # Rename columns
        me_rc = @rename_cols(me, Industry = Sector)
        @test "Industry" in names(me_rc.col_indices)
        @test !("Sector" in names(me_rc.col_indices))

        # SeriesEntry rename
        se_r = @rename_cols(se, Industry = Sector)
        @test "Industry" in names(se_r.col_indices)
        @test !("Sector" in names(se_r.col_indices))
    end

    @testset "@slice_rows and @slice_cols macros" begin
        # Slice rows
        me_sl = @slice_rows(me, 1:2)
        @test size(me_sl.data) == (2, 3)
        @test me_sl.row_indices.Country == ["USA", "CHN"]

        # Slice columns
        me_slc = @slice_cols(me, 2:3)
        @test size(me_slc.data) == (3, 2)
        @test me_slc.col_indices.Sector == ["Man", "Ser"]
    end
end

@testset "Macros work without Tidier imports (H3)" begin
    # A bare module importing only Juliora and DataFrames: macro expansion
    # must not leak TidierData/nrow/copy/update_* into caller scope.
    mod = Module(:TidierHygieneProbe)
    Core.eval(mod, :(using Juliora, DataFrames))
    @test !isdefined(mod, :Tidier)
    @test !isdefined(mod, :TidierData)
    Core.eval(mod, :(h_data = [10.0 20.0 30.0; 40.0 50.0 60.0; 70.0 80.0 90.0]))
    Core.eval(mod, :(h_rows = DataFrame(Country = ["USA", "CHN", "DEU"], Developed = [true, false, true], GDP = [20000, 14000, 4000])))
    Core.eval(mod, :(h_cols = DataFrame(Sector = ["Agr", "Man", "Ser"], Type = ["Primary", "Secondary", "Tertiary"])))
    Core.eval(mod, :(h_me = MatrixEntry(h_data, h_cols, h_rows)))

    @test Core.eval(mod, :(size((@filter_rows(h_me, Developed == true)).data))) == (2, 3)
    @test Core.eval(mod, :((@filter_rows(h_me, Developed == true)).row_indices.Country)) == ["USA", "DEU"]
    @test Core.eval(mod, :(size((@filter_cols(h_me, Type == "Secondary")).data))) == (3, 1)
    @test Core.eval(mod, :(names((@mutate_rows(h_me, GDP_Double = GDP * 2)).row_indices))) ==
        ["Country", "Developed", "GDP", "GDP_Double"]
    @test Core.eval(mod, :(names((@mutate_cols(h_me, Sector_Upper = uppercase(Sector))).col_indices))) ==
        ["Sector", "Type", "Sector_Upper"]
    @test Core.eval(mod, :(names((@select_rows(h_me, Country)).row_indices))) == ["Country"]
    @test Core.eval(mod, :(names((@select_cols(h_me, Sector)).col_indices))) == ["Sector"]
    @test Core.eval(mod, :(names((@rename_rows(h_me, Nation = Country)).row_indices))) ==
        ["Nation", "Developed", "GDP"]
    @test Core.eval(mod, :(names((@rename_cols(h_me, Industry = Sector)).col_indices))) ==
        ["Industry", "Type"]
    @test Core.eval(mod, :(size((@slice_rows(h_me, 1:2)).data))) == (2, 3)
    @test Core.eval(mod, :(size((@slice_cols(h_me, 2:3)).data))) == (3, 2)

    # Caller-side `!!` interpolation still resolves in the caller module.
    Core.eval(mod, :(h_thr = 5000))
    @test Core.eval(mod, :((@filter_rows(h_me, GDP > !!h_thr)).row_indices.Country)) == ["USA", "CHN"]
end

@testset "Sentinel column collisions throw (M5)" begin
    cols = DataFrame(Sector = ["A", "B"])

    bad_rows = DataFrame(Country = ["USA", "CHN"], __row_id__ = [10, 20])
    bad = MatrixEntry([1.0 2.0; 3.0 4.0], cols, bad_rows)
    err = try
        @filter_rows(bad, Country == "USA")
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("__row_id__", err.msg)

    bad_cols = DataFrame(Sector = ["A", "B"], __col_id__ = [1, 2])
    bad2 = MatrixEntry([1.0 2.0; 3.0 4.0], bad_cols, DataFrame(Country = ["USA", "CHN"]))
    @test_throws ArgumentError @filter_cols(bad2, Sector == "A")
    @test_throws ArgumentError @slice_cols(bad2, 1)
    @test_throws ArgumentError @slice_rows(bad, 1)

    # Frames without the sentinel name are unaffected.
    good = MatrixEntry([1.0 2.0; 3.0 4.0], cols, DataFrame(Country = ["USA", "CHN"]))
    @test size((@filter_rows(good, Country == "USA")).data) == (1, 2)
end
