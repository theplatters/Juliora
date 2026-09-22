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

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)

    @test size(matrix_entry.data) == (3, 3)
    @test matrix_entry.data == data
    @test nrow(matrix_entry.row_indices) == 3
    @test nrow(matrix_entry.col_indices) == 3
    @test length(matrix_entry.row_lookup) == 3
    @test length(matrix_entry.col_lookup) == 3

    # Test dimension mismatch error
    wrong_data = [1.0 2.0; 3.0 4.0]  # 2x2 but indices are 3x3
    @test_throws DimensionMismatch IO.MatrixEntry(wrong_data, col_df, row_df)

    # Test lookup dictionary functionality
    usa_agr_key = (Country = "USA", Sector = "Agriculture")
    @test haskey(matrix_entry.row_lookup, usa_agr_key)
    @test matrix_entry.row_lookup[usa_agr_key] == 1

    chn_man_key = (Country = "CHN", Sector = "Manufacturing")
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

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)

    # Test valid indexing
    @test matrix_entry[(Country = "USA", Sector = "Agr"), (Country = "USA", Sector = "Agr")] == 1.0
    @test matrix_entry[(Country = "CHN", Sector = "Man"), (Country = "CHN", Sector = "Man")] == 4.0
    @test matrix_entry[(Country = "DEU", Sector = "Ser"), (Country = "CHN", Sector = "Man")] == 6.0

    # Test invalid row key
    @test_throws BoundsError matrix_entry[(Country = "JPN", Sector = "Agr"), (Country = "USA", Sector = "Agr")]

    # Test invalid column key
    @test_throws BoundsError matrix_entry[(Country = "USA", Sector = "Agr"), (Country = "JPN", Sector = "Agr")]

    # Test missing sector
    @test_throws BoundsError matrix_entry[(Country = "USA", Sector = "Tech"), (Country = "USA", Sector = "Agr")]
end

@testset "MatrixEntry Edge Cases" begin
    # Test with single row/column
    single_data = reshape([42.0], 1, 1)
    single_row = DataFrame(Country = ["USA"], Sector = ["Total"])
    single_col = DataFrame(Country = ["USA"], Sector = ["Total"])

    single_matrix = IO.MatrixEntry(single_data, single_col, single_row)
    @test single_matrix[(Country = "USA", Sector = "Total"), (Country = "USA", Sector = "Total")] == 42.0

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

    mixed_matrix = IO.MatrixEntry(mixed_data, mixed_col, mixed_row)
    @test mixed_matrix[(ID = 1, Name = "A", Active = true), (ID = 10, Type = "X")] == 1.0
    @test mixed_matrix[(ID = 2, Name = "B", Active = false), (ID = 20, Type = "Y")] == 4.0
end

@testset "MatrixEntry Type Stability" begin
    data = [1.0 2.0; 3.0 4.0]
    row_df = DataFrame(A = [1, 2], B = ["X", "Y"])
    col_df = DataFrame(C = [10, 20], D = ["P", "Q"])

    matrix_entry = IO.MatrixEntry(data, col_df, row_df)

    # Test that returned values are Float64
    val = matrix_entry[(A = 1, B = "X"), (C = 10, D = "P")]
    @test val isa Float64
    @test val == 1.0

    # Test that lookup dictionaries have correct types
    @test matrix_entry.row_lookup isa Dict{NamedTuple, Int}
    @test matrix_entry.col_lookup isa Dict{NamedTuple, Int}
end

@testset "drop! does not corrupt parents or siblings (H1)" begin
    data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    row_df = DataFrame(Country = ["USA", "CHN", "DEU"], Sector = ["Agr", "Man", "Ser"])
    col_df = DataFrame(Country = ["USA", "CHN", "DEU"], Sector = ["Agr", "Man", "Ser"])
    parent = IO.MatrixEntry(data, col_df, row_df)
    parent_col_lookup = copy(parent.col_lookup)
    parent_row_lookup = copy(parent.row_lookup)

    # Derive a child via a Bool mask: index frames are shared by reference.
    child = parent[[true, true, false], :]
    @test child.col_indices === parent.col_indices

    # Dropping a column on the child must not touch the parent.
    drop!(child, (Country = "USA", Sector = "Agr"); dims = 2)
    @test size(parent.data) == (3, 3)
    @test parent.data == data
    @test nrow(parent.col_indices) == 3
    @test parent.col_lookup == parent_col_lookup
    @test parent.row_lookup == parent_row_lookup
    @test parent[(Country = "USA", Sector = "Agr"), (Country = "USA", Sector = "Agr")] == 1.0
    @test parent[(Country = "CHN", Sector = "Man"), (Country = "CHN", Sector = "Man")] == 5.0

    # The child itself is consistent: replaced frames, rebuilt lookups.
    @test size(child.data) == (2, 2)
    @test child.data == [2.0 3.0; 5.0 6.0]
    @test nrow(child.col_indices) == 2
    @test child.col_indices !== parent.col_indices
    @test size(child.data) == (nrow(child.row_indices), nrow(child.col_indices))
    @test child[(Country = "USA", Sector = "Agr"), (Country = "CHN", Sector = "Man")] == 2.0
    @test_throws BoundsError child[(Country = "USA", Sector = "Agr"), (Country = "USA", Sector = "Agr")]

    # Same guarantee along rows.
    child2 = parent[:, [true, false, true]]
    @test child2.row_indices === parent.row_indices
    drop!(child2, (Country = "CHN",); dims = 1)
    @test size(parent.data) == (3, 3)
    @test nrow(parent.row_indices) == 3
    @test parent.row_lookup == parent_row_lookup
    @test size(child2.data) == (2, 2)
    @test child2.row_indices.Country == ["USA", "DEU"]
end

@testset "make_named_tuple validation (H2)" begin
    @test IO.make_named_tuple("Country", "USA") == (Country = "USA",)
    @test IO.make_named_tuple(["A", "B"], [1, 2]) == (A = 1, B = 2)
    @test IO.make_named_tuple("K", [1]) == (K = 1,)

    # Scalar keys broadcast into every key collapsed data: must throw instead.
    err = try
        IO.make_named_tuple("Country", ["USA", "CHN"])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("1", err.msg) && occursin("2", err.msg)

    err2 = try
        IO.make_named_tuple(["A", "B", "C"], [1, 2])
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("3", err2.msg) && occursin("2", err2.msg)

    # Vector form: outer lists must match, and each element is validated.
    @test IO.make_named_tuple_vector([["A"], ["B"]], [[1], [2]]) == [(A = 1,), (B = 2,)]
    @test_throws ArgumentError IO.make_named_tuple_vector([["A"]], [[1], [2]])
    @test_throws ArgumentError IO.make_named_tuple_vector([["A", "B"]], [[1]])
end

@testset "duplicate labels warn and keep last-wins lookups (M1)" begin
    d = [1.0 2.0; 3.0 4.0]
    dup_cols = DataFrame(K = ["a", "a"])
    rows = DataFrame(R = ["x", "y"])
    @test_logs (:warn,) IO.MatrixEntry(d, dup_cols, rows)
    dup_entry = @test_logs (:warn,) IO.MatrixEntry(d, dup_cols, rows)
    @test dup_entry.col_lookup[(K = "a",)] == 2

    dup_rows = DataFrame(R = ["x", "x"])
    ok_cols = DataFrame(K = ["a", "b"])
    @test_logs (:warn,) IO.MatrixEntry(d, ok_cols, dup_rows)

    # Unique labels stay silent.
    @test_logs IO.MatrixEntry(reshape([1.0], 1, 1), DataFrame(K = ["a"]), DataFrame(R = ["x"]))
end

@testset "missing/NaN partial-key lookups (M2)" begin
    data = [1.0 2.0; 3.0 4.0]
    col_df = DataFrame(S = ["a", "b"])
    row_df = DataFrame(C = ["USA", "CHN"], P = [330, missing])
    m = IO.MatrixEntry(data, col_df, row_df)

    # Exact lookups still work with missing values.
    @test m[(C = "CHN", P = missing), (S = "b",)] == 4.0

    # Partial lookups match missing metadata instead of erroring.
    sub = m[(P = missing,), :]
    @test sub isa SeriesEntry
    @test sub.data == [3.0, 4.0]
    sub_usa = m[(C = "USA",), :]
    @test sub_usa isa SeriesEntry
    @test sub_usa.data == [1.0, 2.0]

    # NaN metadata matches itself (isequal semantics).
    row_nan = DataFrame(C = ["USA", "CHN"], V = [1.0, NaN])
    mn = IO.MatrixEntry(data, col_df, row_nan)
    nan_sub = mn[(V = NaN,), :]
    @test nan_sub isa SeriesEntry
    @test nan_sub.data == [3.0, 4.0]
    @test mn[(C = "USA", V = 1.0), (S = "a",)] == 1.0

    # drop/drop! use the same matching, including missing keys.
    d = drop(m, (P = missing,); dims = 1)
    @test size(d.data) == (1, 2)
    @test d.row_indices.C == ["USA"]
    dm = IO.MatrixEntry(data, col_df, row_df)
    drop!(dm, (P = missing,); dims = 1)
    @test size(dm.data) == (1, 2)
    @test dm.row_indices.C == ["USA"]
end

@testset "invalid dims raise ArgumentError (M3)" begin
    m = IO.MatrixEntry([1.0 2.0; 3.0 4.0], DataFrame(S = ["a", "b"]), DataFrame(C = ["x", "y"]))
    @test_throws ArgumentError drop(m, (C = "x",); dims = 0)
    @test_throws ArgumentError drop(m, (C = "x",); dims = 3)
    @test_throws ArgumentError drop(m, [(C = "x",)]; dims = -1)
    @test_throws ArgumentError drop!(m, (C = "x",); dims = 3)
    @test_throws ArgumentError drop!(m, [(C = "x",)]; dims = 0)
    @test_throws ArgumentError filter(_ -> true, m; dims = 3)
    @test_throws ArgumentError filter(_ -> true, m; dims = 0)
    # Failed validation leaves the entry untouched.
    @test size(m.data) == (2, 2)
    @test nrow(m.row_indices) == 2
end

@testset "drop/drop! on unmatched keys (M4)" begin
    data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    row_df = DataFrame(Country = ["USA", "CHN", "DEU"], Sector = ["Agr", "Man", "Ser"])
    col_df = DataFrame(Country = ["USA", "CHN", "DEU"], Sector = ["Agr", "Man", "Ser"])
    m = IO.MatrixEntry(data, col_df, row_df)

    # Zero matches throw instead of silently no-op'ing.
    @test_throws BoundsError drop(m, (Country = "JPN", Sector = "X"); dims = 1)
    @test_throws BoundsError drop(m, (Country = "JPN",); dims = 2)
    @test_throws BoundsError drop(m, [(Country = "JPN",)]; dims = 1)
    @test_throws BoundsError drop!(m, (Country = "JPN",); dims = 1)
    @test_throws BoundsError drop!(m, [(Country = "JPN",)]; dims = 2)
    # Failed drops leave the entry untouched.
    @test size(m.data) == (3, 3)

    # Partial vector matches warn and drop the matched keys.
    kept = @test_logs (:warn,) drop(m, [(Country = "USA", Sector = "Agr"), (Country = "ZZZ",)]; dims = 1)
    @test size(kept.data) == (2, 3)
    @test kept.row_indices.Country == ["CHN", "DEU"]

    mut = IO.MatrixEntry(data, col_df, row_df)
    @test_logs (:warn,) drop!(mut, [(Country = "USA", Sector = "Agr"), (Country = "ZZZ",)]; dims = 1)
    @test size(mut.data) == (2, 3)
    @test mut.row_indices.Country == ["CHN", "DEU"]

    # Fully matched vector drops stay silent.
    @test_logs drop(m, [(Country = "USA", Sector = "Agr")]; dims = 1)
end

@testset "aggregate over zero groups returns empty entries (LOW)" begin
    data = [1.0 2.0; 3.0 4.0]
    row_df = DataFrame(C = ["USA", "CHN"], G = ["x", "y"])
    col_df = DataFrame(S = ["a", "b"])
    m = IO.MatrixEntry(data, col_df, row_df)

    empty_rows = filter_rows(m, _ -> false)
    @test size(empty_rows.data) == (0, 2)
    agg = aggregate(IO.groupby(empty_rows, :G; dims = 1), sum)
    @test agg isa IO.MatrixEntry
    @test size(agg.data) == (0, 2)
    @test nrow(agg.row_indices) == 0
    @test nrow(agg.col_indices) == 2
    @test size(agg.data) == (nrow(agg.row_indices), nrow(agg.col_indices))

    empty_cols = filter_cols(m, _ -> false)
    agg2 = aggregate(IO.groupby(empty_cols, :S; dims = 2), sum)
    @test agg2 isa IO.MatrixEntry
    @test size(agg2.data) == (2, 0)
    @test nrow(agg2.col_indices) == 0
    @test nrow(agg2.row_indices) == 2
end
