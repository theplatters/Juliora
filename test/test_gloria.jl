using Test
using Juliora
import Juliora.Parser as P
using SparseArrays
using ZipArchives
using Mmap
using Random

@testset "Parser Parser Rules" begin

    @testset "find_row_starts" begin
        # 1. Valid inputs/syntax
        bytes_valid = UInt8.(collect("1,2,3\n4,5,6\n7,8,9\n"))
        row_starts = @inferred P.find_row_starts(bytes_valid)
        @test row_starts == [1, 7, 13]

        # 2. Empty inputs
        @test (@inferred P.find_row_starts(UInt8[])) == Int[]

        # 3. Invalid/malformed inputs (e.g. carriage returns only or no newlines at all)
        bytes_cr = UInt8.(collect("1,2,3\r4,5,6\r7,8,9\r"))
        @test (@inferred P.find_row_starts(bytes_cr)) == [1]

        bytes_none = UInt8.(collect("no newlines here"))
        @test (@inferred P.find_row_starts(bytes_none)) == [1]

        # Non-ASCII and binary data
        bytes_binary = UInt8[0x00, 0x01, 0x0a, 0xff, 0x0a, 0x12]
        @test (@inferred P.find_row_starts(bytes_binary)) == [1, 4, 6]
    end

    @testset "custom_gloria_sparse_parser" begin
        # 1. Valid inputs/syntax
        csv_str = "1.0,0.0,3.0\n0.0,5.5,0.0\n7.0,0.0,9.9\n"
        bytes = UInt8.(collect(csv_str))

        # Test typical parse
        sp_mat = @inferred P.custom_gloria_sparse_parser(bytes, 3, 3)
        @test sp_mat isa SparseMatrixCSC{Float64, Int}
        @test size(sp_mat) == (3, 3)
        @test sp_mat[1, 1] == 1.0
        @test sp_mat[1, 2] == 0.0
        @test sp_mat[1, 3] == 3.0
        @test sp_mat[2, 2] == 5.5
        @test sp_mat[3, 1] == 7.0
        @test sp_mat[3, 3] == 9.9

        # Delimiter variants (carriage returns)
        csv_str_alt = "1.0,0.0,3.0\r\n0.0,5.5,0.0\r\n7.0,0.0,9.9\r\n"
        bytes_alt = Vector{UInt8}(csv_str_alt)
        sp_mat_alt = @inferred P.custom_gloria_sparse_parser(bytes_alt, 3, 3)
        @test sp_mat_alt == sp_mat

        # 2. Empty inputs
        empty_bytes = UInt8[]
        sp_empty = @inferred P.custom_gloria_sparse_parser(empty_bytes, 0, 0)
        @test size(sp_empty) == (0, 0)

        # 3. Invalid/malformed inputs
        # String tokens instead of numbers
        csv_malformed = "1.0,abc,3.0\nxyz,5.5,0.0\n7.0,0.0,def\n"
        bytes_malformed = UInt8.(collect(csv_malformed))
        sp_malformed = @inferred P.custom_gloria_sparse_parser(bytes_malformed, 3, 3)
        @test sp_malformed isa SparseMatrixCSC{Float64, Int}

        # Fewer columns than specified
        csv_fewer_cols = "1.0,2.0\n3.0\n"
        sp_fewer = @inferred P.custom_gloria_sparse_parser(UInt8.(collect(csv_fewer_cols)), 2, 3)
        @test size(sp_fewer) == (2, 3)
        @test sp_fewer[1, 1] == 1.0
        @test sp_fewer[1, 2] == 2.0

        # Negative dimensions (boundary check)
        @test_throws ArgumentError P.custom_gloria_sparse_parser(bytes, -1, 3)
        @test_throws ArgumentError P.custom_gloria_sparse_parser(bytes, 3, -1)
    end

    @testset "read_csv_to_sparse_matrix" begin
        mktempdir() do tmpdir
            zip_path = joinpath(tmpdir, "test.zip")

            # 1. Valid inputs/syntax
            ZipArchives.ZipWriter(zip_path) do w
                ZipArchives.zip_newfile(w, "valid.csv")
                write(w, "1.0,0.0,3.0\n0.0,5.5,0.0\n7.0,0.0,9.9\n")

                ZipArchives.zip_newfile(w, "empty.csv")
                write(w, "")

                ZipArchives.zip_newfile(w, "malformed.csv")
                write(w, "1.0,abc\n2.0,3.0,4.0\n")
            end

            # Read the zip archive into memory
            zip_bytes = read(zip_path)
            zip_reader = ZipArchives.ZipReader(zip_bytes)

            # Test valid file
            sp_valid = @inferred P.read_csv_to_sparse_matrix(zip_reader, "valid.csv")
            @test sp_valid isa SparseMatrixCSC{Float64, Int}
            @test size(sp_valid) == (3, 3)
            @test sp_valid[1, 3] == 3.0
            @test sp_valid[2, 2] == 5.5
            @test sp_valid[3, 3] == 9.9

            # 2. Empty input
            sp_empty = @inferred P.read_csv_to_sparse_matrix(zip_reader, "empty.csv")
            @test size(sp_empty) == (0, 0)
            @test sp_empty isa SparseMatrixCSC{Float64, Int}

            # 3. Invalid/malformed csv
            sp_malformed = @inferred P.read_csv_to_sparse_matrix(zip_reader, "malformed.csv")
            @test size(sp_malformed) == (2, 2)
            @test sp_malformed[2, 1] == 2.0

            # Non-existent file key
            @test_throws Exception P.read_csv_to_sparse_matrix(zip_reader, "nonexistent.csv")
        end
    end

    @testset "parse_gloria_sut" begin
        mktempdir() do tmpdir
            year = 2020
            version = 60
            price = P.BasePrice()
            ext = P.get_extention(price)

            zip_name = "GLORIA_MRIOs_$(version)_$(year).zip"
            zip_path = joinpath(tmpdir, zip_name)

            t_file = "20260121_120secMother_AllCountries_002_T-Results_$(year)_0$(version)_$(ext).csv"
            y_file = "20260121_120secMother_AllCountries_002_Y-Results_$(year)_0$(version)_$(ext).csv"
            va_file = "20260121_120secMother_AllCountries_002_V-Results_$(year)_0$(version)_$(ext).csv"

            # 1. Valid inputs/syntax
            ZipArchives.ZipWriter(zip_path) do w
                ZipArchives.zip_newfile(w, t_file)
                write(w, "1.0,0.0\n0.0,2.0\n")
                ZipArchives.zip_newfile(w, y_file)
                write(w, "3.0,0.0\n0.0,4.0\n")
                ZipArchives.zip_newfile(w, va_file)
                write(w, "5.0,0.0\n0.0,6.0\n")
            end

            sut = P.parse_gloria_sut(tmpdir, year; version = version, price = price)
            @test sut isa Dict{String, SparseMatrixCSC{Float64, Int}}
            @test haskey(sut, "T")
            @test haskey(sut, "Y")
            @test haskey(sut, "VA")
            @test sut["T"] == sparse([1.0 0.0; 0.0 2.0])
            @test sut["Y"] == sparse([3.0 0.0; 0.0 4.0])
            @test sut["VA"] == sparse([5.0 0.0; 0.0 6.0])

            # Test type stability via a helper
            function run_parse()
                P.parse_gloria_sut(tmpdir, year; version = version, price = price)
            end
            @test (@inferred run_parse()) isa Dict{String, SparseMatrixCSC{Float64, Int}}

            # 2. Empty inputs
            zip_name_empty = "GLORIA_MRIOs_$(version)_$(year + 1).zip"
            zip_path_empty = joinpath(tmpdir, zip_name_empty)
            t_file_empty = "20260121_120secMother_AllCountries_002_T-Results_$(year + 1)_0$(version)_$(ext).csv"
            y_file_empty = "20260121_120secMother_AllCountries_002_Y-Results_$(year + 1)_0$(version)_$(ext).csv"
            va_file_empty = "20260121_120secMother_AllCountries_002_V-Results_$(year + 1)_0$(version)_$(ext).csv"
            ZipArchives.ZipWriter(zip_path_empty) do w
                ZipArchives.zip_newfile(w, t_file_empty)
                write(w, "")
                ZipArchives.zip_newfile(w, y_file_empty)
                write(w, "")
                ZipArchives.zip_newfile(w, va_file_empty)
                write(w, "")
            end

            sut_empty = P.parse_gloria_sut(tmpdir, year + 1; version = version, price = price)
            @test sut_empty["T"] == dropzeros(sparse(Int[], Int[], Float64[], 0, 0))
            @test sut_empty["Y"] == dropzeros(sparse(Int[], Int[], Float64[], 0, 0))
            @test sut_empty["VA"] == dropzeros(sparse(Int[], Int[], Float64[], 0, 0))

            # 3. Invalid/malformed inputs (missing VA file entry)
            zip_name_missing = "GLORIA_MRIOs_$(version)_$(year + 2).zip"
            zip_path_missing = joinpath(tmpdir, zip_name_missing)
            t_file_missing = "20260121_120secMother_AllCountries_002_T-Results_$(year + 2)_0$(version)_$(ext).csv"
            y_file_missing = "20260121_120secMother_AllCountries_002_Y-Results_$(year + 2)_0$(version)_$(ext).csv"
            ZipArchives.ZipWriter(zip_path_missing) do w
                ZipArchives.zip_newfile(w, t_file_missing)
                write(w, "1.0,0.0\n")
                ZipArchives.zip_newfile(w, y_file_missing)
                write(w, "3.0,0.0\n")
            end
            @test_throws Exception P.parse_gloria_sut(tmpdir, year + 2; version = version, price = price)

            # 4. Unzipped Directory Parsing Tests
            # Setup unzipped directory directly
            unzipped_dir = joinpath(tmpdir, "GLORIA_MRIOs_$(version)_$(year)")
            mkpath(unzipped_dir)
            write(joinpath(unzipped_dir, t_file), "1.0,0.0\n0.0,2.0\n")
            write(joinpath(unzipped_dir, y_file), "3.0,0.0\n0.0,4.0\n")
            write(joinpath(unzipped_dir, va_file), "5.0,0.0\n0.0,6.0\n")

            # Test explicit is_unzipped = true with direct directory path
            sut_unzipped1 = P.parse_gloria_sut(unzipped_dir, year, true; version = version, price = price)
            @test sut_unzipped1 isa Dict{String, SparseMatrixCSC{Float64, Int}}
            @test sut_unzipped1["T"] == sparse([1.0 0.0; 0.0 2.0])
            @test sut_unzipped1["Y"] == sparse([3.0 0.0; 0.0 4.0])
            @test sut_unzipped1["VA"] == sparse([5.0 0.0; 0.0 6.0])

            # Test explicit is_unzipped = true with parent directory path
            sut_unzipped2 = P.parse_gloria_sut(tmpdir, year, true; version = version, price = price)
            @test sut_unzipped2 isa Dict{String, SparseMatrixCSC{Float64, Int}}
            @test sut_unzipped2["T"] == sparse([1.0 0.0; 0.0 2.0])

            # Test automatic detection (fallback) when zip file does not exist
            # Create a new year for testing fallback without a zip file
            year_auto = year + 4
            t_file_auto = "20260121_120secMother_AllCountries_002_T-Results_$(year_auto)_0$(version)_$(ext).csv"
            y_file_auto = "20260121_120secMother_AllCountries_002_Y-Results_$(year_auto)_0$(version)_$(ext).csv"
            va_file_auto = "20260121_120secMother_AllCountries_002_V-Results_$(year_auto)_0$(version)_$(ext).csv"

            unzipped_dir_auto = joinpath(tmpdir, "GLORIA_MRIOs_$(version)_$(year_auto)")
            mkpath(unzipped_dir_auto)
            write(joinpath(unzipped_dir_auto, t_file_auto), "1.0,0.0\n0.0,2.0\n")
            write(joinpath(unzipped_dir_auto, y_file_auto), "3.0,0.0\n0.0,4.0\n")
            write(joinpath(unzipped_dir_auto, va_file_auto), "5.0,0.0\n0.0,6.0\n")

            # Since no zip file exists for year_auto in tmpdir, it should detect and parse the unzipped directory
            sut_auto = P.parse_gloria_sut(tmpdir, year_auto; version = version, price = price)
            @test sut_auto isa Dict{String, SparseMatrixCSC{Float64, Int}}
            @test sut_auto["T"] == sparse([1.0 0.0; 0.0 2.0])
            @test sut_auto["Y"] == sparse([3.0 0.0; 0.0 4.0])
            @test sut_auto["VA"] == sparse([5.0 0.0; 0.0 6.0])

            # Test non-existent unzipped directory path throws exception
            @test_throws Exception P.parse_gloria_sut(joinpath(tmpdir, "does_not_exist"), year, true; version = version, price = price)
        end
    end

    @testset "__construct_IO" begin
        # 1. Valid inputs/syntax
        V = sparse([2.0 0.0 2.0; 0.0 4.0 0.0])
        U = sparse([1.0 0.0; 0.0 2.0; 3.0 4.0])
        data_sut = Dict("VA" => V, "T" => U)

        res = P._construct_IO(data_sut; construct = "B")
        @test res isa Dict{String, SparseMatrixCSC{Float64, Int}}
        @test haskey(res, "A")
        @test res["A"] == sparse([0.5 0.0 0.5; 0.0 2.0 0.0; 1.5 4.0 1.5])

        res = P._construct_IO(data_sut; construct = "B")
        @test res isa Dict{String, SparseMatrixCSC{Float64, Int}}

        # 2. Empty inputs
        data_empty = Dict(
            "VA" => dropzeros(sparse(Int[], Int[], Float64[], 0, 0)),
            "T" => dropzeros(sparse(Int[], Int[], Float64[], 0, 0))
        )
        res_empty = P._construct_IO(data_empty; construct = "B")
        @test res_empty isa Dict{String, SparseMatrixCSC{Float64, Int}}
        @test size(res_empty["A"]) == (0, 0)

        # 3. Invalid/malformed inputs
        data_missing = Dict("VA" => V)
        @test_throws KeyError P._construct_IO(data_missing; construct = "B")

        V_bad = sparse(rand(4, 3))
        U_bad = sparse(rand(3, 3))
        data_bad_dim = Dict("VA" => V_bad, "T" => U_bad)
        @test_throws DimensionMismatch P.__construct_IO(data_bad_dim; construct = "B")
    end
end

@testset "Differential & Thread Safety Testing" begin
    # Compare single-threaded vs multithreaded parsing outputs
    nrows = 100
    ncols = 100
    Random.seed!(1234)

    dense_mat = zeros(nrows, ncols)
    for i in 1:nrows, j in 1:ncols
        if rand() < 0.15
            dense_mat[i, j] = round(rand() * 100, digits = 2)
        end
    end

    csv_io = IOBuffer()
    for i in 1:nrows
        join(csv_io, dense_mat[i, :], ",")
        write(csv_io, "\n")
    end
    csv_bytes = take!(csv_io)

    # Sequential (single-threaded) run
    st_result = P.custom_gloria_sparse_parser(csv_bytes, nrows, ncols)

    # Multithreaded runs
    ntasks = 8
    tasks = [Threads.@spawn P.custom_gloria_sparse_parser(csv_bytes, nrows, ncols) for _ in 1:ntasks]
    mt_results = fetch.(tasks)

    for res in mt_results
        @test res == st_result
    end

    # ZipReader concurrent read safety
    mktempdir() do tmpdir
        zip_path = joinpath(tmpdir, "threads.zip")
        ZipArchives.ZipWriter(zip_path) do w
            ZipArchives.zip_newfile(w, "data.csv")
            write(w, "1.0,2.0,3.0\n4.0,5.0,6.0\n")
        end
        zip_bytes = read(zip_path)
        zip_reader = ZipArchives.ZipReader(zip_bytes)

        st_res = P.read_csv_to_sparse_matrix(zip_reader, "data.csv")

        tasks_zip = [Threads.@spawn P.read_csv_to_sparse_matrix(zip_reader, "data.csv") for _ in 1:8]
        mt_zip_results = fetch.(tasks_zip)

        for res in mt_zip_results
            @test res == st_res
        end
    end
end

@testset "Stress & Memory Constraint Testing" begin
    # 1. Stress test with large sparse dimensions
    nrows = 5000
    ncols = 5000
    line = "0.0,0.0,1.5,0.0,0.0\n"
    csv_data = Vector{UInt8}(repeat(line, 5000))

    sp = P.custom_gloria_sparse_parser(csv_data, 5000, 5)
    @test size(sp) == (5000, 5)
    @test nnz(sp) == 5000
    @test all(sp[:, 3] .== 1.5)

    # 2. Verify no stack overflow with massive token stream in a single line
    large_line_io = IOBuffer()
    for j in 1:10000
        print(large_line_io, Float64(j))
        if j < 10000
            print(large_line_io, ",")
        end
    end
    print(large_line_io, "\n")
    large_line_bytes = take!(large_line_io)

    sp_long = P.custom_gloria_sparse_parser(large_line_bytes, 1, 10000)
    @test size(sp_long) == (1, 10000)
    @test nnz(sp_long) == 10000
    @test sp_long[1, 5000] == 5000.0

    # 3. Malformed/large coordinate arrays checking behavior
    sp_oversized = P.custom_gloria_sparse_parser(UInt8.(collect("1,2\n3,4\n")), 1000, 1000)
    @test size(sp_oversized) == (1000, 1000)
    @test sp_oversized[1, 1] == 1.0
    @test sp_oversized[1, 2] == 2.0
    @test sp_oversized[1, 3] == 3.0
    @test sp_oversized[1, 4] == 4.0
    @test sp_oversized[2, 1] == 3.0
    @test sp_oversized[2, 2] == 4.0
end

@testset "parse_gloria integration" begin
    path = "data/GLORIA/2019"
    year = 2019
    version = 60

    mrio = P.parse_gloria(path, year; version = version)

    @test mrio isa IO.MRIO
    @test mrio.A isa IO.MatrixEntry
    @test mrio.T isa IO.MatrixEntry
    @test mrio.VA isa IO.MatrixEntry
    @test mrio.FD isa IO.MatrixEntry
    @test mrio.L isa IO.LeontiefFactorization
    @test mrio.X isa IO.SeriesEntry
    @test mrio.env isa IO.EnvironmentalExtension

    # Verify dimensions for 164 regions, 120 sectors, 6 fd categories, 7 va categories
    # The intermediate matrices after constructing symmetric IOT should be (164 * 120) x (164 * 120) = 19680 x 19680
    @test size(mrio.T.data) == (19680, 19680)
    @test size(mrio.A.data) == (19680, 19680)
    @test size(mrio.FD.data) == (19680, 984) # 164 * 6 = 984 final demand columns
    @test size(mrio.VA.data) == (1148, 19680) # 164 * 7 = 1148 value added rows
    @test length(mrio.X.data) == 19680

    # Verify EnvironmentalExtension has empty matrices as requested by rule 4
    @test size(mrio.env.F.data) == (0, 19680)
    @test size(mrio.env.A.data) == (0, 19680)
end
