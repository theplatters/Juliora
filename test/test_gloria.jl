using Test
using Juliora
import Juliora.Parser as P
using ZipArchives
using Mmap
using Random
using DataFrames
using JLD2
using XLSX

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

    @testset "parse TFile" begin
        # 1. Valid inputs/syntax
        csv_str = "1.0,2.0\n3.0,4.0\n"
        bytes = UInt8.(collect(csv_str))

        S, U = @inferred P.parse(P.TFile(), bytes; n_regions = 1, n_sectors = 1)
        @test S isa Matrix{Float64}
        @test U isa Matrix{Float64}
        @test size(S) == (1, 1)
        @test size(U) == (1, 1)
        @test S[1, 1] == 2.0
        @test U[1, 1] == 3.0

        # Delimiter variants (carriage returns, semicolons)
        csv_str_alt = "1.0;2.0\r\n3.0;4.0\r\n"
        bytes_alt = Vector{UInt8}(csv_str_alt)
        S_alt, U_alt = @inferred P.parse(P.TFile(), bytes_alt; delim = ';', n_regions = 1, n_sectors = 1)
        @test S_alt == S
        @test U_alt == U

        # 2. Empty inputs
        empty_bytes = UInt8[]
        S_empty, U_empty = @inferred P.parse(P.TFile(), empty_bytes; n_regions = 0, n_sectors = 0)
        @test size(S_empty) == (0, 0)
        @test size(U_empty) == (0, 0)

        # 3. Invalid/malformed inputs: unparseable tokens default to 0.0 and
        # consume the whole field (no mid-field re-scan phantoms)
        csv_malformed = "1.0,abc\nxyz,4.0\n"
        bytes_malformed = UInt8.(collect(csv_malformed))
        S_m, U_m = @inferred P.parse(P.TFile(), bytes_malformed; n_regions = 1, n_sectors = 1)
        @test S_m[1, 1] == 0.0 # skipped, default 0
        @test U_m[1, 1] == 0.0 # skipped, default 0

        # Rows are strictly bounded: short rows throw instead of wrapping
        # into the next row
        csv_fewer = "1.0\n3.0,4.0\n"
        @test_throws P.ParserError P.parse(P.TFile(), UInt8.(collect(csv_fewer)); n_regions = 1, n_sectors = 1)

        # ... as do rows with extra trailing fields
        csv_extra = "1.0,2.0,3.0\n3.0,4.0,5.0\n"
        @test_throws P.ParserError P.parse(P.TFile(), UInt8.(collect(csv_extra)); n_regions = 1, n_sectors = 1)

        # Negative dimensions (boundary check)
        @test_throws ArgumentError P.parse(P.TFile(), bytes; n_regions = -1, n_sectors = 1)
        @test_throws ArgumentError P.parse(P.TFile(), bytes; n_regions = 1, n_sectors = -1)
    end

    @testset "parse VAFile" begin
        # 1. Valid inputs/syntax
        csv_str = "6.0,0.0\n7.0,0.0\n"
        bytes = UInt8.(collect(csv_str))

        VA = @inferred P.parse(P.VAFile(), bytes; n_regions = 1, n_sectors = 1)
        @test VA isa Matrix{Float64}
        @test size(VA) == (2, 1)
        @test VA[1, 1] == 6.0
        @test VA[2, 1] == 7.0

        # Delimiter variants
        csv_str_alt = "6.0;0.0\r\n7.0;0.0\r\n"
        bytes_alt = Vector{UInt8}(csv_str_alt)
        VA_alt = @inferred P.parse(P.VAFile(), bytes_alt; delim = ';', n_regions = 1, n_sectors = 1)
        @test VA_alt == VA

        # 2. Empty inputs
        empty_bytes = UInt8[]
        VA_empty = @inferred P.parse(P.VAFile(), empty_bytes; n_regions = 0, n_sectors = 0)
        @test size(VA_empty) == (0, 0)

        # 3. Invalid/malformed inputs
        csv_malformed = "abc,0.0\n7.0,xyz\n"
        bytes_malformed = UInt8.(collect(csv_malformed))
        VA_m = @inferred P.parse(P.VAFile(), bytes_malformed; n_regions = 1, n_sectors = 1)
        @test VA_m[1, 1] == 0.0
        @test VA_m[2, 1] == 7.0

        # Empty fields parse as 0.0 without shifting later columns
        VA_e = P.parse(P.VAFile(), UInt8.(collect("6.0,\n,0.0\n")); n_regions = 1, n_sectors = 1)
        @test VA_e == [6.0; 0.0;;]

        # A token with garbage around digits ("x5") must not fabricate a
        # phantom value: the whole field defaults to 0.0
        VA_x = P.parse(P.VAFile(), UInt8.(collect("x5,0.0\n7.0,0.0\n")); n_regions = 1, n_sectors = 1)
        @test VA_x == [0.0; 7.0;;]

        # Short and long rows throw
        @test_throws P.ParserError P.parse(P.VAFile(), UInt8.(collect("6.0\n7.0,0.0\n")); n_regions = 1, n_sectors = 1)
        @test_throws P.ParserError P.parse(P.VAFile(), UInt8.(collect("6.0,0.0,1.0\n7.0,0.0,1.0\n")); n_regions = 1, n_sectors = 1)

        # Negative dimensions
        @test_throws ArgumentError P.parse(P.VAFile(), bytes; n_regions = -1, n_sectors = 1)
        @test_throws ArgumentError P.parse(P.VAFile(), bytes; n_regions = 1, n_sectors = -1)
    end

    @testset "parse YFile" begin
        # 1. Valid inputs/syntax
        csv_str = "0.0,0.0,0.0,0.0,0.0,0.0\n5.0,5.0,5.0,5.0,5.0,5.0\n"
        bytes = UInt8.(collect(csv_str))

        Y = @inferred P.parse(P.YFile(), bytes; n_regions = 1, n_sectors = 1)
        @test Y isa Matrix{Float64}
        @test size(Y) == (1, 6)
        @test all(Y[1, :] .== 5.0)

        # Delimiter variants
        csv_str_alt = "0.0;0.0;0.0;0.0;0.0;0.0\r\n5.0;5.0;5.0;5.0;5.0;5.0\r\n"
        bytes_alt = Vector{UInt8}(csv_str_alt)
        Y_alt = P.parse(P.YFile(), bytes_alt; delim = ';', n_regions = 1, n_sectors = 1)
        @test Y_alt == Y

        # 2. Empty inputs
        empty_bytes = UInt8[]
        Y_empty = P.parse(P.YFile(), empty_bytes; n_regions = 0, n_sectors = 0)
        @test size(Y_empty) == (0, 0)

        # 3. Invalid/malformed inputs
        csv_malformed = "0.0,0.0,0.0,0.0,0.0,0.0\n5.0,abc,5.0,5.0,5.0,5.0\n"
        bytes_malformed = UInt8.(collect(csv_malformed))
        Y_m = P.parse(P.YFile(), bytes_malformed; n_regions = 1, n_sectors = 1)
        @test Y_m[1, 1] == 5.0
        @test Y_m[1, 2] == 0.0 # skipped, defaults to 0

        # Empty mid-row field -> 0.0 for that column only
        Y_e = P.parse(P.YFile(), UInt8.(collect("0.0,0.0,0.0\n5.0,,5.0\n")); n_regions = 1, n_sectors = 1)
        @test Y_e == [5.0 0.0 5.0]

        # Row count must match 2 * n_regions * n_sectors exactly
        @test_throws P.ParserError P.parse(P.YFile(), UInt8.(collect("0.0,0.0\n")); n_regions = 1, n_sectors = 1)

        # Negative dimensions
        @test_throws ArgumentError P.parse(P.YFile(), bytes; n_regions = -1, n_sectors = 1)
        @test_throws ArgumentError P.parse(P.YFile(), bytes; n_regions = 1, n_sectors = -1)
    end

    @testset "BOM handling" begin
        bom = UInt8[0xef, 0xbb, 0xbf]
        plain_va = UInt8.(collect("6.0,0.0\n7.0,0.0\n"))
        @test P.parse(P.VAFile(), vcat(bom, plain_va); n_regions = 1, n_sectors = 1) ==
            P.parse(P.VAFile(), plain_va; n_regions = 1, n_sectors = 1) == [6.0; 7.0;;]

        S_bom, U_bom = P.parse(P.TFile(), vcat(bom, UInt8.(collect("1.0,2.0\n3.0,4.0\n"))); n_regions = 1, n_sectors = 1)
        @test S_bom == [2.0;;]
        @test U_bom == [3.0;;]
    end

    @testset "dimension validation" begin
        # T row count mismatch names the file
        e_t = try
            P.parse(P.TFile(), UInt8.(collect("1.0,2.0\n")); n_regions = 1, n_sectors = 1, fname = "T66")
            nothing
        catch err
            err
        end
        @test e_t isa P.ParserError
        @test occursin("T66", e_t.msg)
        @test occursin("expected 2 rows", e_t.msg)

        # T rows with more columns than expected throw as well
        @test_throws P.ParserError P.parse(P.TFile(), UInt8.(collect("1.0,2.0,3.0\n4.0,5.0,6.0\n")); n_regions = 1, n_sectors = 1)

        # VA column count mismatch names the file
        e_va = try
            P.parse(P.VAFile(), UInt8.(collect("1.0,2.0,3.0\n4.0,5.0,6.0\n")); n_regions = 1, n_sectors = 1, fname = "V66")
            nothing
        catch err
            err
        end
        @test e_va isa P.ParserError
        @test occursin("V66", e_va.msg)

        # Y row count mismatch names the file
        e_y = try
            P.parse(P.YFile(), UInt8.(collect("1.0,2.0\n")); n_regions = 1, n_sectors = 1, fname = "Y66")
            nothing
        catch err
            err
        end
        @test e_y isa P.ParserError
        @test occursin("Y66", e_y.msg)

        # Q column count mismatch names the file
        @test_throws P.ParserError P.parse(P.QFile(), UInt8.(collect("1.0\n2.0\n")); n_regions = 1, n_sectors = 1)
    end

    @testset "satellite suffix honors price" begin
        @test P.gloria_satellite_suffix("T", 2019, 60) ==
            "_120secMother_AllCountries_002_TQ-Results_2019_060_Markup001(full).csv"
        @test P.gloria_satellite_suffix("T", 2019, 60, P.BasePrice()) == P.gloria_satellite_suffix("T", 2019, 60)
        pp = P.gloria_satellite_suffix("T", 2019, 60, P.PPrice())
        @test pp == "_120secMother_AllCountries_002_TQ-Results_2019_060_Markup005(full).csv"
        @test occursin(P.get_extension(P.PPrice()), pp)
    end

    @testset "Type Stability" begin
        bytes_t = UInt8.(collect("1.0,2.0\n3.0,4.0\n"))
        @test (@inferred P.parse(P.TFile(), bytes_t; n_regions = 1, n_sectors = 1)) isa Tuple{Matrix{Float64}, Matrix{Float64}}

        bytes_va = UInt8.(collect("1.0,0.0\n2.0,0.0\n"))
        @test (@inferred P.parse(P.VAFile(), bytes_va; n_regions = 1, n_sectors = 1)) isa Matrix{Float64}

        bytes_y = UInt8.(collect("0.0,0.0\n5.0,5.0\n"))
        @test (@inferred P.parse(P.YFile(), bytes_y; n_regions = 1, n_sectors = 1)) isa Matrix{Float64}
    end

    @testset "parse_gloria_sut" begin
        mktempdir() do tmpdir
            year = 2020
            version = 60
            price = P.BasePrice()
            ext = P.get_extension(price)

            zip_name = "GLORIA_MRIOs_$(version)_$(year).zip"
            zip_path = joinpath(tmpdir, zip_name)

            t_file = "20260121_120secMother_AllCountries_002_T-Results_$(year)_0$(version)_$(ext).csv"
            y_file = "20260121_120secMother_AllCountries_002_Y-Results_$(year)_0$(version)_$(ext).csv"
            va_file = "20260121_120secMother_AllCountries_002_V-Results_$(year)_0$(version)_$(ext).csv"

            # 1. Valid inputs/syntax
            ZipArchives.ZipWriter(zip_path) do w
                ZipArchives.zip_newfile(w, t_file)
                write(w, "1.0,2.0\n3.0,4.0\n")
                ZipArchives.zip_newfile(w, y_file)
                write(w, "0.0,0.0,0.0,0.0,0.0,0.0\n5.0,5.0,5.0,5.0,5.0,5.0\n")
                ZipArchives.zip_newfile(w, va_file)
                write(w, "6.0,0.0\n7.0,0.0\n")
            end

            sut = P.parse_gloria_sut(tmpdir, year; version = version, price = price)
            @test P.parse_gloria_sut(zip_path, year; version = version, price = price) == sut
            @test sut isa Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
            V, U, Y, VA = sut
            @test V == [2.0;;]
            @test U == [3.0;;]
            @test Y == [5.0 5.0 5.0 5.0 5.0 5.0]
            @test VA == [6.0; 7.0;;]

            # A direct ZIP remains authoritative when an extracted sibling has
            # conflicting data.
            conflicting_dir = joinpath(tmpdir, "GLORIA_MRIOs_$(version)_$(year)")
            mkpath(conflicting_dir)
            write(joinpath(conflicting_dir, t_file), "9.0,9.0\n9.0,9.0\n")
            write(joinpath(conflicting_dir, y_file), "0.0,0.0,0.0,0.0,0.0,0.0\n9.0,9.0,9.0,9.0,9.0,9.0\n")
            write(joinpath(conflicting_dir, va_file), "9.0,0.0\n9.0,0.0\n")
            @test P.parse_gloria_sut(zip_path, year; version = version, price = price)[1] == [2.0;;]

            # Test type stability via a helper
            function run_parse()
                P.parse_gloria_sut(tmpdir, year; version = version, price = price)
            end
            @test (@inferred run_parse()) isa Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}

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
            @test sut_empty isa Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
            V_e, U_e, Y_e, VA_e = sut_empty
            @test size(V_e) == (0, 0)
            @test size(U_e) == (0, 0)
            @test size(Y_e) == (0, 0)
            @test size(VA_e) == (0, 0)

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
            write(joinpath(unzipped_dir, t_file), "1.0,2.0\n3.0,4.0\n")
            write(joinpath(unzipped_dir, y_file), "0.0,0.0,0.0,0.0,0.0,0.0\n5.0,5.0,5.0,5.0,5.0,5.0\n")
            write(joinpath(unzipped_dir, va_file), "6.0,0.0\n7.0,0.0\n")

            # Test explicit is_unzipped = true with direct directory path
            sut_unzipped1 = P.parse_gloria_sut(unzipped_dir, year, true; version = version, price = price)
            @test sut_unzipped1 isa Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
            V_uz1, U_uz1, Y_uz1, VA_uz1 = sut_unzipped1
            @test V_uz1 == [2.0;;]
            @test U_uz1 == [3.0;;]

            # Test explicit is_unzipped = true with parent directory path
            sut_unzipped2 = P.parse_gloria_sut(tmpdir, year, true; version = version, price = price)
            @test sut_unzipped2 isa Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
            V_uz2, U_uz2, Y_uz2, VA_uz2 = sut_unzipped2
            @test V_uz2 == [2.0;;]

            # Test automatic detection (fallback) when zip file does not exist
            # Create a new year for testing fallback without a zip file
            year_auto = year + 4
            t_file_auto = "20260121_120secMother_AllCountries_002_T-Results_$(year_auto)_0$(version)_$(ext).csv"
            y_file_auto = "20260121_120secMother_AllCountries_002_Y-Results_$(year_auto)_0$(version)_$(ext).csv"
            va_file_auto = "20260121_120secMother_AllCountries_002_V-Results_$(year_auto)_0$(version)_$(ext).csv"

            unzipped_dir_auto = joinpath(tmpdir, "GLORIA_MRIOs_$(version)_$(year_auto)")
            mkpath(unzipped_dir_auto)
            write(joinpath(unzipped_dir_auto, t_file_auto), "1.0,2.0\n3.0,4.0\n")
            write(joinpath(unzipped_dir_auto, y_file_auto), "0.0,0.0,0.0,0.0,0.0,0.0\n5.0,5.0,5.0,5.0,5.0,5.0\n")
            write(joinpath(unzipped_dir_auto, va_file_auto), "6.0,0.0\n7.0,0.0\n")

            # Since no zip file exists for year_auto in tmpdir, it should detect and parse the unzipped directory
            sut_auto = P.parse_gloria_sut(tmpdir, year_auto; version = version, price = price)
            @test sut_auto isa Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
            V_auto, U_auto, Y_auto, VA_auto = sut_auto
            @test V_auto == [2.0;;]

            # Test non-existent unzipped directory path throws exception
            @test_throws Exception P.parse_gloria_sut(joinpath(tmpdir, "does_not_exist"), year, true; version = version, price = price)

            # The legacy is_unzipped flag is honored, not ignored
            @test_throws P.ParserError P.parse_gloria_sut(zip_path, year, true; version = version, price = price)
            @test P.parse_gloria_sut(tmpdir, year, false; version = version, price = price)[1] == [2.0;;]

            # Unzipped-only year: false demands a ZIP, true parses the directory
            year_dir_only = year + 5
            t_file_dir = "20260121_120secMother_AllCountries_002_T-Results_$(year_dir_only)_0$(version)_$(ext).csv"
            y_file_dir = "20260121_120secMother_AllCountries_002_Y-Results_$(year_dir_only)_0$(version)_$(ext).csv"
            va_file_dir = "20260121_120secMother_AllCountries_002_V-Results_$(year_dir_only)_0$(version)_$(ext).csv"
            unzipped_dir_only = joinpath(tmpdir, "GLORIA_MRIOs_$(version)_$(year_dir_only)")
            mkpath(unzipped_dir_only)
            write(joinpath(unzipped_dir_only, t_file_dir), "1.0,2.0\n3.0,4.0\n")
            write(joinpath(unzipped_dir_only, y_file_dir), "0.0,0.0,0.0,0.0,0.0,0.0\n5.0,5.0,5.0,5.0,5.0,5.0\n")
            write(joinpath(unzipped_dir_only, va_file_dir), "6.0,0.0\n7.0,0.0\n")
            @test_throws P.ParserError P.parse_gloria_sut(tmpdir, year_dir_only, false; version = version, price = price)
            @test P.parse_gloria_sut(tmpdir, year_dir_only, true; version = version, price = price)[1] == [2.0;;]
        end
    end

    @testset "satellite path and readers" begin
        mktempdir() do tmpdir
            year = 2019
            version = 60
            q_suffix = "_120secMother_AllCountries_002_TQ-Results_$(year)_0$(version)_Markup001(full).csv"
            q_name = "20260121$(q_suffix)"

            write(joinpath(tmpdir, "notes.txt"), "ignore me")
            sat_dir = joinpath(tmpdir, "GLORIA_SatelliteAccounds_060_$(year)")
            mkpath(sat_dir)
            write(joinpath(sat_dir, q_name), "1.0,2.0\n3.0,4.0\n")

            sat_path, sat_kind = P.find_satellite_path(tmpdir, version, year)
            @test sat_path == sat_dir
            @test sat_kind === P.Unzipped

            Q = P.read_satellites(sat_kind, sat_path, q_suffix; n_regions = 1, n_sectors = 1)
            @test Q == [1.0; 3.0;;]

            zip_year = year + 1
            q_zip_suffix = "_120secMother_AllCountries_002_TQ-Results_$(zip_year)_0$(version)_Markup001(full).csv"
            zip_path = joinpath(tmpdir, "GLORIA_SatelliteAccounts_060_$(zip_year).zip")
            ZipArchives.ZipWriter(zip_path) do w
                ZipArchives.zip_newfile(w, "nested/20260121$(q_zip_suffix)")
                write(w, "9.0,10.0\n")
            end

            sat_zip_path, sat_zip_kind = P.find_satellite_path(tmpdir, version, zip_year)
            @test sat_zip_path == zip_path
            @test sat_zip_kind === P.Zipped

            Q_zip = P.read_satellites(sat_zip_kind, sat_zip_path, q_zip_suffix; n_regions = 1, n_sectors = 1)
            @test Q_zip == [9.0;;]

            @test_throws P.ParserError P.find_satellite_path(tmpdir, version, year + 2)

            # Strict version matching: version 6 must not match a "060"
            # fixture (the old substring check matched it), and the error
            # lists the candidates whose version token mismatched
            err_sat = try
                P.find_satellite_path(tmpdir, 6, year)
                nothing
            catch e
                e
            end
            @test err_sat isa P.ParserError
            @test occursin("060", err_sat.msg)
        end
    end

    @testset "_construct_IO" begin
        # 1. Valid inputs/syntax
        regions = ["USA", "CHN"]
        sectors = ["Sec1", "Sec2"]
        va_cats = ["VA1", "VA2"]
        fd_cats = ["FD1", "FD2"]

        V = [
            1.0 0.0 0.0 0.0;
            0.0 2.0 0.0 0.0;
            0.0 0.0 3.0 0.0;
            0.0 0.0 0.0 4.0
        ]
        U = [
            0.5 0.0 0.0 0.0;
            0.0 0.5 0.0 0.0;
            0.0 0.0 0.5 0.0;
            0.0 0.0 0.0 0.5
        ]
        Y = [
            1.0 0.0 0.0 0.0;
            0.0 1.0 0.0 0.0;
            0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 1.0
        ]
        VA = [
            2.0 0.0 0.0 0.0;
            0.0 2.0 0.0 0.0;
            0.0 0.0 2.0 0.0;
            0.0 0.0 0.0 2.0
        ]

        Q_SUT = [
            10.0 20.0 30.0 40.0;
            1.0 2.0 3.0 4.0
        ]
        sat_df = DataFrame(Stressor = ["CO2", "Water"], Source = ["Air", "Fresh"], Unit = ["kg", "m3"])

        res = P._construct_IO(V, U, Y, VA, regions, sectors, va_cats, fd_cats, Q_SUT, sat_df)
        @test res isa Juliora.MRIO
        @test size(res.T.data) == (4, 4)
        @test size(res.A.data) == (4, 4)
        @test size(res.FD.data) == (4, 4)
        @test size(res.VA.data) == (4, 4)
        @test size(res.env.F.data) == (2, 4)
        @test size(res.env.A.data) == (2, 4)
        @test res.env.F.row_indices == sat_df

        # 2. Empty inputs
        empty_sat_df = DataFrame(Stressor = String[], Source = String[], Unit = String[])
        res_empty = P._construct_IO(
            Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
            Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
            String[], String[], String[], String[],
            Matrix{Float64}(undef, 0, 0), empty_sat_df
        )
        @test res_empty isa Juliora.MRIO
        @test size(res_empty.T.data) == (0, 0)
        @test size(res_empty.env.F.data) == (0, 0)
    end

    @testset "degenerate IO warning" begin
        # Zero row/output sums and negative values emit exactly one warning
        V = [1.0 0.0; 0.0 0.0]
        U = [0.5 0.0; 0.0 -1.0]
        Y = [1.0 0.0; 0.0 0.0]
        VA = [2.0 0.0; 0.0 0.0]
        Q_SUT = [10.0 20.0]
        @test_logs (:warn, r"ParserWarning") P._calculate_io_matrices(V, U, Y, VA, Q_SUT)

        # Clean inputs stay silent
        Vc = [1.0 0.0; 0.0 2.0]
        Uc = [0.5 0.0; 0.0 0.5]
        Yc = [1.0 0.0; 0.0 1.0]
        VAc = [2.0 0.0; 0.0 2.0]
        Qc = [10.0 20.0]
        @test_logs P._calculate_io_matrices(Vc, Uc, Yc, VAc, Qc)
    end
end

@testset "GLORIA JLD2 cache" begin
    indices = DataFrame(CountryCode = ["AAA"], Sector = ["test"])
    fd_indices = DataFrame(CountryCode = ["AAA"], Category = ["final"])
    va_indices = DataFrame(CountryCode = ["AAA"], Category = ["value"])
    sat_indices = DataFrame(CountryCode = ["AAA"], Stressor = ["co2"])
    # Build via the positional constructor so the MRIO carries a real
    # environmental extension (the kwargs constructor honestly sets env=nothing),
    # keeping the cache round-trip coverage for env.
    Z_entry = Juliora.MatrixEntry([0.2;;], indices, indices)
    Y_entry = Juliora.MatrixEntry([1.0;;], fd_indices, indices)
    VA_entry = Juliora.MatrixEntry([2.0;;], va_indices, indices)
    x_out = Juliora.calculate_total_output(Z_entry.data, Y_entry.data)  # [1.2]
    A_entry = Juliora.calculate_technical_coefficients(Z_entry, x_out)
    L_fact = Juliora.calculate_leontief_factorization(A_entry)
    X_series = Juliora.SeriesEntry(x_out, indices)
    env_ext = Juliora.EnvironmentalExtension(
        Juliora.MatrixEntry([3.0;;], sat_indices, indices),
        Juliora.MatrixEntry([0.25;;], sat_indices, indices)
    )
    mrio = Juliora.MRIO(A_entry, Z_entry, VA_entry, Y_entry, L_fact, X_series, env_ext)

    mktempdir() do tmpdir
        cache = joinpath(tmpdir, "nested", "mrio.jld2")
        @test Juliora.save_gloria_cache(cache, mrio) == cache
        loaded = Juliora.load_gloria_cache(cache)
        @test loaded isa Juliora.MRIO
        @test loaded.T.data == mrio.T.data
        @test loaded.A.data == mrio.A.data
        @test loaded.VA.data == mrio.VA.data
        @test loaded.FD.data == mrio.FD.data
        @test loaded.X.data == mrio.X.data
        @test loaded.env.F.data == mrio.env.F.data
        @test loaded.T.row_indices == mrio.T.row_indices
        @test haskey(loaded.T.row_lookup, (CountryCode = "AAA", Sector = "test"))
        @test Juliora.solve_leontief(loaded.L, [1.0]) ≈ [1.2]
        @test Juliora.Gloria(cache) isa Juliora.MRIO
        @test Juliora.Gloria(cache, 60, 2019) isa Juliora.MRIO

        alias = joinpath(tmpdir, "mrio.jdl2")
        @test Juliora.save_gloria_cache(mrio, alias) == alias
        @test Juliora.load_gloria_cache(alias).T.data == mrio.T.data

        # Existing valid caches are replaced, while a directory target is not.
        replacement = Juliora.MRIO(;
            Z = Juliora.MatrixEntry([0.3;;], indices, indices),
            Y = Juliora.MatrixEntry([1.0;;], fd_indices, indices),
            VA = Juliora.MatrixEntry([2.0;;], va_indices, indices)
        )
        @test Juliora.save_gloria_cache(cache, replacement) == cache
        @test Juliora.load_gloria_cache(cache).T.data == [0.3;;]
        directory_cache = joinpath(tmpdir, "protected.jld2")
        mkpath(directory_cache)
        marker = joinpath(directory_cache, "keep.txt")
        write(marker, "keep")
        @test_throws ArgumentError Juliora.save_gloria_cache(directory_cache, mrio)
        @test read(marker, String) == "keep"
        @test_throws ArgumentError Juliora.load_gloria_cache(directory_cache)

        malformed = joinpath(tmpdir, "malformed.jld2")
        write(malformed, "not a JLD2 file")
        @test_throws Exception Juliora.load_gloria_cache(malformed)
        wrong_schema = joinpath(tmpdir, "wrong.jld2")
        JLD2.jldopen(wrong_schema, "w") do file
            file["schema"] = "other"
            file["schema_version"] = 1
            file["mrio"] = 1
        end
        @test_throws ArgumentError Juliora.load_gloria_cache(wrong_schema)
        missing_fields = joinpath(tmpdir, "missing_fields.jld2")
        JLD2.jldopen(missing_fields, "w") do file
            file["schema"] = "Juliora.GLORIA.MRIO"
        end
        @test_throws ArgumentError Juliora.load_gloria_cache(missing_fields)
        wrong_type = joinpath(tmpdir, "wrong_type.jld2")
        JLD2.jldopen(wrong_type, "w") do file
            file["schema"] = "Juliora.GLORIA.MRIO"
            file["schema_version"] = 1
            file["mrio"] = 1
        end
        @test_throws ArgumentError Juliora.load_gloria_cache(wrong_type)
        wrong_version = joinpath(tmpdir, "wrong_version.jld2")
        JLD2.jldopen(wrong_version, "w") do file
            file["schema"] = "Juliora.GLORIA.MRIO"
            file["schema_version"] = 999
            file["mrio"] = mrio
        end
        @test_throws ArgumentError Juliora.load_gloria_cache(wrong_version)
        @test_throws ArgumentError Juliora.load_gloria_cache(joinpath(tmpdir, "missing.jld2"))
        @test_throws ArgumentError Juliora.Gloria(tmpdir)
        @test_throws ArgumentError Juliora.Gloria(joinpath(tmpdir, "input.csv"), 60, 2019)
    end
end

@testset "Differential & Thread Safety Testing" begin
    n_reg = 5
    n_sec = 10
    nrows = 2 * n_reg * n_sec # 100
    ncols = nrows
    Random.seed!(1234)

    dense_mat = zeros(nrows, ncols)
    for i in 1:nrows, j in 1:ncols
        if rand() < 0.2
            dense_mat[i, j] = round(rand() * 100, digits = 2)
        end
    end

    csv_io = IOBuffer()
    for i in 1:nrows
        join(csv_io, dense_mat[i, :], ",")
        write(csv_io, "\n")
    end
    csv_bytes = take!(csv_io)

    # Reference run
    S_ref, U_ref = P.parse(P.TFile(), csv_bytes; n_regions = n_reg, n_sectors = n_sec)

    # Concurrent runs (Thread safety check)
    ntasks = 8
    tasks_t = [Threads.@spawn P.parse(P.TFile(), csv_bytes; n_regions = n_reg, n_sectors = n_sec) for _ in 1:ntasks]
    results_t = fetch.(tasks_t)

    for (S_res, U_res) in results_t
        @test S_res == S_ref
        @test U_res == U_ref
    end

    # Do the same for VAFile
    va_rows = 10
    va_dense = zeros(va_rows, ncols)
    for i in 1:va_rows, j in 1:ncols
        if rand() < 0.2
            va_dense[i, j] = round(rand() * 100, digits = 2)
        end
    end
    va_csv_io = IOBuffer()
    for i in 1:va_rows
        join(va_csv_io, va_dense[i, :], ",")
        write(va_csv_io, "\n")
    end
    va_bytes = take!(va_csv_io)

    VA_ref = P.parse(P.VAFile(), va_bytes; n_regions = n_reg, n_sectors = n_sec)
    tasks_va = [Threads.@spawn P.parse(P.VAFile(), va_bytes; n_regions = n_reg, n_sectors = n_sec) for _ in 1:ntasks]
    results_va = fetch.(tasks_va)
    for VA_res in results_va
        @test VA_res == VA_ref
    end

    # Do the same for YFile
    y_cols = 10
    y_dense = zeros(nrows, y_cols)
    for i in 1:nrows, j in 1:y_cols
        if rand() < 0.2
            y_dense[i, j] = round(rand() * 100, digits = 2)
        end
    end
    y_csv_io = IOBuffer()
    for i in 1:nrows
        join(y_csv_io, y_dense[i, :], ",")
        write(y_csv_io, "\n")
    end
    y_bytes = take!(y_csv_io)

    Y_ref = P.parse(P.YFile(), y_bytes; n_regions = n_reg, n_sectors = n_sec)
    tasks_y = [Threads.@spawn P.parse(P.YFile(), y_bytes; n_regions = n_reg, n_sectors = n_sec) for _ in 1:ntasks]
    results_y = fetch.(tasks_y)
    for Y_res in results_y
        @test Y_res == Y_ref
    end

    # ZipReader concurrent read safety of matrix files
    mktempdir() do tmpdir
        zip_path = joinpath(tmpdir, "threads.zip")
        ZipArchives.ZipWriter(zip_path) do w
            ZipArchives.zip_newfile(w, "data.csv")
            write(w, csv_bytes)
        end
        zip_bytes = read(zip_path)
        zip_reader = ZipArchives.ZipReader(zip_bytes)

        st_bytes = ZipArchives.zip_readentry(zip_reader, "data.csv")
        S_st, U_st = P.parse(P.TFile(), st_bytes; n_regions = n_reg, n_sectors = n_sec)

        tasks_zip = [
            Threads.@spawn begin
                    b = ZipArchives.zip_readentry(zip_reader, "data.csv")
                    P.parse(P.TFile(), b; n_regions = n_reg, n_sectors = n_sec)
                end for _ in 1:8
        ]
        mt_zip_results = fetch.(tasks_zip)

        for (S_mt, U_mt) in mt_zip_results
            @test S_mt == S_st
            @test U_mt == U_st
        end
    end
end

@testset "parse_gloria readme labels" begin
    function build_mini_gloria(tmpdir, year, version, region_col)
        ext = P.get_extension(P.BasePrice())
        t_file = "20260121_120secMother_AllCountries_002_T-Results_$(year)_0$(version)_$(ext).csv"
        y_file = "20260121_120secMother_AllCountries_002_Y-Results_$(year)_0$(version)_$(ext).csv"
        va_file = "20260121_120secMother_AllCountries_002_V-Results_$(year)_0$(version)_$(ext).csv"
        sut_dir = joinpath(tmpdir, "GLORIA_MRIOs_$(version)_$(year)")
        mkpath(sut_dir)
        write(joinpath(sut_dir, t_file), "1.0,2.0\n3.0,4.0\n")
        write(joinpath(sut_dir, y_file), "0.5,0.5,0.5,0.5,0.5,0.5\n0.7,0.7,0.7,0.7,0.7,0.7\n")
        write(joinpath(sut_dir, va_file), "0.3,0.0\n")

        sat_dir = joinpath(tmpdir, "GLORIA_Satellite_060_$(year)")
        mkpath(sat_dir)
        q_suffix = "_120secMother_AllCountries_002_TQ-Results_$(year)_0$(version)_Markup001(full).csv"
        write(joinpath(sat_dir, "20260121" * q_suffix), "0.1,0.0\n")

        readme = joinpath(tmpdir, "GLORIA_ReadMe_0$(version).xlsx")
        XLSX.writetable(
            readme, [
                "Regions" => DataFrame(Region_acronyms = region_col, Region_names = ["Region 1", "Region 2"][1:length(region_col)]),
                "Sectors" => DataFrame(Sector_names = ["S1"]),
                "Value added and final demand" => DataFrame(Value_added_names = Union{String, Missing}["VA1", missing, missing, missing, missing, missing], Final_demand_names = ["FD1", "FD2", "FD3", "FD4", "FD5", "FD6"]),
                "Satellites" => DataFrame(Sat_indicator = ["CO2"], Sat_head_indicator = ["Air"], Sat_unit = ["kg"]),
            ]; overwrite = true
        )
        return readme
    end

    mktempdir() do tmpdir
        year = 2020
        version = 60
        build_mini_gloria(tmpdir, year, version, ["R1"])
        mrio = P.parse_gloria(tmpdir, year; version = version)
        @test mrio isa Juliora.MRIO
        @test size(mrio.T.data) == (1, 1)
    end

    # A blank readme cell throws instead of silently shifting labels
    mktempdir() do tmpdir
        year = 2020
        version = 60
        build_mini_gloria(tmpdir, year, version, Union{String, Missing}["", "R2"])
        @test_throws P.ParserError P.parse_gloria(tmpdir, year; version = version)
    end
end

@testset "Stress & Memory Constraint Testing" begin
    # 1. Stress test with large dimensions
    n_reg = 10
    n_sec = 50
    nrows = 2 * n_reg * n_sec # 1000
    line = join(fill("1.5", nrows), ",") * "\n"
    csv_data = Vector{UInt8}(repeat(line, nrows))

    S, U = P.parse(P.TFile(), csv_data; n_regions = n_reg, n_sectors = n_sec)
    @test size(S) == (500, 500)
    @test size(U) == (500, 500)
    @test all(S .== 1.5)
    @test all(U .== 1.5)

    # 2. Verify no stack overflow with massive token stream in a single line
    # (a single-row VA table: VA row counts are file-sized, so one 10000-field
    # row is valid input; single-row T/Y tables are rejected as malformed)
    large_line_io = IOBuffer()
    for j in 1:10000
        print(large_line_io, Float64(j))
        if j < 10000
            print(large_line_io, ",")
        end
    end
    print(large_line_io, "\n")
    large_line_bytes = take!(large_line_io)

    VA_long = P.parse(P.VAFile(), large_line_bytes; n_regions = 1, n_sectors = 5000)
    @test size(VA_long) == (1, 5000)
    @test VA_long[1, 1] == 1.0
    @test VA_long[1, 5000] == 5000.0
end

@testset "parse_gloria integration" begin
    path = "data/GLORIA"
    if !isdir(path)
        path = joinpath(dirname(dirname(pathof(Juliora))), "data", "GLORIA")
    end
    if !isdir(path)
        path = "/home/franzs/Schreibtisch/Arbeit/Juliora/data/GLORIA"
    end
    year = 2019
    version = 60

    mrio = P.parse_gloria(path, year; version = version)

    @test mrio isa Juliora.MRIO
    @test mrio.A isa Juliora.MatrixEntry
    @test mrio.T isa Juliora.MatrixEntry
    @test mrio.VA isa Juliora.MatrixEntry
    @test mrio.FD isa Juliora.MatrixEntry
    @test mrio.L isa Juliora.LeontiefFactorization
    @test mrio.X isa Juliora.SeriesEntry
    @test mrio.env isa Juliora.EnvironmentalExtension

    # Verify dimensions for 164 regions (1 clean empty country removed -> 163 regions remaining), 120 sectors, 6 fd categories, 6 va categories
    # The intermediate matrices after constructing symmetric IOT should be (163 * 120) x (163 * 120) = 19560 x 19560
    @test size(mrio.T.data) == (19560, 19560)
    @test size(mrio.A.data) == (19560, 19560)
    @test size(mrio.FD.data) == (19560, 978) # 163 * 6 = 978 final demand columns
    @test size(mrio.VA.data) == (978, 19560) # 163 * 6 = 978 value added rows
    @test length(mrio.X.data) == 19560

    # Verify EnvironmentalExtension has filled matrices
    @test size(mrio.env.F.data) == (6130, 19560)
    @test size(mrio.env.A.data) == (6130, 19560)
end
