module Parser

using CSV
using DataFrames
import ZipArchives as za
using LinearAlgebra
using Mmap
using Parsers
using SparseArrays # Crucial for low-RAM high-performance processing
using XLSX
import ..Juliora: MRIO, MatrixEntry, SeriesEntry, EnvironmentalExtension, calculate_leontief_factorization

export ParserError, ParserWarning, parse_gloria, parse_gloria_sut, _construct_IO

# Custom Exceptions
struct ParserError <: Exception
    msg::String
end
struct ParserWarning <: Exception
    msg::String
end

const NEW_LINE = 0x0a
const CARRIAGE_RETURN = 0x0d
const N_REGIONS = 164
const N_SECTORS = 120

# Global Constants matching Python's IDX_NAMES structure
const IDX_NAMES = Dict(
    :T_col => ["region", "system", "sector"],
    :T_row => ["region", "system", "sector"],
    :Y_col2 => ["region", "category"],
    :F_row_cat_unit => ["stressor", "compartment", "unit"],
    :VA_row_region => ["region", "inputtype"],
    :unit => ["unit"]
)


abstract type AbstractGloriaElement end
struct TFile <: AbstractGloriaElement end
struct YFile <: AbstractGloriaElement end
struct VAFile <: AbstractGloriaElement end

abstract type AbstractPrice end
struct BasePrice <: AbstractPrice end
struct PPrice <: AbstractPrice end

get_extention(::BasePrice) = "Markup001(full)"
get_extention(::PPrice) = "Markup005(full)"

"""
    find_row_starts(raw_bytes::Vector{UInt8})

Scans raw byte vectors for newline delimiters to map out matrix coordinate rows.
"""
function find_row_starts(raw_bytes::Vector{UInt8})
    len = length(raw_bytes)
    row_starts = Int[]
    sizehint!(row_starts, len ÷ 50)

    if len > 0
        push!(row_starts, 1)
    end

    @inbounds for pos in 1:(len - 1)
        if raw_bytes[pos] == NEW_LINE # '\n'
            push!(row_starts, pos + 1)
        end
    end
    return row_starts
end


@inline function is_delim_or_newline(b::UInt8, delim_byte::UInt8)
    return b == delim_byte || b == NEW_LINE || b == CARRIAGE_RETURN
end

@inline try_skip_zero(raw_bytes::Vector{UInt8}, pos::Int, len::Int, delim_byte::UInt8) = pos <= len && raw_bytes[pos] == 0x30 &&  (pos == len || is_delim_or_newline(raw_bytes[pos + 1], delim_byte))


abstract type ParsingStrategy end
struct Serial <: ParsingStrategy end
struct Parallel <: ParsingStrategy end

function custom_gloria_sparse_parser(raw_bytes::Vector{UInt8}, nrows::Int, ncols::Int; delim::Char = ',')
    if nrows < 0 || ncols < 0
        throw(ArgumentError("Dimensions must be non-negative"))
    end

    len = length(raw_bytes)
    row_starts = find_row_starts(raw_bytes)
    actual_rows = min(nrows, length(row_starts))

    n_threads = Threads.nthreads()
    strategy = (n_threads <= 1 || actual_rows < 20) ? Serial() : Parallel()

    return custom_gloria_sparse_parser(strategy, raw_bytes, len, row_starts, actual_rows, nrows, ncols, delim, n_threads)
end

function custom_gloria_sparse_parser(::Serial, raw_bytes::Vector{UInt8}, len::Int, row_starts::Vector{Int}, actual_rows::Int, nrows::Int, ncols::Int, delim::Char, n_threads::Int)
    delim_byte = UInt8(delim)
    opts = Parsers.Options(delim = delim)

    est_nnz = (actual_rows * ncols) ÷ 5
    I = Vector{Int}(undef, est_nnz)
    J = Vector{Int}(undef, est_nnz)
    V = Vector{Float64}(undef, est_nnz)

    parse_chunk_range!(I, J, V, raw_bytes, len, row_starts, 1, actual_rows, ncols, delim_byte, opts)

    @info "Finished parsing matrix into highly compressed sparse format"
    return construct_sparse_from_row_major(I, J, V, nrows, ncols)
end


function global_to_local_industry(global_idx::Int, n_sectors::Int)
    region = (global_idx - 1) ÷ (2 * n_sectors)
    rem = (global_idx - 1) % (2 * n_sectors)

    return (region * n_sectors) + rem + 1
end

function global_to_local_product(global_idx::Int, n_sectors::Int)
    region = (global_idx - 1) ÷ (2 * n_sectors)
    rem = (global_idx - 1) % (2 * n_sectors)

    return (region * n_sectors) + (rem - n_sectors) + 1
end

function parse(::TFile, raw_bytes::Vector{UInt8}; delim::Char = ',')
    len = length(raw_bytes)
    row_starts = find_row_starts(raw_bytes)
    delim_byte = UInt8(delim)
    opts = Parsers.Options(delim = delim)

    S = zeros(N_SECTORS * N_REGIONS, N_SECTORS * N_REGIONS)
    U = zeros(N_SECTORS * N_REGIONS, N_SECTORS * N_REGIONS)
    nrows = ncols = N_SECTORS * N_REGIONS * 2

    industry_mask = repeat([fill(true, N_SECTORS); fill(false, N_SECTORS)], N_REGIONS)
    product_mask = .!industry_mask  # Product mask is just the exact inverse


    Threads.@threads for i in 1:nrows
        pos = row_starts[i]
        is_ind_i = industry_mask[i]

        # Pre-calculate row local destination based on identity
        local_i = is_ind_i ?
            global_to_local_industry(i, N_SECTORS) :
            global_to_local_product(i, N_SECTORS)

        for j in 1:ncols
            while pos <= len && (is_delim_or_newline(raw_bytes[pos], delim_byte))
                pos += 1
            end

            res = Parsers.xparse(Float64, raw_bytes, pos, len, opts)
            is_ind_j = industry_mask[j]

            if is_ind_i && !is_ind_j  # Industry row, Product col -> Matrix S
                local_j = global_to_local_product(j, N_SECTORS)
                if Parsers.ok(res.code)
                    S[local_i, local_j] = res.val
                end
            elseif !is_ind_i && is_ind_j # Product row, Industry col -> Matrix U
                local_j = global_to_local_industry(j, N_SECTORS)
                if Parsers.ok(res.code)
                    U[local_i, local_j] = res.val
                end
            end

            pos += res.tlen
        end
    end

    return (S, U)
end

function parse(::VAFile, raw_data::Vector{UInt8})
    len = length(raw_data)
    opts = Parsers.Options(delim = ',')

    # Detect the delimiter (either ',' or ';') by inspecting the first line
    delim = ','


    # Determine structural column count from the initial data row using the detected delimiter
    ncols = 1
    delim_byte = UInt8(delim)
    @inbounds for pos in 1:len
        b = raw_data[pos]
        if b == delim_byte
            ncols += 1
        elseif b == 0x0a # '\n'
            break
        end
    end

    row_starts = find_row_starts(raw_data)
    nrows = length(row_starts)

    A = zeros(nrows, ncols)
    nrows, ncols = size(A)
    len = length(raw_data)

    industry_mask = repeat([fill(true, N_SECTORS); fill(false, N_SECTORS)], N_REGIONS)
    product_mask = .!industry_mask  # Product mask is just the exact inverse


    Threads.@threads  for i in 1:nrows
        pos = row_starts[i]
        for j in 1:ncols


            while pos <= len && is_delim_or_newline(raw_data[pos], delim_byte)
                pos += 1
            end

            if product_mask[j]
                while pos <= len && !is_delim_or_newline(raw_data[pos], delim_byte)
                    pos += 1
                end
                continue
            end

            res = Parsers.xparse(Float64, raw_data, pos, len, opts)
            if Parsers.ok(res.code) && product_mask[j]
                local_j = global_to_local_industry(j, N_SECTORS)
                A[i, local_j] = res.val
                pos += res.tlen
            else
                pos += res.tlen
            end
        end
    end

    return A
end

function parse(::YFile, raw_data::Vector{UInt8})
    len = length(raw_data)
    opts = Parsers.Options(delim = ',')

    # Detect the delimiter (either ',' or ';') by inspecting the first line
    delim = ','


    # Determine structural column count from the initial data row using the detected delimiter
    ncols = 1
    delim_byte = UInt8(delim)
    @inbounds for pos in 1:len
        b = raw_data[pos]
        if b == delim_byte
            ncols += 1
        elseif b == 0x0a # '\n'
            break
        end
    end

    row_starts = find_row_starts(raw_data)
    nrows = length(row_starts)

    A = zeros(nrows, ncols)
    nrows, ncols = size(A)
    len = length(raw_data)

    industry_mask = repeat([fill(true, N_SECTORS); fill(false, N_SECTORS)], N_REGIONS)
    product_mask = .!industry_mask  # Product mask is just the exact inverse


    for i in 1:nrows
        industry_mask[i] && continue
        pos = row_starts[i]
        for j in 1:ncols

            while pos <= len && (is_delim_or_newline(raw_data[pos], delim_byte))
                pos += 1
            end

            res = Parsers.xparse(Float64, raw_data, pos, len, opts)
            if Parsers.ok(res.code)

                local_i = global_to_local_product(j, N_SECTORS)
                A[local_i, j] = res.val

                pos += res.tlen
            else
                pos += 1
            end
        end
    end

    return A
end


function read(file_type::AbstractGloriaElement, filename)
    return open(filename, "r") do file
        raw_data = Mmap.mmap(file)
        return parse(file_type, raw_data)
    end
end

"""
    parse_gloria_sut(path::String, year::Int, is_unzipped::Bool; version = 60, price::AbstractPrice = BasePrice())

Reads GLORIA SUT matrices from an unzipped directory.
"""
function parse_gloria_sut(path::String; year::Integer = 2019, version::Integer = 60, price::AbstractPrice = BasePrice())

    extension = get_extention(price)


    S, U = read(TFile(), joinpath(path, "20260121_120secMother_AllCountries_002_T-Results_$(year)_0$(version)_$(extension).csv"))
    Y = read(YFile(), joinpath(path, "20260121_120secMother_AllCountries_002_Y-Results_$(year)_0$(version)_$(extension).csv"))
    VA = read(VAFile(), joinpath(path, "20260121_120secMother_AllCountries_002_V-Results_$(year)_0$(version)_$(extension).csv"))
    return (S, U, Y, VA)
end


"""
    _construct_IO(sut_matrices::Dict{String, SparseMatrixCSC{Float64, Int}}, regions::Vector, sectors::Vector, va_cats::Vector, fd_cats::Vector)

Constructs the symmetric input-output matrices and indices, returning a complete MRIO struct.
"""
function _construct_IO(
        V, U, Y, VA,
        regions::Vector,
        sectors::Vector,
        va_cats::Vector,
        fd_cats::Vector
    )

    n_regions = 164
    n_sectors = 120
    # Industry output (row sum of V)
    g = vec(sum(V, dims = 2))
    g_inv = map(x -> x == 0.0 ? 0.0 : 1.0 / x, g)

    # T_matrix = diag(g_inv) * V
    T_matrix = g_inv .* V

    # Z (intermediate transaction matrix) = U * T_matrix
    Z_mat = U * T_matrix

    # Commodity output q = row sum of U + row sum of Y
    q = vec(sum(U, dims = 2) + sum(Y, dims = 2))
    q_inv = map(x -> x == 0.0 ? 0.0 : 1.0 / x, q)

    # A = Z_mat * diag(q_inv) = Z_mat .* q_inv'
    A_mat = Z_mat .* q_inv'

    # VA_new = VA * T_matrix
    VA_mat = VA * T_matrix

    # Identify empty countries (Such as DYE in 2022)
    row_sums = vec(sum(Z_mat, dims = 2))
    col_sums = vec(sum(Z_mat, dims = 1))

    empty_country_indices = Int[]
    for i in 1:length(regions)
        r_start = (i - 1) * n_sectors + 1
        r_end = i * n_sectors

        is_empty_row = all(row_sums[r_start:r_end] .== 0.0)
        is_empty_col = all(col_sums[r_start:r_end] .== 0.0)

        if is_empty_row && is_empty_col
            push!(empty_country_indices, i)
        end
    end

    keep_country_mask = fill(true, length(regions))
    keep_country_mask[empty_country_indices] .= false
    regions_clean = regions[keep_country_mask]

    keep_sector_mask = fill(true, length(regions) * n_sectors)
    for idx in empty_country_indices
        r_start = (idx - 1) * n_sectors + 1
        r_end = idx * n_sectors
        keep_sector_mask[r_start:r_end] .= false
    end

    keep_fd_mask = fill(true, length(regions) * 6)
    for idx in empty_country_indices
        r_start = (idx - 1) * 6 + 1
        r_end = idx * 6
        keep_fd_mask[r_start:r_end] .= false
    end

    keep_va_mask = fill(true, length(regions) * length(va_cats))
    for idx in empty_country_indices
        r_start = (idx - 1) * length(va_cats) + 1
        r_end = idx * length(va_cats)
        keep_va_mask[r_start:r_end] .= false
    end

    Z_mat = Z_mat[keep_sector_mask, keep_sector_mask]
    A_mat = A_mat[keep_sector_mask, keep_sector_mask]
    Y = Y[keep_sector_mask, keep_fd_mask]
    VA_mat = VA_mat[keep_va_mask, keep_sector_mask]
    q = q[keep_sector_mask]

    # Construct index DataFrames
    n_regions_clean = length(regions_clean)

    country_codes = Vector{String}(undef, n_regions_clean * n_sectors)
    sector_names = Vector{String}(undef, n_regions_clean * n_sectors)
    idx = 1
    for r in regions_clean
        for s in sectors
            country_codes[idx] = string(r)
            sector_names[idx] = string(s)
            idx += 1
        end
    end
    t_row_indices = DataFrame(CountryCode = country_codes, Sector = sector_names)

    country_codes_fd = Vector{String}(undef, n_regions_clean * length(fd_cats))
    fd_categories = Vector{String}(undef, n_regions_clean * length(fd_cats))
    idx = 1
    for r in regions_clean
        for c in fd_cats
            country_codes_fd[idx] = string(r)
            fd_categories[idx] = string(c)
            idx += 1
        end
    end
    fd_col_indices = DataFrame(CountryCode = country_codes_fd, Category = fd_categories)

    country_codes_va = Vector{String}(undef, n_regions_clean * length(va_cats))
    va_categories = Vector{String}(undef, n_regions_clean * length(va_cats))
    idx = 1
    for r in regions_clean
        for c in va_cats
            country_codes_va[idx] = string(r)
            va_categories[idx] = string(c)
            idx += 1
        end
    end
    va_row_indices = DataFrame(CountryCode = country_codes_va, Category = va_categories)

    @info "Constructing Matrix Entries"
    T_entry = MatrixEntry(Z_mat, t_row_indices, t_row_indices)
    A_entry = MatrixEntry(A_mat, t_row_indices, t_row_indices)
    FD_entry = MatrixEntry(Y, fd_col_indices, t_row_indices)
    VA_entry = MatrixEntry(VA_mat, t_row_indices, va_row_indices)
    X_entry = SeriesEntry(q, t_row_indices)


    @info "Calculating Leontief"
    L_entry = calculate_leontief_factorization(A_entry)

    # Empty EnvironmentalExtension as required by rule 4
    empty_f_data = sparse(zeros(0, size(A_mat, 2)))
    empty_row_indices = DataFrame(Stressor = String[], Source = String[])
    empty_F = MatrixEntry(empty_f_data, t_row_indices, empty_row_indices)
    empty_A = MatrixEntry(empty_f_data, t_row_indices, empty_row_indices)
    env_entry = EnvironmentalExtension(empty_F, empty_A)

    return MRIO(A_entry, T_entry, VA_entry, FD_entry, L_entry, X_entry, env_entry)
end


"""
    parse_gloria(path::String, year::Int; version = 60, price = "bp", country_names = "gloria", construct = "B")

Parses raw GLORIA SUT tables, reads Excel readme metadata, and constructs a complete MRIO database.
"""
function parse_gloria(path::String, year::Int; version = 60, price = BasePrice(), country_names = "gloria")
    (S, U, Y, VA) = parse_gloria_sut(path; year = year, version = version, price = price)

    # Find the readme file
    readme_name = "GLORIA_ReadMe_0$(version).xlsx"
    gloria_meta_path = joinpath(path, readme_name)
    if !isfile(gloria_meta_path)
        files = readdir(path)
        matching = filter(f -> occursin(lowercase("ReadMe"), lowercase(f)) && endswith(lowercase(f), ".xlsx"), files)
        if !isempty(matching)
            gloria_meta_path = joinpath(path, matching[1])
        else
            throw(ParserError("Could not find GLORIA readme xlsx file in $path"))
        end
    end

    # Regions
    df_regions = DataFrame(XLSX.readtable(gloria_meta_path, "Regions"))
    country_col = Symbol(country_names == "gloria" ? "Region_acronyms" : "Region_names")
    regions = collect(skipmissing(df_regions[!, country_col]))

    # Sectors
    df_sectors = DataFrame(XLSX.readtable(gloria_meta_path, "Sectors"))
    sectors = collect(skipmissing(df_sectors.Sector_names))

    # Value added and final demand
    df_va_fd = DataFrame(XLSX.readtable(gloria_meta_path, "Value added and final demand"))
    va_cats = collect(skipmissing(df_va_fd.Value_added_names))
    fd_cats = collect(skipmissing(df_va_fd.Final_demand_names))

    @info "Constructing MRIO"
    return _construct_IO(S, U, Y, VA, regions, sectors, va_cats, fd_cats)
end

end # Module End
