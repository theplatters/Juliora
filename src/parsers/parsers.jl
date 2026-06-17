module Parser

using CSV
using DataFrames
import ZipArchives as za
using LinearAlgebra
using Mmap
using Parsers
using SparseArrays # Crucial for low-RAM high-performance processing
using XLSX
import ..Juliora: MRIO, MatrixEntry, SeriesEntry, EnvironmentalExtension, calculate_leontief_factorization, calculate_technical_coefficients

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

function parse(::TFile, raw_bytes::Vector{UInt8}; delim::Char = ',', n_regions::Integer = N_REGIONS, n_sectors::Integer = N_SECTORS)
    len = length(raw_bytes)
    if len == 0
        return (zeros(0, 0), zeros(0, 0))
    end
    row_starts = find_row_starts(raw_bytes)
    delim_byte = UInt8(delim)
    opts = Parsers.Options(delim = delim)

    S = zeros(n_sectors * n_regions, n_sectors * n_regions)
    U = zeros(n_sectors * n_regions, n_sectors * n_regions)
    nrows = ncols = n_sectors * n_regions * 2

    industry_mask = repeat([fill(true, n_sectors); fill(false, n_sectors)], n_regions)

    actual_rows = min(nrows, length(row_starts))

    Threads.@threads for i in 1:actual_rows
        pos = row_starts[i]
        is_ind_i = industry_mask[i]

        # Pre-calculate row local destination based on identity
        local_i = is_ind_i ?
            global_to_local_industry(i, n_sectors) :
            global_to_local_product(i, n_sectors)

        for j in 1:ncols
            while pos <= len && is_delim_or_newline(raw_bytes[pos], delim_byte)
                pos += 1
            end

            is_ind_j = industry_mask[j]

            if (is_ind_i && is_ind_j) || (!is_ind_i && !is_ind_j)
                while pos <= len && !is_delim_or_newline(raw_bytes[pos], delim_byte)
                    pos += 1
                end
                continue
            end

            res = Parsers.xparse(Float64, raw_bytes, pos, len, opts)
            if Parsers.ok(res.code)
                if is_ind_i  # Industry row, Product col -> Matrix S
                    local_j = global_to_local_product(j, n_sectors)
                    S[local_i, local_j] = res.val
                else         # Product row, Industry col -> Matrix U
                    local_j = global_to_local_industry(j, n_sectors)
                    U[local_i, local_j] = res.val
                end
                pos += res.tlen
            else
                pos += 1
            end
        end
    end

    return (S, U)
end

function parse(::VAFile, raw_data::Vector{UInt8}; delim::Char = ',', n_regions::Integer = N_REGIONS, n_sectors::Integer = N_SECTORS)
    len = length(raw_data)
    if len == 0
        return zeros(0, 0)
    end
    opts = Parsers.Options(delim = delim)

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

    n_industries = n_regions * n_sectors
    A = zeros(nrows, n_industries)

    industry_mask = repeat([fill(true, n_sectors); fill(false, n_sectors)], n_regions)
    product_mask = .!industry_mask  # Product mask is just the exact inverse

    Threads.@threads for i in 1:nrows
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
            if Parsers.ok(res.code)
                local_j = global_to_local_industry(j, n_sectors)
                A[i, local_j] = res.val
                pos += res.tlen
            else
                pos += 1
            end
        end
    end

    return A
end

function parse(::YFile, raw_data::Vector{UInt8}; delim::Char = ',', n_regions::Integer = N_REGIONS, n_sectors::Integer = N_SECTORS)
    len = length(raw_data)
    if len == 0
        return zeros(0, 0)
    end
    opts = Parsers.Options(delim = delim)

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

    n_products = n_regions * n_sectors
    A = zeros(n_products, ncols)

    industry_mask = repeat([fill(true, n_sectors); fill(false, n_sectors)], n_regions)

    Threads.@threads for i in 1:nrows
        industry_mask[i] && continue
        pos = row_starts[i]
        local_i = global_to_local_product(i, n_sectors)
        for j in 1:ncols
            while pos <= len && is_delim_or_newline(raw_data[pos], delim_byte)
                pos += 1
            end

            res = Parsers.xparse(Float64, raw_data, pos, len, opts)
            if Parsers.ok(res.code)
                A[local_i, j] = res.val
                pos += res.tlen
            else
                pos += 1
            end
        end
    end

    return A
end

function detect_gloria_dims(y_bytes::Vector{UInt8}, delim::Char = ',')
    ncols_y = 1
    delim_byte = UInt8(delim)
    len = length(y_bytes)
    if len == 0
        return 0, 0
    end
    for pos in 1:len
        b = y_bytes[pos]
        if b == delim_byte
            ncols_y += 1
        elseif b == 0x0a
            break
        end
    end
    n_regions = max(1, ncols_y ÷ 6)

    # Count rows in y_bytes
    nrows_y = 0
    for pos in 1:len
        if y_bytes[pos] == 0x0a
            nrows_y += 1
        end
    end
    if len > 0 && y_bytes[end] != 0x0a
        nrows_y += 1
    end
    n_sectors = max(1, nrows_y ÷ (2 * n_regions))

    return n_regions, n_sectors
end

function parse_gloria_sut(path::String; year::Integer = 2019, version::Integer = 60, price::AbstractPrice = BasePrice())::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    extension = get_extention(price)

    t_file = "20260121_120secMother_AllCountries_002_T-Results_$(year)_0$(version)_$(extension).csv"
    y_file = "20260121_120secMother_AllCountries_002_Y-Results_$(year)_0$(version)_$(extension).csv"
    va_file = "20260121_120secMother_AllCountries_002_V-Results_$(year)_0$(version)_$(extension).csv"

    # Resolve the correct base path (handling year subdirectory if needed)
    base_path = path
    if !isdir(joinpath(base_path, "GLORIA_MRIOs_$(version)_$(year)")) &&
            !isfile(joinpath(base_path, "GLORIA_MRIOs_$(version)_$(year).zip")) &&
            !isfile(joinpath(base_path, t_file))
        if isdir(joinpath(path, string(year)))
            base_path = joinpath(path, string(year))
        end
    end

    unzipped_dir = joinpath(base_path, "GLORIA_MRIOs_$(version)_$(year)")
    gloria_path = joinpath(base_path, "GLORIA_MRIOs_$(version)_$(year).zip")

    is_unzipped = false
    resolved_dir = ""
    if isdir(unzipped_dir) && isfile(joinpath(unzipped_dir, t_file))
        is_unzipped = true
        resolved_dir = unzipped_dir
    elseif isdir(base_path) && isfile(joinpath(base_path, t_file))
        is_unzipped = true
        resolved_dir = base_path
    elseif !isfile(gloria_path)
        if isdir(base_path)
            subdirs = [joinpath(base_path, "GLORIA_MRIOs_$(version)_$(year)"), base_path]
            found = false
            for sd in subdirs
                if isdir(sd) && isfile(joinpath(sd, t_file))
                    is_unzipped = true
                    resolved_dir = sd
                    found = true
                    break
                end
            end
            if !found
                # Let's check subfolders
                files = readdir(base_path)
                matching_dirs = filter(d -> isdir(joinpath(base_path, d)) && occursin("GLORIA", d) && occursin(string(year), d), files)
                if !isempty(matching_dirs)
                    for md in matching_dirs
                        sd = joinpath(base_path, md)
                        if isfile(joinpath(sd, t_file))
                            is_unzipped = true
                            resolved_dir = sd
                            found = true
                            break
                        end
                    end
                end
                if !found
                    throw(ParserError("Could not find GLORIA SUT files or ZIP archive in $path"))
                end
            end
        else
            throw(ParserError("Path is not a directory: $path"))
        end
    end

    if is_unzipped
        @info "Parsing GLORIA SUT from unzipped directory: $resolved_dir"
        y_bytes = open(joinpath(resolved_dir, y_file), "r") do io
            Mmap.mmap(io)
        end
        n_regions, n_sectors = detect_gloria_dims(y_bytes)
        if n_regions == 0 || n_sectors == 0
            return (zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0))
        end
        S, U = open(joinpath(resolved_dir, t_file), "r") do io
            parse(TFile(), Mmap.mmap(io); n_regions = n_regions, n_sectors = n_sectors)
        end
        Y = parse(YFile(), y_bytes; n_regions = n_regions, n_sectors = n_sectors)
        VA = open(joinpath(resolved_dir, va_file), "r") do io
            parse(VAFile(), Mmap.mmap(io); n_regions = n_regions, n_sectors = n_sectors)
        end
        return (S, U, Y, VA)
    else
        @info "Parsing GLORIA SUT from ZIP archive: $gloria_path"
        return open(gloria_path, "r") do io
            mmap_data = Mmap.mmap(io)
            gloria_zip = za.ZipReader(mmap_data)

            y_bytes = za.zip_readentry(gloria_zip, y_file)
            n_regions, n_sectors = detect_gloria_dims(y_bytes)
            if n_regions == 0 || n_sectors == 0
                return (zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0))
            end

            @info "Parsing T from ZIP"
            t_bytes = za.zip_readentry(gloria_zip, t_file)
            S, U = parse(TFile(), t_bytes; n_regions = n_regions, n_sectors = n_sectors)

            @info "Parsing Y from ZIP"
            Y = parse(YFile(), y_bytes; n_regions = n_regions, n_sectors = n_sectors)

            @info "Parsing VA from ZIP"
            va_bytes = za.zip_readentry(gloria_zip, va_file)
            VA = parse(VAFile(), va_bytes; n_regions = n_regions, n_sectors = n_sectors)

            return (S, U, Y, VA)
        end
    end
end

function parse_gloria_sut(path::String, year::Integer; version::Integer = 60, price::AbstractPrice = BasePrice())
    return parse_gloria_sut(path; year = year, version = version, price = price)
end

function parse_gloria_sut(path::String, year::Integer, is_unzipped::Bool; version::Integer = 60, price::AbstractPrice = BasePrice())
    return parse_gloria_sut(path; year = year, version = version, price = price)
end

struct QFile <: AbstractGloriaElement end
struct QYFile <: AbstractGloriaElement end

function parse(::QFile, raw_data::Vector{UInt8}; delim::Char = ',', n_regions::Integer = N_REGIONS, n_sectors::Integer = N_SECTORS)
    if n_regions < 0 || n_sectors < 0
        throw(ArgumentError("n_regions and n_sectors must be non-negative"))
    end
    len = length(raw_data)
    if len == 0
        return zeros(0, 0)
    end
    opts = Parsers.Options(delim = delim)

    delim_byte = UInt8(delim)
    row_starts = find_row_starts(raw_data)
    nrows = length(row_starts)
    ncols = n_sectors * n_regions * 2

    n_industries = n_regions * n_sectors
    Q = zeros(nrows, n_industries)

    industry_mask = repeat([fill(true, n_sectors); fill(false, n_sectors)], n_regions)

    Threads.@threads for i in 1:nrows
        pos = row_starts[i]
        for j in 1:ncols
            while pos <= len && is_delim_or_newline(raw_data[pos], delim_byte)
                pos += 1
            end

            if !industry_mask[j]
                while pos <= len && !is_delim_or_newline(raw_data[pos], delim_byte)
                    pos += 1
                end
                continue
            end

            res = Parsers.xparse(Float64, raw_data, pos, len, opts)
            if Parsers.ok(res.code)
                local_j = global_to_local_industry(j, n_sectors)
                Q[i, local_j] = res.val
                pos += res.tlen
            else
                pos += 1
            end
        end
    end

    return Q
end

function parse(::QYFile, raw_data::Vector{UInt8}; delim::Char = ',', n_regions::Integer = N_REGIONS, n_sectors::Integer = N_SECTORS)
    if n_regions < 0 || n_sectors < 0
        throw(ArgumentError("n_regions and n_sectors must be non-negative"))
    end
    len = length(raw_data)
    if len == 0
        return zeros(0, 0)
    end
    opts = Parsers.Options(delim = delim)

    delim_byte = UInt8(delim)
    row_starts = find_row_starts(raw_data)
    nrows = length(row_starts)
    
    ncols = 1
    @inbounds for pos in 1:len
        b = raw_data[pos]
        if b == delim_byte
            ncols += 1
        elseif b == 0x0a
            break
        end
    end

    QY = zeros(nrows, ncols)

    Threads.@threads for i in 1:nrows
        pos = row_starts[i]
        for j in 1:ncols
            while pos <= len && is_delim_or_newline(raw_data[pos], delim_byte)
                pos += 1
            end

            res = Parsers.xparse(Float64, raw_data, pos, len, opts)
            if Parsers.ok(res.code)
                QY[i, j] = res.val
                pos += res.tlen
            else
                pos += 1
            end
        end
    end

    return QY
end

function find_file_in_dir(dir::String, suffix::String)
    files = readdir(dir)
    lsuffix = lowercase(suffix)
    for f in files
        if endswith(lowercase(f), lsuffix)
            return joinpath(dir, f)
        end
    end
    throw(ParserError("Could not find file ending with $suffix in directory $dir"))
end

function find_entry_in_zip(zip_reader::za.ZipReader, suffix::String)
    names = za.zip_names(zip_reader)
    lsuffix = lowercase(suffix)
    idx = findfirst(n -> endswith(lowercase(n), lsuffix), names)
    if idx === nothing
        throw(ParserError("Could not find entry ending with $suffix in ZIP"))
    end
    return names[idx]
end

function find_satellite_path(base_path::String, version::Integer, year::Integer)
    files = readdir(base_path)
    for f in files
        full_f = joinpath(base_path, f)
        if isdir(full_f)
            lf = lowercase(f)
            if (occursin("gloria", lf) && occursin("satellite", lf) && occursin(string(year), lf)) ||
               (occursin("gloria", lf) && occursin("sattelite", lf) && occursin(string(year), lf))
                if occursin(string(version), lf)
                    return (full_f, true)
                end
            end
        end
    end
    for f in files
        full_f = joinpath(base_path, f)
        if isdir(full_f)
            lf = lowercase(f)
            if (occursin("gloria", lf) && occursin("satellite", lf) && occursin(string(year), lf)) ||
               (occursin("gloria", lf) && occursin("sattelite", lf) && occursin(string(year), lf))
                return (full_f, true)
            end
        end
    end

    for f in files
        full_f = joinpath(base_path, f)
        if isfile(full_f) && endswith(lowercase(f), ".zip")
            lf = lowercase(f)
            if (occursin("gloria", lf) && occursin("satellite", lf) && occursin(string(year), lf)) ||
               (occursin("gloria", lf) && occursin("sattelite", lf) && occursin(string(year), lf))
                if occursin(string(version), lf)
                    return (full_f, false)
                end
            end
        end
    end
    for f in files
        full_f = joinpath(base_path, f)
        if isfile(full_f) && endswith(lowercase(f), ".zip")
            lf = lowercase(f)
            if (occursin("gloria", lf) && occursin("satellite", lf) && occursin(string(year), lf)) ||
               (occursin("gloria", lf) && occursin("sattelite", lf) && occursin(string(year), lf))
                return (full_f, false)
            end
        end
    end

    throw(ParserError("Could not find GLORIA satellite directory or ZIP archive in $base_path"))
end

function _construct_IO(
        V, U, Y, VA,
        regions::Vector,
        sectors::Vector,
        va_cats::Vector,
        fd_cats::Vector,
        Q_SUT::Matrix{Float64},
        QY_SUT::Matrix{Float64},
        sat_df::DataFrame
    )

    n_regions = length(regions)
    n_sectors = length(sectors)
    n_fd = length(fd_cats)
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

    # Q_mat = Q_SUT * T_matrix
    Q_mat = Q_SUT * T_matrix

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

    keep_fd_mask = fill(true, length(regions) * n_fd)
    for idx in empty_country_indices
        r_start = (idx - 1) * n_fd + 1
        r_end = idx * n_fd
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
    Q_mat = Q_mat[:, keep_sector_mask]
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

    F_entry = MatrixEntry(Q_mat, t_row_indices, sat_df)
    A_env = calculate_technical_coefficients(F_entry, q)
    env_entry = EnvironmentalExtension(F_entry, A_env)

    return MRIO(A_entry, T_entry, VA_entry, FD_entry, L_entry, X_entry, env_entry)
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

    n_regions = length(regions)
    n_sectors = length(sectors)
    n_fd = length(fd_cats)
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

    keep_fd_mask = fill(true, length(regions) * n_fd)
    for idx in empty_country_indices
        r_start = (idx - 1) * n_fd + 1
        r_end = idx * n_fd
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
    # Resolve the correct base path (handling year subdirectory if needed)
    base_path = path
    t_file = "20260121_120secMother_AllCountries_002_T-Results_$(year)_0$(version)_$(get_extention(price)).csv"
    if !isdir(joinpath(base_path, "GLORIA_MRIOs_$(version)_$(year)")) &&
            !isfile(joinpath(base_path, "GLORIA_MRIOs_$(version)_$(year).zip")) &&
            !isfile(joinpath(base_path, t_file))
        if isdir(joinpath(path, string(year)))
            base_path = joinpath(path, string(year))
        end
    end

    (S, U, Y, VA) = parse_gloria_sut(base_path; year = year, version = version, price = price)

    # Find the readme file
    readme_name = "GLORIA_ReadMe_0$(version).xlsx"
    gloria_meta_path = joinpath(base_path, readme_name)
    if !isfile(gloria_meta_path)
        files = readdir(base_path)
        matching = filter(f -> occursin(lowercase("ReadMe"), lowercase(f)) && endswith(lowercase(f), ".xlsx"), files)
        if !isempty(matching)
            gloria_meta_path = joinpath(base_path, matching[1])
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

    # Satellites
    df_sat = DataFrame(XLSX.readtable(gloria_meta_path, "Satellites"))
    stressors = string.(collect(skipmissing(df_sat.Sat_indicator)))
    sources = string.(collect(skipmissing(df_sat.Sat_head_indicator)))
    units = string.(collect(skipmissing(df_sat.Sat_unit)))
    sat_df = DataFrame(Stressor = stressors, Source = sources, Unit = units)

    n_regions = length(regions)
    n_sectors = length(sectors)

    # Find and parse satellite files
    sat_path, is_unzipped = find_satellite_path(base_path, version, year)
    q_suffix = "_120secMother_AllCountries_002_TQ-Results_$(year)_0$(version)_Markup001(full).csv"
    qy_suffix = "_120secMother_AllCountries_002_YQ-Results_$(year)_0$(version)_Markup001(full).csv"

    local Q_SUT::Matrix{Float64}
    local QY_SUT::Matrix{Float64}

    if is_unzipped
        q_file = find_file_in_dir(sat_path, q_suffix)
        qy_file = find_file_in_dir(sat_path, qy_suffix)

        q_bytes = open(q_file, "r") do io
            Mmap.mmap(io)
        end
        qy_bytes = open(qy_file, "r") do io
            Mmap.mmap(io)
        end

        Q_SUT = parse(QFile(), q_bytes; n_regions = n_regions, n_sectors = n_sectors)
        QY_SUT = parse(QYFile(), qy_bytes; n_regions = n_regions, n_sectors = n_sectors)
    else
        Q_SUT, QY_SUT = open(sat_path, "r") do io
            mmap_data = Mmap.mmap(io)
            sat_zip = za.ZipReader(mmap_data)

            q_entry = find_entry_in_zip(sat_zip, q_suffix)
            qy_entry = find_entry_in_zip(sat_zip, qy_suffix)

            q_bytes = za.zip_readentry(sat_zip, q_entry)
            qy_bytes = za.zip_readentry(sat_zip, qy_entry)

            Q_SUT_zip = parse(QFile(), q_bytes; n_regions = n_regions, n_sectors = n_sectors)
            QY_SUT_zip = parse(QYFile(), qy_bytes; n_regions = n_regions, n_sectors = n_sectors)

            return Q_SUT_zip, QY_SUT_zip
        end
    end

    @info "Constructing MRIO"
    return _construct_IO(S, U, Y, VA, regions, sectors, va_cats, fd_cats, Q_SUT, QY_SUT, sat_df)
end


end # Module End
