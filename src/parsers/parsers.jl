module Parser

using CSV
using DataFrames
import ZipArchives as za
using LinearAlgebra
using Mmap
using Parsers
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

get_extension(::BasePrice) = "Markup001(full)"
get_extension(::PPrice) = "Markup005(full)"

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

@inline function skip_delims(raw_data::Vector{UInt8}, pos::Int, len::Int, delim_byte::UInt8)::Int
    while pos <= len && is_delim_or_newline(raw_data[pos], delim_byte)
        pos += 1
    end
    return pos
end

@inline function skip_field(raw_data::Vector{UInt8}, pos::Int, len::Int, delim_byte::UInt8)::Int
    while pos <= len && !is_delim_or_newline(raw_data[pos], delim_byte)
        pos += 1
    end
    return pos
end

@inline function is_zero_field(raw_data::Vector{UInt8}, pos::Int, len::Int, delim_byte::UInt8)::Bool
    return pos <= len &&
        raw_data[pos] == 0x30 && # '0'
        (pos == len || is_delim_or_newline(raw_data[pos + 1], delim_byte))
end

function count_first_row_columns(raw_data::Vector{UInt8}, delim_byte::UInt8)::Int
    ncols = 1
    @inbounds for pos in 1:length(raw_data)
        b = raw_data[pos]
        if b == delim_byte
            ncols += 1
        elseif b == NEW_LINE
            break
        end
    end
    return ncols
end

function count_rows(raw_data::Vector{UInt8})::Int
    len = length(raw_data)
    nrows = 0
    @inbounds for pos in 1:len
        raw_data[pos] == NEW_LINE && (nrows += 1)
    end
    if len > 0 && raw_data[end] != NEW_LINE
        nrows += 1
    end
    return nrows
end

function industry_mask(n_regions::Integer, n_sectors::Integer)::Vector{Bool}
    return repeat([fill(true, n_sectors); fill(false, n_sectors)], n_regions)
end

function mmap_file(path::String)
    return open(path, "r") do io
        Mmap.mmap(io)
    end
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

    ind_mask = industry_mask(n_regions, n_sectors)

    actual_rows = min(nrows, length(row_starts))

    Threads.@threads for i in 1:actual_rows
        pos = row_starts[i]
        is_ind_i = ind_mask[i]

        # Pre-calculate row local destination based on identity
        local_i = is_ind_i ?
            global_to_local_industry(i, n_sectors) :
            global_to_local_product(i, n_sectors)

        for j in 1:ncols
            pos = skip_delims(raw_bytes, pos, len, delim_byte)

            is_ind_j = ind_mask[j]

            if (is_ind_i && is_ind_j) || (!is_ind_i && !is_ind_j)
                pos = skip_field(raw_bytes, pos, len, delim_byte)
                continue
            end

            if is_zero_field(raw_bytes, pos, len, delim_byte)
                pos = skip_field(raw_bytes, pos, len, delim_byte)
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
    delim_byte = UInt8(delim)
    ncols = count_first_row_columns(raw_data, delim_byte)

    row_starts = find_row_starts(raw_data)
    nrows = length(row_starts)

    n_industries = n_regions * n_sectors
    A = zeros(nrows, n_industries)

    product_mask = .!industry_mask(n_regions, n_sectors)

    Threads.@threads for i in 1:nrows
        pos = row_starts[i]
        for j in 1:ncols
            pos = skip_delims(raw_data, pos, len, delim_byte)

            if product_mask[j]
                pos = skip_field(raw_data, pos, len, delim_byte)
                continue
            end

            if is_zero_field(raw_data, pos, len, delim_byte)
                pos = skip_field(raw_data, pos, len, delim_byte)
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
    delim_byte = UInt8(delim)
    ncols = count_first_row_columns(raw_data, delim_byte)

    row_starts = find_row_starts(raw_data)
    nrows = length(row_starts)

    n_products = n_regions * n_sectors
    A = zeros(n_products, ncols)

    ind_mask = industry_mask(n_regions, n_sectors)

    Threads.@threads for i in 1:nrows
        if !ind_mask[i]
            pos = row_starts[i]
            local_i = global_to_local_product(i, n_sectors)
            for j in 1:ncols
                pos = skip_delims(raw_data, pos, len, delim_byte)

                if is_zero_field(raw_data, pos, len, delim_byte)
                    pos = skip_field(raw_data, pos, len, delim_byte)
                    continue
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
    end

    return A
end

function detect_gloria_dims(y_bytes::Vector{UInt8}, delim::Char = ',')
    delim_byte = UInt8(delim)
    len = length(y_bytes)
    if len == 0
        return 0, 0
    end
    ncols_y = count_first_row_columns(y_bytes, delim_byte)
    n_regions = max(1, ncols_y ÷ 6)

    nrows_y = count_rows(y_bytes)
    n_sectors = max(1, nrows_y ÷ (2 * n_regions))

    return n_regions, n_sectors
end

const base_gloria_name = "20260121_120secMother_AllCountries_002_"

gloria_mrios_name(version::Integer, year::Integer) = "GLORIA_MRIOs_$(version)_$(year)"
gloria_mrios_zip_name(version::Integer, year::Integer) = "$(gloria_mrios_name(version, year)).zip"
gloria_result_file(prefix::String, year::Integer, version::Integer, extension::String) = "$(base_gloria_name)$(prefix)-Results_$(year)_0$(version)_$(extension).csv"
gloria_satellite_suffix(prefix::String, year::Integer, version::Integer) = "_120secMother_AllCountries_002_$(prefix)Q-Results_$(year)_0$(version)_Markup001(full).csv"

function resolve_gloria_base_path(path::String, year::Integer, version::Integer, t_file::String)::String
    base_path = path
    if !isdir(joinpath(base_path, gloria_mrios_name(version, year))) &&
            !isfile(joinpath(base_path, gloria_mrios_zip_name(version, year))) &&
            !isfile(joinpath(base_path, t_file)) &&
            isdir(joinpath(path, string(year)))
        base_path = joinpath(path, string(year))
    end
    return base_path
end

function parse_gloria_sut(path::String; year::Integer = 2019, version::Integer = 60, price::AbstractPrice = BasePrice())::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    extension = get_extension(price)

    t_file = gloria_result_file("T", year, version, extension)
    y_file = gloria_result_file("Y", year, version, extension)
    va_file = gloria_result_file("V", year, version, extension)

    # Resolve the correct base path (handling year subdirectory if needed)
    base_path = resolve_gloria_base_path(path, year, version, t_file)

    unzipped_dir = joinpath(base_path, gloria_mrios_name(version, year))
    gloria_path = joinpath(base_path, gloria_mrios_zip_name(version, year))

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
            subdirs = [joinpath(base_path, gloria_mrios_name(version, year)), base_path]
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
        y_bytes = mmap_file(joinpath(resolved_dir, y_file))
        n_regions, n_sectors = detect_gloria_dims(y_bytes)
        if n_regions == 0 || n_sectors == 0
            return (zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0))
        end
        S, U = parse(TFile(), mmap_file(joinpath(resolved_dir, t_file)); n_regions = n_regions, n_sectors = n_sectors)
        Y = parse(YFile(), y_bytes; n_regions = n_regions, n_sectors = n_sectors)
        VA = parse(VAFile(), mmap_file(joinpath(resolved_dir, va_file)); n_regions = n_regions, n_sectors = n_sectors)
        return (S, U, Y, VA)
    else
        @info "Parsing GLORIA SUT from ZIP archive: $gloria_path"
        return open(gloria_path, "r") do io
            mmap_data = Mmap.mmap(io)
            gloria_zip = za.ZipReader(mmap_data)

            y_bytes = read_entry_in_zip(gloria_zip, y_file)
            n_regions, n_sectors = detect_gloria_dims(y_bytes)
            if n_regions == 0 || n_sectors == 0
                return (zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0))
            end

            @info "Parsing T from ZIP"
            t_bytes = read_entry_in_zip(gloria_zip, t_file)
            S, U = parse(TFile(), t_bytes; n_regions = n_regions, n_sectors = n_sectors)

            @info "Parsing Y from ZIP"
            Y = parse(YFile(), y_bytes; n_regions = n_regions, n_sectors = n_sectors)

            @info "Parsing VA from ZIP"
            va_bytes = read_entry_in_zip(gloria_zip, va_file)
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

    ind_mask = industry_mask(n_regions, n_sectors)

    Threads.@threads for i in 1:nrows
        pos = row_starts[i]
        for j in 1:ncols
            pos = skip_delims(raw_data, pos, len, delim_byte)

            if !ind_mask[j]
                pos = skip_field(raw_data, pos, len, delim_byte)
                continue
            end

            if is_zero_field(raw_data, pos, len, delim_byte)
                pos = skip_field(raw_data, pos, len, delim_byte)
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
    ncols = count_first_row_columns(raw_data, delim_byte)

    QY = zeros(nrows, ncols)

    Threads.@threads for i in 1:nrows
        pos = row_starts[i]
        for j in 1:ncols
            pos = skip_delims(raw_data, pos, len, delim_byte)

            if is_zero_field(raw_data, pos, len, delim_byte)
                pos = skip_field(raw_data, pos, len, delim_byte)
                continue
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

function read_entry_in_zip(zip_reader::za.ZipReader, suffix::String)
    return za.zip_readentry(zip_reader, find_entry_in_zip(zip_reader, suffix))
end

function find_readme_path(base_path::String, version::Integer, original_path::String)::String
    readme_name = "GLORIA_ReadMe_0$(version).xlsx"
    gloria_meta_path = joinpath(base_path, readme_name)
    isfile(gloria_meta_path) && return gloria_meta_path

    files = readdir(base_path)
    matching = filter(f -> occursin("readme", lowercase(f)) && endswith(lowercase(f), ".xlsx"), files)
    !isempty(matching) && return joinpath(base_path, matching[1])

    throw(ParserError("Could not find GLORIA readme xlsx file in $original_path"))
end

function collect_nonmissing_strings(df::DataFrame, col::Symbol)::Vector{String}
    return string.(collect(skipmissing(df[!, col])))
end


abstract type Unzipped end
abstract type Zipped end

function _is_satellite_match(f, year)
    lf = lowercase(f)
    return occursin("gloria", lf) &&
        (occursin("satellite", lf) || occursin("sattelite", lf)) &&
        occursin(string(year), lf)

end

function _is_version_match(f, version::Integer)
    lf = lowercase(f)
    version_str = string(version)
    padded_version = lpad(version_str, 3, '0')
    return occursin(version_str, lf) || occursin(padded_version, lf)
end

function find_satellite_path(base_path::String, version::Integer, year::Integer)
    files = readdir(base_path; join = false)

    candidates = Tuple{String, DataType, String}[]

    for f in files
        full_f = joinpath(base_path, f)
        lf = lowercase(f)

        if isdir(full_f) && _is_satellite_match(f, year)
            push!(candidates, (full_f, Unzipped, lf))
        elseif isfile(full_f) && endswith(lf, ".zip") && _is_satellite_match(f, year)
            push!(candidates, (full_f, Zipped, lf))
        end
    end

    for (path, kind, lf) in candidates
        _is_version_match(lf, version) && return (path, kind)
    end

    !isempty(candidates) && return (candidates[1][1], candidates[1][2])

    throw(ParserError("Could not find GLORIA satellite directory or ZIP archive in $base_path"))
end

function _inverse_or_zero(v)
    return map(x -> x == 0.0 ? 0.0 : 1.0 / x, v)
end

function _calculate_io_matrices(V, U, Y, VA, Q_SUT::Matrix{Float64})
    g = vec(sum(V, dims = 2))
    T_matrix = _inverse_or_zero(g) .* V

    Z_mat = U * T_matrix
    q = vec(sum(U, dims = 2) + sum(Y, dims = 2))
    A_mat = Z_mat .* _inverse_or_zero(q)'
    VA_mat = VA * T_matrix
    Q_mat = Q_SUT * T_matrix

    return (Z = Z_mat, A = A_mat, FD = Y, VA = VA_mat, Q = Q_mat, output = q)
end

function _find_empty_country_indices(Z_mat, n_regions::Int, n_sectors::Int)::Vector{Int}
    row_sums = vec(sum(Z_mat, dims = 2))
    col_sums = vec(sum(Z_mat, dims = 1))
    empty_country_indices = Int[]

    for i in 1:n_regions
        r_start = (i - 1) * n_sectors + 1
        r_end = i * n_sectors

        is_empty_row = all(row_sums[r_start:r_end] .== 0.0)
        is_empty_col = all(col_sums[r_start:r_end] .== 0.0)

        if is_empty_row && is_empty_col
            push!(empty_country_indices, i)
        end
    end

    return empty_country_indices
end

function _build_keep_masks(
        empty_country_indices::Vector{Int},
        n_regions::Int,
        n_sectors::Int,
        n_fd::Int,
        n_va::Int
    )
    keep_country_mask = fill(true, n_regions)
    keep_sector_mask = fill(true, n_regions * n_sectors)
    keep_fd_mask = fill(true, n_regions * n_fd)
    keep_va_mask = fill(true, n_regions * n_va)

    keep_country_mask[empty_country_indices] .= false

    for idx in empty_country_indices
        sector_start = (idx - 1) * n_sectors + 1
        sector_end = idx * n_sectors
        keep_sector_mask[sector_start:sector_end] .= false

        fd_start = (idx - 1) * n_fd + 1
        fd_end = idx * n_fd
        keep_fd_mask[fd_start:fd_end] .= false

        va_start = (idx - 1) * n_va + 1
        va_end = idx * n_va
        keep_va_mask[va_start:va_end] .= false
    end

    return (country = keep_country_mask, sector = keep_sector_mask, fd = keep_fd_mask, va = keep_va_mask)
end

function _apply_empty_country_filter(matrices, masks)
    return (
        Z = matrices.Z[masks.sector, masks.sector],
        A = matrices.A[masks.sector, masks.sector],
        FD = matrices.FD[masks.sector, masks.fd],
        VA = matrices.VA[masks.va, masks.sector],
        Q = matrices.Q[:, masks.sector],
        output = matrices.output[masks.sector],
    )
end

function _build_sector_indices(regions_clean, sectors::Vector)::DataFrame
    n_regions_clean = length(regions_clean)
    n_sectors = length(sectors)
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

    return DataFrame(CountryCode = country_codes, Sector = sector_names)
end

function _build_category_indices(regions_clean, categories::Vector)::DataFrame
    n_regions_clean = length(regions_clean)
    n_categories = length(categories)
    country_codes = Vector{String}(undef, n_regions_clean * n_categories)
    category_names = Vector{String}(undef, n_regions_clean * n_categories)

    idx = 1
    for r in regions_clean
        for c in categories
            country_codes[idx] = string(r)
            category_names[idx] = string(c)
            idx += 1
        end
    end

    return DataFrame(CountryCode = country_codes, Category = category_names)
end

function _build_mrio_entries(matrices, t_row_indices::DataFrame, fd_col_indices::DataFrame, va_row_indices::DataFrame, sat_df::DataFrame)
    @info "Constructing Matrix Entries"
    T_entry = MatrixEntry(matrices.Z, t_row_indices, t_row_indices)
    A_entry = MatrixEntry(matrices.A, t_row_indices, t_row_indices)
    FD_entry = MatrixEntry(matrices.FD, fd_col_indices, t_row_indices)
    VA_entry = MatrixEntry(matrices.VA, t_row_indices, va_row_indices)
    X_entry = SeriesEntry(matrices.output, t_row_indices)

    @info "Calculating Leontief"
    L_entry = calculate_leontief_factorization(A_entry)

    F_entry = MatrixEntry(matrices.Q, t_row_indices, sat_df)
    A_env = calculate_technical_coefficients(F_entry, matrices.output)
    env_entry = EnvironmentalExtension(F_entry, A_env)

    return (A = A_entry, T = T_entry, VA = VA_entry, FD = FD_entry, L = L_entry, X = X_entry, env = env_entry)
end

function _construct_IO(
        V, U, Y, VA,
        regions::Vector,
        sectors::Vector,
        va_cats::Vector,
        fd_cats::Vector,
        Q_SUT::Matrix{Float64},
        sat_df::DataFrame
    )

    n_regions = length(regions)
    n_sectors = length(sectors)
    n_fd = length(fd_cats)
    n_va = length(va_cats)

    matrices = _calculate_io_matrices(V, U, Y, VA, Q_SUT)
    empty_country_indices = _find_empty_country_indices(matrices.Z, n_regions, n_sectors)
    masks = _build_keep_masks(empty_country_indices, n_regions, n_sectors, n_fd, n_va)
    matrices_clean = _apply_empty_country_filter(matrices, masks)

    regions_clean = regions[masks.country]
    t_row_indices = _build_sector_indices(regions_clean, sectors)
    fd_col_indices = _build_category_indices(regions_clean, fd_cats)
    va_row_indices = _build_category_indices(regions_clean, va_cats)

    entries = _build_mrio_entries(matrices_clean, t_row_indices, fd_col_indices, va_row_indices, sat_df)

    return MRIO(entries.A, entries.T, entries.VA, entries.FD, entries.L, entries.X, entries.env)
end


"""
    parse_gloria(path::String, year::Int; version = 60, price = "bp", country_names = "gloria", construct = "B")

Parses raw GLORIA SUT tables, reads Excel readme metadata, and constructs a complete MRIO database.
"""
function parse_gloria(path::String, year::Int; version = 60, price = BasePrice(), country_names = "gloria")
    # Resolve the correct base path (handling year subdirectory if needed)
    t_file = gloria_result_file("T", year, version, get_extension(price))
    base_path = resolve_gloria_base_path(path, year, version, t_file)

    (S, U, Y, VA) = parse_gloria_sut(base_path; year = year, version = version, price = price)

    # Find the readme file
    gloria_meta_path = find_readme_path(base_path, version, path)

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
    stressors = collect_nonmissing_strings(df_sat, :Sat_indicator)
    sources = collect_nonmissing_strings(df_sat, :Sat_head_indicator)
    units = collect_nonmissing_strings(df_sat, :Sat_unit)
    sat_df = DataFrame(Stressor = stressors, Source = sources, Unit = units)

    n_regions = length(regions)
    n_sectors = length(sectors)

    # Find and parse satellite files
    sat_path, is_unzipped = find_satellite_path(base_path, version, year)
    q_suffix = gloria_satellite_suffix("T", year, version)

    Q_SUT = read_satellites(is_unzipped, sat_path, q_suffix; n_regions = n_regions, n_sectors = n_sectors)

    @info "Constructing MRIO"
    return _construct_IO(S, U, Y, VA, regions, sectors, va_cats, fd_cats, Q_SUT, sat_df)
end

function read_satellites(::Type{Zipped}, sat_path, q_suffix; n_regions, n_sectors)
    return open(sat_path, "r") do io
        mmap_data = Mmap.mmap(io)
        sat_zip = za.ZipReader(mmap_data)

        q_bytes = read_entry_in_zip(sat_zip, q_suffix)

        return parse(QFile(), q_bytes; n_regions = n_regions, n_sectors = n_sectors)
    end
end

function read_satellites(::Type{Unzipped}, sat_path, q_suffix; n_regions, n_sectors)
    q_file = find_file_in_dir(sat_path, q_suffix)

    q_bytes = mmap_file(q_file)

    return parse(QFile(), q_bytes; n_regions = n_regions, n_sectors = n_sectors)
end

end # Module End
