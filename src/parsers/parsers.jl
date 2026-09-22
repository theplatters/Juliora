"""
Juliora GLORIA parser.

Reads unquoted, delimiter-separated GLORIA SUT/satellite CSV files with a fast
byte scanner and builds complete [`MRIO`](@ref) databases.

# CSV assumptions

- Files are plain unquoted CSV: fields are separated by a single delimiter
  (default `','`) and rows by `'\\n'` (`'\\r\\n'` is tolerated). Quote handling
  is NOT implemented: a `'"'` character is treated as ordinary field content
  and such a field falls back to `0.0` (see below).
- Every row must contain exactly the expected number of fields. Empty fields
  (two adjacent delimiters, or a delimiter at a row start/end) parse as `0.0`;
  short/long rows throw [`ParserError`](@ref).
- Any number of trailing delimiters/whitespace after the expected field count
  at end-of-row is tolerated (`_check_row_trailing`): values are fully
  consumed by then, so no misalignment is possible.
- Unparseable tokens are leniently treated as `0.0` (the whole field is
  skipped, never re-scanned mid-field), so sparse/placeholder cells do not
  abort a parse.
- A leading UTF-8 BOM (`EF BB BF`) is skipped.
- Table dimensions are inferred from the Y file assuming 6 final-demand
  categories per region, then cross-validated against the T file.
"""
module Parser

using CSV
using DataFrames
import ZipArchives as za
using LinearAlgebra
using Mmap
using Parsers
using XLSX
using JLD2
import ..Juliora: MRIO, MatrixEntry, SeriesEntry, EnvironmentalExtension, calculate_leontief_factorization, calculate_technical_coefficients

export ParserError, ParserWarning, parse_gloria, parse_gloria_sut, save_gloria_cache, load_gloria_cache, is_gloria_cache_path, is_gloria_zip_path, _construct_IO

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

const GLORIA_CACHE_SCHEMA = "Juliora.GLORIA.MRIO"
const GLORIA_CACHE_VERSION = 1 # Bump when the serialized layout or required fields change.

is_gloria_cache_path(path::String) = endswith(lowercase(path), ".jld2") || endswith(lowercase(path), ".jdl2")
is_gloria_zip_path(path::String) = endswith(lowercase(path), ".zip")

function _check_cache_path(path::String)
    return is_gloria_cache_path(path) ||
        throw(ArgumentError("GLORIA cache path must end in .jld2 (the .jdl2 alias is also accepted): $path"))
end

"""
    save_gloria_cache(path::String, mrio::MRIO)

Save the complete `MRIO` to a versioned JLD2 cache, including labels,
environmental data, and its Leontief factorization. Creates parent
directories and writes atomically. `.jdl2` is accepted as an alias for
`.jld2`; returns the supplied path.
"""
function save_gloria_cache(path::String, mrio::MRIO)
    _check_cache_path(path)
    target = abspath(path)
    isdir(target) && throw(ArgumentError("Cannot save GLORIA cache: target is a directory: $path"))
    parent = dirname(target)
    mkpath(parent)
    temporary_path, temporary_io = mktemp(parent)
    close(temporary_io)
    try
        jldopen(temporary_path, "w") do file
            file["schema"] = GLORIA_CACHE_SCHEMA
            file["schema_version"] = GLORIA_CACHE_VERSION
            file["mrio"] = mrio
        end
        Base.mv(temporary_path, target; force = true)
    catch err
        isfile(temporary_path) && rm(temporary_path; force = true)
        throw(ErrorException("Could not save GLORIA cache $path: $(sprint(showerror, err))"))
    end
    return path
end

save_gloria_cache(mrio::MRIO, path::String) = save_gloria_cache(path, mrio)

"""
    load_gloria_cache(path::String)::MRIO

Load and validate a complete MRIO from a versioned `.jld2` cache. The
`.jdl2` spelling is also accepted. Missing, malformed, incompatible, and
non-MRIO files produce descriptive errors.
"""
function load_gloria_cache(path::String)::MRIO
    _check_cache_path(path)
    isdir(path) && throw(ArgumentError("Cannot load GLORIA cache: path is a directory: $path"))
    isfile(path) || throw(ArgumentError("GLORIA cache file does not exist: $path"))
    file = try
        jldopen(path, "r")
    catch err
        throw(ErrorException("Malformed GLORIA cache $path: $(sprint(showerror, err))"))
    end
    try
        haskey(file, "schema") && haskey(file, "schema_version") && haskey(file, "mrio") ||
            throw(ArgumentError("GLORIA cache has the wrong schema (missing required fields): $path"))
        file["schema"] == GLORIA_CACHE_SCHEMA && file["schema_version"] == GLORIA_CACHE_VERSION ||
            throw(ArgumentError("Unsupported GLORIA cache schema or version in $path"))
        mrio = try
            file["mrio"]
        catch err
            throw(
                ErrorException(
                    "GLORIA cache at $path appears stale or incompatible (written by another Juliora version); " *
                        "re-parse with parse_gloria and overwrite the cache: $(sprint(showerror, err))"
                )
            )
        end
        mrio isa MRIO ||
            throw(ArgumentError("GLORIA cache contains $(typeof(mrio)), not an MRIO: $path"))
        return mrio
    finally
        close(file)
    end
end

const UTF8_BOM = (0xef, 0xbb, 0xbf)

"""Return 3 when `raw` starts with a UTF-8 BOM, else 0 (scan start offset)."""
@inline function _bom_offset(raw::Vector{UInt8})::Int
    return length(raw) >= 3 && raw[1] == UTF8_BOM[1] && raw[2] == UTF8_BOM[2] && raw[3] == UTF8_BOM[3] ? 3 : 0
end

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
function find_row_starts(raw_bytes::Vector{UInt8}, est_rows::Integer = 0)
    len = length(raw_bytes)
    row_starts = Int[]
    est_rows > 0 && sizehint!(row_starts, Int(est_rows))

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


"""Index of the next `delim_byte` at or after `pos`, bounded by `row_end`.

`row_end` is exclusive: the returned index satisfies `pos <= result <= row_end`
and never points past the end of the current row, so field scanning cannot
wrap into the next row.
"""
@inline function _field_end(raw_data::Vector{UInt8}, pos::Int, row_end::Int, delim_byte::UInt8)::Int
    fend = pos
    while fend < row_end
        @inbounds b = raw_data[fend]
        b == delim_byte && break
        fend += 1
    end
    return fend
end

"""Parse the field `raw_data[fstart:(fend - 1)]` as `Float64`.

Empty fields and unparseable tokens leniently yield `0.0`; the caller always
advances past the whole field (to `fend + 1`), never re-scanning mid-field.
"""
@inline function _parse_field_value(raw_data::Vector{UInt8}, fstart::Int, fend::Int, opts::Parsers.Options)::Float64
    fstart >= fend && return 0.0
    if fend == fstart + 1
        @inbounds return raw_data[fstart] == 0x30 ? 0.0 : _xparse_or_zero(raw_data, fstart, fend, opts)
    end
    return _xparse_or_zero(raw_data, fstart, fend, opts)
end

@inline function _xparse_or_zero(raw_data::Vector{UInt8}, fstart::Int, fend::Int, opts::Parsers.Options)::Float64
    res = Parsers.xparse(Float64, raw_data, fstart, fend - 1, opts)
    return Parsers.ok(res.code) ? res.val : 0.0
end

"""Check that `raw_data[pos:(row_end - 1)]` holds only delimiters/whitespace.

Called after the expected field count was consumed; any other byte means the
row has extra fields. Throws `ParserError` naming `fname` and the row.
"""
function _check_row_trailing(raw_data::Vector{UInt8}, pos::Int, row_end::Int, delim_byte::UInt8, fname::String, i::Integer, ncols::Integer)
    p = pos
    while p < row_end
        @inbounds b = raw_data[p]
        if b == delim_byte || b == 0x20 || b == 0x09 || b == CARRIAGE_RETURN
            p += 1
        else
            throw(ParserError("$fname row $i: expected $ncols fields, found more"))
        end
    end
    return nothing
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

function parse(::TFile, raw_bytes::Vector{UInt8}; delim::Char = ',', n_regions::Integer = N_REGIONS, n_sectors::Integer = N_SECTORS, fname::String = "T")
    (n_regions < 0 || n_sectors < 0) && throw(ArgumentError("n_regions and n_sectors must be non-negative"))
    len = length(raw_bytes)
    if len == 0
        return (zeros(0, 0), zeros(0, 0))
    end
    ncols = n_sectors * n_regions * 2
    nrows = ncols
    row_starts = find_row_starts(raw_bytes, cld(len, 2 * ncols + 2))
    length(row_starts) == nrows ||
        throw(ParserError("$fname: expected $nrows rows (2 x $n_regions regions x $n_sectors sectors), found $(length(row_starts))"))

    delim_byte = UInt8(delim)
    opts = Parsers.Options(delim = delim)
    bom = _bom_offset(raw_bytes)

    S = zeros(n_sectors * n_regions, n_sectors * n_regions)
    U = zeros(n_sectors * n_regions, n_sectors * n_regions)

    ind_mask = industry_mask(n_regions, n_sectors)

    row_err = Ref{Union{Nothing, ParserError}}(nothing)
    Threads.@threads for i in 1:nrows
        row_err[] !== nothing && continue
        try
            row_end = i < nrows ? row_starts[i + 1] - 1 : len + 1
            pos = row_starts[i] + (i == 1 ? bom : 0)
            is_ind_i = ind_mask[i]

            # Pre-calculate row local destination based on identity
            local_i = is_ind_i ?
                global_to_local_industry(i, n_sectors) :
                global_to_local_product(i, n_sectors)

            for j in 1:ncols
                if pos >= row_end
                    # A row ending in a delimiter (or a wholly empty first
                    # field) still holds one final empty field -> 0.0.
                    if pos == row_end && (j == 1 || raw_bytes[pos - 1] == delim_byte)
                        pos += 1
                        continue
                    end
                    throw(ParserError("$fname row $i: expected $ncols fields, found $(j - 1)"))
                end
                @inbounds b = raw_bytes[pos]
                if b == delim_byte
                    # Empty field -> 0.0, advance exactly one delimiter.
                    pos += 1
                    continue
                end
                fend = _field_end(raw_bytes, pos, row_end, delim_byte)

                is_ind_j = ind_mask[j]
                if (is_ind_i && is_ind_j) || (!is_ind_i && !is_ind_j)
                    pos = fend + 1
                    continue
                end

                v = _parse_field_value(raw_bytes, pos, fend, opts)
                if v != 0.0
                    if is_ind_i  # Industry row, Product col -> Matrix S
                        @inbounds S[local_i, global_to_local_product(j, n_sectors)] = v
                    else         # Product row, Industry col -> Matrix U
                        @inbounds U[local_i, global_to_local_industry(j, n_sectors)] = v
                    end
                end
                pos = fend + 1
            end
            _check_row_trailing(raw_bytes, pos, row_end, delim_byte, fname, i, ncols)
        catch e
            e isa ParserError || rethrow()
            row_err[] === nothing && (row_err[] = e)
        end
    end
    row_err[] !== nothing && throw(row_err[])

    return (S, U)
end

function parse(::VAFile, raw_data::Vector{UInt8}; delim::Char = ',', n_regions::Integer = N_REGIONS, n_sectors::Integer = N_SECTORS, fname::String = "VA")
    (n_regions < 0 || n_sectors < 0) && throw(ArgumentError("n_regions and n_sectors must be non-negative"))
    len = length(raw_data)
    if len == 0
        return zeros(0, 0)
    end
    opts = Parsers.Options(delim = delim)

    # Determine structural column count from the initial data row using the detected delimiter
    delim_byte = UInt8(delim)
    ncols_expected = 2 * n_regions * n_sectors
    ncols = count_first_row_columns(raw_data, delim_byte)
    ncols == ncols_expected ||
        throw(ParserError("$fname: expected $ncols_expected columns (2 x $n_regions regions x $n_sectors sectors), found $ncols"))

    row_starts = find_row_starts(raw_data, cld(len, 2 * ncols + 2))
    nrows = length(row_starts)
    bom = _bom_offset(raw_data)

    n_industries = n_regions * n_sectors
    A = zeros(nrows, n_industries)

    product_mask = .!industry_mask(n_regions, n_sectors)

    row_err = Ref{Union{Nothing, ParserError}}(nothing)
    Threads.@threads for i in 1:nrows
        row_err[] !== nothing && continue
        try
            row_end = i < nrows ? row_starts[i + 1] - 1 : len + 1
            pos = row_starts[i] + (i == 1 ? bom : 0)
            for j in 1:ncols
                if pos >= row_end
                    # A row ending in a delimiter (or a wholly empty first
                    # field) still holds one final empty field -> 0.0.
                    if pos == row_end && (j == 1 || raw_data[pos - 1] == delim_byte)
                        pos += 1
                        continue
                    end
                    throw(ParserError("$fname row $i: expected $ncols fields, found $(j - 1)"))
                end
                @inbounds b = raw_data[pos]
                if b == delim_byte
                    # Empty field -> 0.0, advance exactly one delimiter.
                    pos += 1
                    continue
                end
                fend = _field_end(raw_data, pos, row_end, delim_byte)

                if product_mask[j]
                    pos = fend + 1
                    continue
                end

                v = _parse_field_value(raw_data, pos, fend, opts)
                if v != 0.0
                    @inbounds A[i, global_to_local_industry(j, n_sectors)] = v
                end
                pos = fend + 1
            end
            _check_row_trailing(raw_data, pos, row_end, delim_byte, fname, i, ncols)
        catch e
            e isa ParserError || rethrow()
            row_err[] === nothing && (row_err[] = e)
        end
    end
    row_err[] !== nothing && throw(row_err[])

    return A
end

function parse(::YFile, raw_data::Vector{UInt8}; delim::Char = ',', n_regions::Integer = N_REGIONS, n_sectors::Integer = N_SECTORS, fname::String = "Y")
    (n_regions < 0 || n_sectors < 0) && throw(ArgumentError("n_regions and n_sectors must be non-negative"))
    len = length(raw_data)
    if len == 0
        return zeros(0, 0)
    end
    opts = Parsers.Options(delim = delim)

    # Determine structural column count from the initial data row using the detected delimiter
    delim_byte = UInt8(delim)
    ncols = count_first_row_columns(raw_data, delim_byte)

    row_starts = find_row_starts(raw_data, cld(len, 2 * ncols + 2))
    nrows_expected = 2 * n_regions * n_sectors
    nrows = length(row_starts)
    nrows == nrows_expected ||
        throw(ParserError("$fname: expected $nrows_expected rows (2 x $n_regions regions x $n_sectors sectors), found $nrows"))
    bom = _bom_offset(raw_data)

    n_products = n_regions * n_sectors
    A = zeros(n_products, ncols)

    ind_mask = industry_mask(n_regions, n_sectors)

    row_err = Ref{Union{Nothing, ParserError}}(nothing)
    Threads.@threads for i in 1:nrows
        row_err[] !== nothing && continue
        try
            row_end = i < nrows ? row_starts[i + 1] - 1 : len + 1
            pos = row_starts[i] + (i == 1 ? bom : 0)
            keep_row = !ind_mask[i]
            local_i = keep_row ? global_to_local_product(i, n_sectors) : 0
            for j in 1:ncols
                if pos >= row_end
                    # A row ending in a delimiter (or a wholly empty first
                    # field) still holds one final empty field -> 0.0.
                    if pos == row_end && (j == 1 || raw_data[pos - 1] == delim_byte)
                        pos += 1
                        continue
                    end
                    throw(ParserError("$fname row $i: expected $ncols fields, found $(j - 1)"))
                end
                @inbounds b = raw_data[pos]
                if b == delim_byte
                    # Empty field -> 0.0, advance exactly one delimiter.
                    pos += 1
                    continue
                end
                fend = _field_end(raw_data, pos, row_end, delim_byte)

                if keep_row
                    v = _parse_field_value(raw_data, pos, fend, opts)
                    if v != 0.0
                        @inbounds A[local_i, j] = v
                    end
                end
                pos = fend + 1
            end
            _check_row_trailing(raw_data, pos, row_end, delim_byte, fname, i, ncols)
        catch e
            e isa ParserError || rethrow()
            row_err[] === nothing && (row_err[] = e)
        end
    end
    row_err[] !== nothing && throw(row_err[])

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

"""
    validate_sut_dims(t_bytes, y_bytes, n_regions, n_sectors, t_file)

Cross-validate dimensions detected from the Y file against the T file: the Y
column count must be divisible by 6 final-demand categories per region, the Y
row count must split evenly over regions, and the T file must hold exactly
`2 * n_regions * n_sectors` rows and columns. Throws `ParserError` naming the
offending file (with expected/actual counts) otherwise.
"""
function validate_sut_dims(t_bytes::Vector{UInt8}, y_bytes::Vector{UInt8}, n_regions::Integer, n_sectors::Integer, t_file::String)
    delim_byte = UInt8(',')
    y_ncols = count_first_row_columns(y_bytes, delim_byte)
    y_ncols % 6 == 0 ||
        throw(ParserError("Y file has $y_ncols columns, which is not divisible by 6 final-demand categories per region (detected $n_regions regions); the ÷6 FD-category assumption may be wrong"))
    y_nrows = count_rows(y_bytes)
    y_nrows % (2 * n_regions) == 0 ||
        throw(ParserError("Y file has $y_nrows rows, which does not split evenly over $n_regions regions (detected $n_sectors sectors); the ÷6 FD-category assumption may be wrong"))

    expected = 2 * n_regions * n_sectors
    nrows_t = count_rows(t_bytes)
    nrows_t == expected ||
        throw(ParserError("$t_file has $nrows_t rows, expected $expected (2 x $n_regions regions x $n_sectors sectors); the ÷6 FD-category assumption may be wrong"))
    ncols_t = count_first_row_columns(t_bytes, delim_byte)
    ncols_t == expected ||
        throw(ParserError("$t_file first row has $ncols_t columns, expected $expected (2 x $n_regions regions x $n_sectors sectors)"))
    return nothing
end

const base_gloria_name = "20260121_120secMother_AllCountries_002_"

gloria_mrios_name(version::Integer, year::Integer) = "GLORIA_MRIOs_$(version)_$(year)"
gloria_mrios_zip_name(version::Integer, year::Integer) = "$(gloria_mrios_name(version, year)).zip"
gloria_result_file(prefix::String, year::Integer, version::Integer, extension::String) = "$(base_gloria_name)$(prefix)-Results_$(year)_0$(version)_$(extension).csv"
gloria_satellite_suffix(prefix::String, year::Integer, version::Integer, price::AbstractPrice = BasePrice()) = "_120secMother_AllCountries_002_$(prefix)Q-Results_$(year)_0$(version)_$(get_extension(price)).csv"

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

function _parse_sut_dir_sequence(resolved_dir::String, t_file::String, y_file::String, va_file::String)::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    for f in (t_file, y_file, va_file)
        isfile(joinpath(resolved_dir, f)) ||
            throw(ParserError("Required GLORIA file $f not found in directory $resolved_dir"))
    end
    @info "Parsing GLORIA SUT from unzipped directory: $resolved_dir"
    y_bytes = mmap_file(joinpath(resolved_dir, y_file))
    n_regions, n_sectors = detect_gloria_dims(y_bytes)
    if n_regions == 0 || n_sectors == 0
        t_empty = filesize(joinpath(resolved_dir, t_file)) == 0
        va_empty = filesize(joinpath(resolved_dir, va_file)) == 0
        y_bytes = nothing
        (t_empty && va_empty) ||
            throw(ParserError("Y file $y_file is empty but T/VA files in $resolved_dir are not; cannot detect dimensions"))
        return (zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0))
    end
    filesize(joinpath(resolved_dir, t_file)) == 0 &&
        throw(ParserError("Required GLORIA file $t_file in $resolved_dir is empty"))
    filesize(joinpath(resolved_dir, va_file)) == 0 &&
        throw(ParserError("Required GLORIA file $va_file in $resolved_dir is empty"))
    t_bytes = mmap_file(joinpath(resolved_dir, t_file))
    va_bytes = mmap_file(joinpath(resolved_dir, va_file))
    validate_sut_dims(t_bytes, y_bytes, n_regions, n_sectors, t_file)
    S, U = parse(TFile(), t_bytes; n_regions = n_regions, n_sectors = n_sectors, fname = t_file)
    t_bytes = nothing
    Y = parse(YFile(), y_bytes; n_regions = n_regions, n_sectors = n_sectors, fname = y_file)
    y_bytes = nothing
    VA = parse(VAFile(), va_bytes; n_regions = n_regions, n_sectors = n_sectors, fname = va_file)
    va_bytes = nothing
    return (S, U, Y, VA)
end

function _parse_sut_zip_sequence(gloria_path::String, t_file::String, y_file::String, va_file::String)::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    @info "Parsing GLORIA SUT from ZIP archive: $gloria_path"
    return open(gloria_path, "r") do io
        mmap_data = Mmap.mmap(io)
        gloria_zip = za.ZipReader(mmap_data)

        y_bytes = read_entry_in_zip(gloria_zip, y_file, gloria_path)
        n_regions, n_sectors = detect_gloria_dims(y_bytes)
        if n_regions == 0 || n_sectors == 0
            t_empty = length(read_entry_in_zip(gloria_zip, t_file, gloria_path)) == 0
            va_empty = length(read_entry_in_zip(gloria_zip, va_file, gloria_path)) == 0
            y_bytes = nothing
            (t_empty && va_empty) ||
                throw(ParserError("Y file $y_file is empty but T/VA entries in $gloria_path are not; cannot detect dimensions"))
            return (zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0))
        end

        @info "Parsing T from ZIP"
        t_bytes = read_entry_in_zip(gloria_zip, t_file, gloria_path)
        isempty(t_bytes) && throw(ParserError("Required GLORIA entry $t_file in $gloria_path is empty"))
        @info "Parsing VA from ZIP"
        va_bytes = read_entry_in_zip(gloria_zip, va_file, gloria_path)
        isempty(va_bytes) && throw(ParserError("Required GLORIA entry $va_file in $gloria_path is empty"))

        validate_sut_dims(t_bytes, y_bytes, n_regions, n_sectors, t_file)

        S, U = parse(TFile(), t_bytes; n_regions = n_regions, n_sectors = n_sectors, fname = t_file)
        t_bytes = nothing

        @info "Parsing Y from ZIP"
        Y = parse(YFile(), y_bytes; n_regions = n_regions, n_sectors = n_sectors, fname = y_file)
        y_bytes = nothing

        @info "Parsing VA from ZIP"
        VA = parse(VAFile(), va_bytes; n_regions = n_regions, n_sectors = n_sectors, fname = va_file)
        va_bytes = nothing
        mmap_data = nothing
        gloria_zip = nothing

        return (S, U, Y, VA)
    end
end

function parse_gloria_sut(path::String; year::Integer = 2019, version::Integer = 60, price::AbstractPrice = BasePrice())::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    extension = get_extension(price)

    t_file = gloria_result_file("T", year, version, extension)
    y_file = gloria_result_file("Y", year, version, extension)
    va_file = gloria_result_file("V", year, version, extension)

    # A direct ZIP is authoritative; otherwise retain the historical directory search.
    direct_zip = isfile(path) && is_gloria_zip_path(path)
    if isfile(path) && !direct_zip
        throw(ParserError("Unsupported GLORIA source file (expected .zip): $path"))
    end
    base_path = direct_zip ? dirname(path) : resolve_gloria_base_path(path, year, version, t_file)

    unzipped_dir = joinpath(base_path, gloria_mrios_name(version, year))
    gloria_path = direct_zip ? path : joinpath(base_path, gloria_mrios_zip_name(version, year))

    is_unzipped = false
    resolved_dir = ""
    if !direct_zip && isdir(unzipped_dir) && isfile(joinpath(unzipped_dir, t_file))
        is_unzipped = true
        resolved_dir = unzipped_dir
    elseif !direct_zip && isdir(base_path) && isfile(joinpath(base_path, t_file))
        is_unzipped = true
        resolved_dir = base_path
    elseif !direct_zip && !isfile(gloria_path)
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
        return _parse_sut_dir_sequence(resolved_dir, t_file, y_file, va_file)
    else
        return _parse_sut_zip_sequence(gloria_path, t_file, y_file, va_file)
    end
end

function parse_gloria_sut(path::String, year::Integer; version::Integer = 60, price::AbstractPrice = BasePrice())
    return parse_gloria_sut(path; year = year, version = version, price = price)
end

function parse_gloria_sut(path::String, year::Integer, is_unzipped::Bool; version::Integer = 60, price::AbstractPrice = BasePrice())
    extension = get_extension(price)
    t_file = gloria_result_file("T", year, version, extension)
    y_file = gloria_result_file("Y", year, version, extension)
    va_file = gloria_result_file("V", year, version, extension)
    if is_unzipped
        isfile(path) &&
            throw(ParserError("parse_gloria_sut: is_unzipped=true but path is a file, not a directory: $path"))
        isdir(path) ||
            throw(ParserError("parse_gloria_sut: is_unzipped=true but path is not a directory: $path"))
        if isfile(joinpath(path, t_file))
            resolved_dir = path
        elseif isfile(joinpath(path, gloria_mrios_name(version, year), t_file))
            resolved_dir = joinpath(path, gloria_mrios_name(version, year))
        else
            throw(ParserError("parse_gloria_sut: is_unzipped=true but required file $t_file not found under $path"))
        end
        return _parse_sut_dir_sequence(resolved_dir, t_file, y_file, va_file)
    else
        if isfile(path) && is_gloria_zip_path(path)
            zip_path = path
        elseif isdir(path)
            zip_path = joinpath(path, gloria_mrios_zip_name(version, year))
            isfile(zip_path) ||
                throw(ParserError("parse_gloria_sut: is_unzipped=false but ZIP archive $zip_path not found"))
        else
            throw(ParserError("parse_gloria_sut: is_unzipped=false but path is neither a ZIP file nor a directory: $path"))
        end
        return _parse_sut_zip_sequence(zip_path, t_file, y_file, va_file)
    end
end

struct QFile <: AbstractGloriaElement end

function parse(::QFile, raw_data::Vector{UInt8}; delim::Char = ',', n_regions::Integer = N_REGIONS, n_sectors::Integer = N_SECTORS, fname::String = "Q")
    if n_regions < 0 || n_sectors < 0
        throw(ArgumentError("n_regions and n_sectors must be non-negative"))
    end
    len = length(raw_data)
    if len == 0
        return zeros(0, 0)
    end
    opts = Parsers.Options(delim = delim)

    delim_byte = UInt8(delim)
    ncols_expected = n_sectors * n_regions * 2
    ncols = count_first_row_columns(raw_data, delim_byte)
    ncols == ncols_expected ||
        throw(ParserError("$fname: expected $ncols_expected columns (2 x $n_regions regions x $n_sectors sectors), found $ncols"))
    row_starts = find_row_starts(raw_data, cld(len, 2 * ncols + 2))
    nrows = length(row_starts)
    bom = _bom_offset(raw_data)

    n_industries = n_regions * n_sectors
    Q = zeros(nrows, n_industries)

    ind_mask = industry_mask(n_regions, n_sectors)

    row_err = Ref{Union{Nothing, ParserError}}(nothing)
    Threads.@threads for i in 1:nrows
        row_err[] !== nothing && continue
        try
            row_end = i < nrows ? row_starts[i + 1] - 1 : len + 1
            pos = row_starts[i] + (i == 1 ? bom : 0)
            for j in 1:ncols
                if pos >= row_end
                    # A row ending in a delimiter (or a wholly empty first
                    # field) still holds one final empty field -> 0.0.
                    if pos == row_end && (j == 1 || raw_data[pos - 1] == delim_byte)
                        pos += 1
                        continue
                    end
                    throw(ParserError("$fname row $i: expected $ncols fields, found $(j - 1)"))
                end
                @inbounds b = raw_data[pos]
                if b == delim_byte
                    # Empty field -> 0.0, advance exactly one delimiter.
                    pos += 1
                    continue
                end
                fend = _field_end(raw_data, pos, row_end, delim_byte)

                if !ind_mask[j]
                    pos = fend + 1
                    continue
                end

                v = _parse_field_value(raw_data, pos, fend, opts)
                if v != 0.0
                    @inbounds Q[i, global_to_local_industry(j, n_sectors)] = v
                end
                pos = fend + 1
            end
            _check_row_trailing(raw_data, pos, row_end, delim_byte, fname, i, ncols)
        catch e
            e isa ParserError || rethrow()
            row_err[] === nothing && (row_err[] = e)
        end
    end
    row_err[] !== nothing && throw(row_err[])

    return Q
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

function find_entry_in_zip(zip_reader::za.ZipReader, suffix::String, archive_path::String = "")
    names = za.zip_names(zip_reader)
    lsuffix = lowercase(suffix)
    idx = findfirst(n -> endswith(lowercase(n), lsuffix), names)
    if idx === nothing
        where = isempty(archive_path) ? "in ZIP" : "in ZIP archive $archive_path"
        throw(ParserError("Could not find entry ending with $suffix $where"))
    end
    return names[idx]
end

function read_entry_in_zip(zip_reader::za.ZipReader, suffix::String, archive_path::String = "")
    return za.zip_readentry(zip_reader, find_entry_in_zip(zip_reader, suffix, archive_path))
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
    return string.(coalesce.(df[!, col], ""))
end

"""Read a readme label column without shifting positions.

Blank cells become `""` (positions are preserved, unlike `skipmissing` which
shifts all subsequent labels); trailing blank cells are dropped. Mid-table
blanks are kept as `""` so `_require_nonblank_labels` can report them.
"""
function collect_readme_labels(df::DataFrame, col::Symbol, sheet::String)::Vector{String}
    labels = string.(coalesce.(df[!, col], ""))
    last = findlast(l -> !isempty(strip(l)), labels)
    return last === nothing ? String[] : labels[1:last]
end

"""Throw `ParserError` if any label is blank; `sheet` names the readme sheet."""
function _require_nonblank_labels(labels::Vector{String}, sheet::String, what::String)
    for (k, label) in enumerate(labels)
        isempty(strip(label)) &&
            throw(ParserError("$sheet sheet: blank $what label at position $k"))
    end
    return nothing
end


abstract type Unzipped end
abstract type Zipped end

function _is_satellite_match(f, year)
    lf = lowercase(f)
    return occursin("gloria", lf) &&
        (occursin("satellite", lf) || occursin("sattelite", lf)) &&
        occursin(string(year), lf)

end

function _has_version_token(lf::String, token::String)::Bool
    return occursin(Regex(string("(?<![0-9])", token, "(?![0-9])")), lf)
end

function _is_version_match(f, version::Integer)
    lf = lowercase(f)
    version_str = string(version)
    padded_version = lpad(version_str, 3, '0')
    return _has_version_token(lf, version_str) ||
        (padded_version != version_str && _has_version_token(lf, padded_version))
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

    if isempty(candidates)
        throw(ParserError("Could not find GLORIA satellite directory or ZIP archive in $base_path"))
    end
    found = join([c[1] for c in candidates], ", ")
    throw(ParserError("Could not find GLORIA satellite directory or ZIP archive for version $version in $base_path; candidates found (version token mismatch): $found"))
end

function _inverse_or_zero(v)
    return map(x -> x == 0.0 ? 0.0 : 1.0 / x, v)
end

function _warn_degenerate_io(g::AbstractVector, q::AbstractVector, mats...)
    g_zero = findall(iszero, g)
    q_zero = findall(iszero, q)
    n_negative = sum(count(<(0), m) for m in mats; init = 0)
    isempty(g_zero) && isempty(q_zero) && n_negative == 0 && return nothing
    g_show = g_zero[1:min(end, 10)]
    q_show = q_zero[1:min(end, 10)]
    @warn "ParserWarning: degenerate IO inputs detected" n_zero_row_sums = length(g_zero) zero_row_sum_indices = g_show n_zero_output_sums = length(q_zero) zero_output_sum_indices = q_show n_negative_values = n_negative
    return nothing
end

function _calculate_io_matrices(V, U, Y, VA, Q_SUT::Matrix{Float64})
    g = vec(sum(V, dims = 2))
    T_matrix = _inverse_or_zero(g) .* V

    Z_mat = U * T_matrix
    q = vec(sum(U, dims = 2) + sum(Y, dims = 2))
    A_mat = Z_mat .* _inverse_or_zero(q)'
    VA_mat = VA * T_matrix
    Q_mat = Q_SUT * T_matrix

    _warn_degenerate_io(g, q, V, U, Y, VA, Q_SUT)

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
    parse_gloria(path::String, year::Int; version::Integer = 60, price::AbstractPrice = BasePrice(), country_names::String = "gloria")

Parse raw GLORIA SUT tables for `year` found at `path` (a directory, a year
subdirectory layout, or a direct `.zip` archive), read the Excel readme
metadata (`version`), parse the satellite accounts, and construct a complete
`MRIO` database.

- `version`: GLORIA release number (e.g. `60`), matched against file names.
- `price`: `BasePrice()` (base-price `Markup001(full)` files) or `PPrice()`
  (purchaser-price `Markup005(full)` files, including satellites).
- `country_names`: `"gloria"` uses the readme `Region_acronyms` column,
  anything else uses `Region_names`.
"""
function parse_gloria(path::String, year::Int; version = 60, price = BasePrice(), country_names = "gloria")
    # Resolve the correct base path (handling year subdirectory if needed)
    t_file = gloria_result_file("T", year, version, get_extension(price))
    if isfile(path) && !is_gloria_zip_path(path)
        throw(ParserError("Unsupported GLORIA source file (expected .zip): $path"))
    end
    direct_zip = isfile(path) && is_gloria_zip_path(path)
    base_path = direct_zip ? dirname(path) : resolve_gloria_base_path(path, year, version, t_file)

    sut_path = direct_zip ? path : base_path
    (S, U, Y, VA) = parse_gloria_sut(sut_path; year = year, version = version, price = price)

    # Find the readme file
    gloria_meta_path = find_readme_path(base_path, version, path)

    # Regions (blank cells stay positional so labels cannot shift silently)
    df_regions = DataFrame(XLSX.readtable(gloria_meta_path, "Regions"))
    country_col = Symbol(country_names == "gloria" ? "Region_acronyms" : "Region_names")
    regions = collect_readme_labels(df_regions, country_col, "Regions")
    _require_nonblank_labels(regions, "Regions", "region")

    # Sectors
    df_sectors = DataFrame(XLSX.readtable(gloria_meta_path, "Sectors"))
    sectors = collect_readme_labels(df_sectors, :Sector_names, "Sectors")
    _require_nonblank_labels(sectors, "Sectors", "sector")

    # Value added and final demand
    df_va_fd = DataFrame(XLSX.readtable(gloria_meta_path, "Value added and final demand"))
    va_cats = collect_readme_labels(df_va_fd, :Value_added_names, "Value added and final demand")
    fd_cats = collect_readme_labels(df_va_fd, :Final_demand_names, "Value added and final demand")
    _require_nonblank_labels(va_cats, "Value added and final demand", "value-added")
    _require_nonblank_labels(fd_cats, "Value added and final demand", "final-demand")

    n_regions = length(regions)
    n_sectors = length(sectors)

    n_regions * n_sectors == size(S, 1) ||
        throw(ParserError("Regions/Sectors sheets imply $(n_regions) x $n_sectors = $(n_regions * n_sectors) industries but S has $(size(S, 1)) rows"))
    size(Y, 2) == n_regions * length(fd_cats) ||
        throw(ParserError("Value added and final demand sheet lists $(length(fd_cats)) final-demand categories for $n_regions regions = $(n_regions * length(fd_cats)) columns but Y has $(size(Y, 2)) columns"))
    size(VA, 1) == n_regions * length(va_cats) ||
        throw(ParserError("Value added and final demand sheet lists $(length(va_cats)) value-added categories for $n_regions regions = $(n_regions * length(va_cats)) rows but VA has $(size(VA, 1)) rows"))

    # Satellites
    df_sat = DataFrame(XLSX.readtable(gloria_meta_path, "Satellites"))
    stressors = collect_nonmissing_strings(df_sat, :Sat_indicator)
    sources = collect_nonmissing_strings(df_sat, :Sat_head_indicator)
    units = collect_nonmissing_strings(df_sat, :Sat_unit)
    sat_df = DataFrame(Stressor = stressors, Source = sources, Unit = units)

    # Find and parse satellite files
    sat_path, is_unzipped = find_satellite_path(base_path, version, year)
    q_suffix = gloria_satellite_suffix("T", year, version, price)

    Q_SUT = read_satellites(is_unzipped, sat_path, q_suffix; n_regions = n_regions, n_sectors = n_sectors)
    size(Q_SUT, 1) == nrow(sat_df) ||
        throw(ParserError("Satellites sheet lists $(nrow(sat_df)) indicators but the satellite table has $(size(Q_SUT, 1)) rows"))

    @info "Constructing MRIO"
    return _construct_IO(S, U, Y, VA, regions, sectors, va_cats, fd_cats, Q_SUT, sat_df)
end

function read_satellites(::Type{Zipped}, sat_path, q_suffix; n_regions, n_sectors)
    return open(sat_path, "r") do io
        mmap_data = Mmap.mmap(io)
        sat_zip = za.ZipReader(mmap_data)

        q_bytes = read_entry_in_zip(sat_zip, q_suffix, sat_path)

        return parse(QFile(), q_bytes; n_regions = n_regions, n_sectors = n_sectors)
    end
end

function read_satellites(::Type{Unzipped}, sat_path, q_suffix; n_regions, n_sectors)
    q_file = find_file_in_dir(sat_path, q_suffix)

    q_bytes = mmap_file(q_file)

    return parse(QFile(), q_bytes; n_regions = n_regions, n_sectors = n_sectors)
end

end # Module End
