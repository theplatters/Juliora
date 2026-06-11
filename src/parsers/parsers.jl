module PyMRIOParser

using CSV
using DataFrames
import ZipArchives as za
using LinearAlgebra
using Mmap
using Parsers
using SparseArrays # Crucial for low-RAM high-performance processing

export ParserError, ParserWarning, parse_gloria, parse_gloria_sut, __construct_IO

# Custom Exceptions
struct ParserError <: Exception
    msg::String
end
struct ParserWarning <: Exception
    msg::String
end

# Global Constants matching Python's IDX_NAMES structure
const IDX_NAMES = Dict(
    :T_col => ["region", "system", "sector"],
    :T_row => ["region", "system", "sector"],
    :Y_col2 => ["region", "category"],
    :F_row_cat_unit => ["stressor", "compartment", "unit"],
    :VA_row_region => ["region", "inputtype"],
    :unit => ["unit"]
)

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
        if raw_bytes[pos] == 0x0a # '\n'
            push!(row_starts, pos + 1)
        end
    end
    return row_starts
end

"""
    custom_gloria_sparse_parser(raw_bytes::Vector{UInt8}, nrows::Int, ncols::Int; delim::Char = ',')

Streams line-by-line using byte tokens and directly compiles a SparseMatrixCSC.
Bypasses intermediate dense grid allocations completely.
"""
@inline function is_delim_or_newline(b::UInt8, delim_byte::UInt8)
    return b == delim_byte || b == 0x0a || b == 0x0d
end

@inline function try_skip_zero(raw_bytes::Vector{UInt8}, pos::Int, len::Int, delim_byte::UInt8)
    if pos <= len && raw_bytes[pos] == 0x30
        if pos == len || is_delim_or_newline(raw_bytes[pos + 1], delim_byte)
            return 1
        elseif pos + 2 <= len && raw_bytes[pos + 1] == 0x2e && raw_bytes[pos + 2] == 0x30 && (pos + 2 == len || is_delim_or_newline(raw_bytes[pos + 3], delim_byte))
            return 3
        end
    end
    return 0
end

function parse_chunk_range!(
        I::Vector{Int}, J::Vector{Int}, V::Vector{Float64},
        raw_bytes::Vector{UInt8}, len::Int, row_starts::Vector{Int},
        c_start::Int, c_end::Int, ncols::Int, delim_byte::UInt8, opts::Parsers.Options
    )
    idx = 1
    max_len = length(I)

    for i in c_start:c_end
        pos = row_starts[i]
        for j in 1:ncols
            # Skip delimiters
            while pos <= len && is_delim_or_newline(raw_bytes[pos], delim_byte)
                pos += 1
            end

            if pos > len
                break
            end

            skip_len = try_skip_zero(raw_bytes, pos, len, delim_byte)
            if skip_len > 0
                pos += skip_len
                continue
            end

            res = Parsers.xparse(Float64, raw_bytes, pos, len, opts)
            if Parsers.ok(res.code)
                val = res.val
                if val != 0.0
                    if idx > max_len
                        max_len = max(10, max_len * 2)
                        resize!(I, max_len)
                        resize!(J, max_len)
                        resize!(V, max_len)
                    end
                    @inbounds I[idx] = i
                    @inbounds J[idx] = j
                    @inbounds V[idx] = val
                    idx += 1
                end
                pos += res.tlen
            else
                pos += 1
            end
        end
    end

    resize!(I, idx - 1)
    resize!(J, idx - 1)
    resize!(V, idx - 1)

    return I, J, V
end

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

function custom_gloria_sparse_parser(::Parallel, raw_bytes::Vector{UInt8}, len::Int, row_starts::Vector{Int}, actual_rows::Int, nrows::Int, ncols::Int, delim::Char, n_threads::Int)
    delim_byte = UInt8(delim)
    opts = Parsers.Options(delim = delim)

    n_chunks = min(actual_rows, 4 * n_threads)
    chunks = Vector{Tuple{Int, Int}}(undef, n_chunks)
    chunk_size = div(actual_rows, n_chunks)
    rem_rows = rem(actual_rows, n_chunks)

    start_idx = 1
    for t in 1:n_chunks
        end_idx = start_idx + chunk_size - 1
        if t <= rem_rows
            end_idx += 1
        end
        chunks[t] = (start_idx, end_idx)
        start_idx = end_idx + 1
    end

    thread_I = Vector{Vector{Int}}(undef, n_chunks)
    thread_J = Vector{Vector{Int}}(undef, n_chunks)
    thread_V = Vector{Vector{Float64}}(undef, n_chunks)

    for t in 1:n_chunks
        c_start, c_end = chunks[t]
        chunk_rows = c_end - c_start + 1
        est_nnz_t = (chunk_rows * ncols) ÷ 5
        thread_I[t] = Vector{Int}(undef, est_nnz_t)
        thread_J[t] = Vector{Int}(undef, est_nnz_t)
        thread_V[t] = Vector{Float64}(undef, est_nnz_t)
    end

    tasks = Vector{Task}(undef, n_chunks)
    for t in 1:n_chunks
        tasks[t] = Threads.@spawn begin
            c_start, c_end = chunks[t]
            chunk_rows = c_end - c_start + 1
            if chunk_rows > 0
                parse_chunk_range!(thread_I[t], thread_J[t], thread_V[t], raw_bytes, len, row_starts, c_start, c_end, ncols, delim_byte, opts)
            else
                resize!(thread_I[t], 0)
                resize!(thread_J[t], 0)
                resize!(thread_V[t], 0)
            end
        end
    end

    for t in 1:n_chunks
        fetch(tasks[t])
    end

    total_nnz = sum(length, thread_I)
    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Vector{Float64}(undef, total_nnz)

    curr = 1
    for t in 1:n_chunks
        len_t = length(thread_I[t])
        if len_t > 0
            copyto!(I, curr, thread_I[t], 1, len_t)
            copyto!(J, curr, thread_J[t], 1, len_t)
            copyto!(V, curr, thread_V[t], 1, len_t)
            curr += len_t
        end
    end

    @info "Finished parsing matrix into highly compressed sparse format"
    return construct_sparse_from_row_major(I, J, V, nrows, ncols)
end

"""
    construct_sparse_from_row_major(I, J, V, nrows, ncols)

Converts row-major sorted coordinates (I, J, V) directly into a SparseMatrixCSC without sorting,
halving the allocations and speeding up execution compared to generic `sparse(I, J, V)`.
"""
function construct_sparse_from_row_major(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64}, nrows::Int, ncols::Int)
    if nrows < 0 || ncols < 0
        throw(ArgumentError("Dimensions must be non-negative"))
    end
    len_nnz = length(I)

    col_counts = zeros(Int, ncols)
    @inbounds for k in 1:len_nnz
        col_counts[J[k]] += 1
    end

    colptr = Vector{Int}(undef, ncols + 1)
    colptr[1] = 1
    @inbounds for j in 1:ncols
        colptr[j + 1] = colptr[j] + col_counts[j]
    end

    next_idx = copy(colptr)
    rowval = Vector{Int}(undef, len_nnz)
    nzval = Vector{Float64}(undef, len_nnz)

    @inbounds for k in 1:len_nnz
        r = I[k]
        c = J[k]
        val = V[k]
        idx = next_idx[c]
        rowval[idx] = r
        nzval[idx] = val
        next_idx[c] += 1
    end

    return SparseMatrixCSC(nrows, ncols, colptr, rowval, nzval)
end

"""
    read_csv_to_sparse_matrix(zip_reader, filename)

Extracts compressed files into byte vectors and initializes the sparse compiler.
"""
function read_csv_to_sparse_matrix(zip_reader::za.ZipReader, filename::String)
    raw_data = za.zip_readentry(zip_reader, filename)
    return parse_raw_bytes_to_sparse_matrix(raw_data)
end

"""
    read_csv_to_sparse_matrix(filepath::String)

Reads a CSV file directly from disk (using memory-mapping) and compiles it into a sparse matrix.
"""
function read_csv_to_sparse_matrix(filepath::String)
    if !isfile(filepath)
        throw(ParserError("File not found: $filepath"))
    end

    io = open(filepath, "r")
    try
        raw_data = Mmap.mmap(io, Vector{UInt8})
        return parse_raw_bytes_to_sparse_matrix(raw_data)
    finally
        close(io)
    end
end

"""
    parse_raw_bytes_to_sparse_matrix(raw_data::Vector{UInt8})

Internal helper to compile a raw byte vector of CSV data into a SparseMatrixCSC.
"""
function parse_raw_bytes_to_sparse_matrix(raw_data::Vector{UInt8})
    len = length(raw_data)

    if len == 0
        return dropzeros(sparse(Int[], Int[], Float64[], 0, 0))
    end

    # Detect the delimiter (either ',' or ';') by inspecting the first line
    delim = ','
    @inbounds for pos in 1:len
        b = raw_data[pos]
        if b == 0x3b # ';'
            delim = ';'
            break
        elseif b == 0x0a # '\n'
            break
        end
    end

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

    @info "Started parsing matrix: Size specified as $nrows x $ncols"
    return custom_gloria_sparse_parser(raw_data, nrows, ncols; delim = delim)
end

"""
    parse_gloria_sut(path::String, year::Int; version = 60, price::AbstractPrice = BasePrice())
"""
function parse_gloria_sut(path::String, year::Int; version = 60, price::AbstractPrice = BasePrice())
    extension = get_extention(price)

    gloria_mrio_files = (
        "T" => "20260121_120secMother_AllCountries_002_T-Results_$(year)_0$(version)_$(extension).csv",
        "Y" => "20260121_120secMother_AllCountries_002_Y-Results_$(year)_0$(version)_$(extension).csv",
        "VA" => "20260121_120secMother_AllCountries_002_V-Results_$(year)_0$(version)_$(extension).csv",
    )

    gloria_path = joinpath(path, "GLORIA_MRIOs_$(version)_$year.zip")

    # If the zip file does not exist, but we detect files/directories that look unzipped, fallback to unzipped.
    if !isfile(gloria_path)
        first_file = gloria_mrio_files[1].second
        if isdir(path) && (isfile(joinpath(path, first_file)) || isdir(joinpath(path, "GLORIA_MRIOs_$(version)_$year")))
            @info "GLORIA zip file not found, but unzipped directory exists. Fallback to unzipped parsing."
            return parse_gloria_sut(path, year, true; version = version, price = price)
        end
    end

    io = open(gloria_path, "r")
    mmap_data = Mmap.mmap(io)
    gloria_zip = za.ZipReader(mmap_data)

    sut_matrices = Dict{String, SparseMatrixCSC{Float64, Int}}()

    for (k, v) in gloria_mrio_files
        println("Streaming and parsing $k...")
        sut_matrices[k] = read_csv_to_sparse_matrix(gloria_zip, v)
        @info "$k parsed successfully"

        # Explicitly clear space after every major structural ingestion loop
    end

    close(io)

    return sut_matrices
end

"""
    parse_gloria_sut(path::String, year::Int, is_unzipped::Bool; version = 60, price::AbstractPrice = BasePrice())

Reads GLORIA SUT matrices from an unzipped directory.
If `is_unzipped` is true, this method expects the files to be located in `path` or `path/GLORIA_MRIOs_\$(version)_\$(year)`.
"""
function parse_gloria_sut(path::String, year::Int, is_unzipped::Bool; version = 60, price::AbstractPrice = BasePrice())
    if !is_unzipped
        return parse_gloria_sut(path, year; version = version, price = price)
    end

    extension = get_extention(price)

    gloria_mrio_files = (
        "T" => "20260121_120secMother_AllCountries_002_T-Results_$(year)_0$(version)_$(extension).csv",
        "Y" => "20260121_120secMother_AllCountries_002_Y-Results_$(year)_0$(version)_$(extension).csv",
        "VA" => "20260121_120secMother_AllCountries_002_V-Results_$(year)_0$(version)_$(extension).csv",
    )

    # Determine unzipped directory path
    dir_path = path
    if !isdir(dir_path)
        throw(ParserError("Path is not a directory: $dir_path"))
    end

    first_file = gloria_mrio_files[1].second
    if !isfile(joinpath(dir_path, first_file))
        subdir = joinpath(path, "GLORIA_MRIOs_$(version)_$year")
        if isdir(subdir) && isfile(joinpath(subdir, first_file))
            dir_path = subdir
        else
            throw(ParserError("Could not find GLORIA SUT files in $path or $subdir"))
        end
    end

    sut_matrices = Dict{String, SparseMatrixCSC{Float64, Int}}()

    for (k, v) in gloria_mrio_files
        filepath = joinpath(dir_path, v)
        println("Streaming and parsing $k from $filepath...")
        sut_matrices[k] = read_csv_to_sparse_matrix(filepath)
        @info "$k parsed successfully"
    end


    return sut_matrices
end

"""
    __construct_IO(data_sut::Dict{String, SparseMatrixCSC{Float64, Int}}; construct="B")

Runs low-RAM matrix transformations directly on sparse structures.
"""
function __construct_IO(data_sut::Dict{String, SparseMatrixCSC{Float64, Int}}; construct = "B")
    V = data_sut["VA"]
    U = data_sut["T"]

    # Summing across a sparse matrix row dimensions returns a predictable dense 2D column array
    g = vec(sum(V, dims = 2))
    g_inv = map(x -> x == 0.0 ? 0.0 : 1.0 / x, g)

    if construct == "B"
        # Row-wise vector broadcasting (.*) against a SparseMatrixCSC scales
        # non-zero elements in-place while perfectly maintaining original matrix structure.
        T_matrix = g_inv .* V

        # Sparse * Sparse Matrix Multiplication (Native, extremely quick, runs in O(nnz) RAM)
        A_mat = U * T_matrix

        return Dict{String, SparseMatrixCSC{Float64, Int}}("A" => A_mat)
    else
        throw(ArgumentError("Unsupported construction type: $construct"))
    end
end

function parse_gloria(path::String, year::Int; version = 59, price = "bp", country_names = "gloria", construct = "B")
    # Implementation wrapper...
end

end # Module End
