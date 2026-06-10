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
function custom_gloria_sparse_parser(raw_bytes::Vector{UInt8}, nrows::Int, ncols::Int; delim::Char = ',')
    # Coordinate collection vectors for sparse matrix generation
    I = Int[]
    J = Int[]
    V = Float64[]

    # Pre-allocate using estimated ~15% matrix density to block array resizing overhead
    est_nnz = (nrows * ncols) ÷ 5
    sizehint!(I, est_nnz)
    sizehint!(J, est_nnz)
    sizehint!(V, est_nnz)

    len = length(raw_bytes)
    row_starts = find_row_starts(raw_bytes)
    actual_rows = min(nrows, length(row_starts))

    opts = Parsers.Options(delim=delim)
    delim_byte = UInt8(delim)

    for i in 1:actual_rows
        pos = row_starts[i]

        for j in 1:ncols
            # Allocation-free delimiter skipping
            while pos <= len && (
                    raw_bytes[pos] == delim_byte ||
                        raw_bytes[pos] == 0x0a || raw_bytes[pos] == 0x0d
                )
                pos += 1
            end

            if pos > len
                break
            end

            res = Parsers.xparse(Float64, raw_bytes, pos, len, opts)

            if Parsers.ok(res.code)
                val = res.val
                if val != 0.0  # ONLY keep true non-zero coefficients
                    push!(I, i)
                    push!(J, j)
                    push!(V, val)
                end
                pos += res.tlen
            else
                pos += 1
            end
        end
    end

    @info "Finished parsing matrix into highly compressed sparse format"
    return sparse(I, J, V, nrows, ncols)
end

"""
    read_csv_to_sparse_matrix(zip_reader, filename)

Extracts compressed files into byte vectors and initializes the sparse compiler.
"""
function read_csv_to_sparse_matrix(zip_reader::za.ZipReader, filename::String)
    raw_data = za.zip_readentry(zip_reader, filename)
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
    return custom_gloria_sparse_parser(raw_data, nrows, ncols; delim=delim)
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

    io = open(gloria_path, "r")
    mmap_data = Mmap.mmap(io)
    gloria_zip = za.ZipReader(mmap_data)

    sut_matrices = Dict{String, SparseMatrixCSC{Float64, Int}}()

    for (k, v) in gloria_mrio_files
        println("Streaming and parsing $k...")
        # Fixed bug: Captured the return values inside your state tracking dictionary
        sut_matrices[k] = read_csv_to_sparse_matrix(gloria_zip, v)
        @info "$k parsed successfully"

        # Explicitly clear space after every major structural ingestion loop
        GC.gc()
    end

    close(io)
    GC.gc()

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
