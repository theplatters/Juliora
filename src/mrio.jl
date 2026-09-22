"""
  MRIO	

Complete global multi-region input-output (MRIO) database structure.

# Fields
- `A::MatrixEntry`: Technical coefficients matrix (intermediate inputs per unit output)
- `T::MatrixEntry`: Intermediate transaction matrix (monetary flows between sectors)
- `VA::MatrixEntry`: Value added matrix (primary inputs by sector)
- `FD::MatrixEntry`: Final demand matrix (consumption, investment, government, exports)
- `L::Union{LeontiefFactorization,Nothing}`: Leontief factorization (lu of I-A), nothing for non-square systems
- `X::SeriesEntry`: Total output vector by sector (a zero vector for non-square systems built via the keyword constructor)
- `env::Union{EnvironmentalExtension,Nothing}`: Environmental impact data, nothing when constructed without environmental data


# Matrix Dimensions
All matrices share consistent country-sector dimensions, typically:
- Rows/Columns: Countries × Sectors (e.g., 189 countries × 26 sectors)
- Environmental: Stressors × (Countries × Sectors)

# Keyword Constructor
`MRIO(; Z, Y, VA)` builds a database from transaction (`Z`), final demand
(`Y`), and value added (`VA`) matrices. It validates that
`size(Y.data, 1) == size(Z.data, 1)` and `size(VA.data, 2) == size(Z.data, 2)`
and throws a `DimensionMismatch` otherwise. Square systems get real `A`/`L`/`X`
values; non-square systems get `L = nothing` and a zero `X` vector (so `A`
equals the raw monetary flows, see `calculate_technical_coefficients`). The
keyword constructor never fabricates environmental data: `env` is always
`nothing`, and `environmental_impact`/`induced_production` throw a clear
`ArgumentError` when the data they need is absent.
"""
struct MRIO
    A::MatrixEntry
    T::MatrixEntry
    VA::MatrixEntry
    FD::MatrixEntry
    L::Union{LeontiefFactorization, Nothing}
    X::SeriesEntry
    env::Union{EnvironmentalExtension, Nothing}
end

"""
	Eora(path::String)

Load and construct complete Eora MRIO database from file directory.

# Arguments
- `path::String`: Directory path containing Eora database files

# Required Files
- `T.txt`: Intermediate transactions matrix
- `VA.txt`: Value added matrix
- `FD.txt`: Final demand matrix
- `labels_T.txt`: Sector labels for T matrix (Country, Industry, Sector)
- `labels_VA.txt`: Value added category labels
- `labels_FD.txt`: Final demand category labels
- Environmental files (Q.txt, labels_Q.txt) for environmental extension

# File Layout Requirement
`VA.txt` columns and `Q.txt` columns must align one-to-one with the rows of
`T.txt` (one column per `T` row, including the `ROW` aggregate region, which
is filtered out during loading). A `DimensionMismatch` is thrown otherwise.

# Returns
- `MRIO`: Complete MRIO database with all matrices and environmental data

# Calculations Performed
- Technical coefficients: A = T ./ x (where x is total output)
- Total output: x = rowSums(T) + rowSums(FD)  
- Leontief inverse: L = inv(I - A)
- Environmental intensities: F ./ x

# Examples
```julia
# Load Eora database for 2017
eora = Eora("data/2017/")

# Access different components
trade_matrix = eora.T
tech_coefficients = eora.A
multipliers = eora.L
co2_impacts = eora.env.F

# Perform analysis
usa_exports = sum_by_country(eora.T; dimension=:rows)
manufacturing_linkages = filter_rows(eora.A, row -> row.Sector == "Manufacturing")
```
"""
function Eora(path::String)
    isdir(path) || throw(ArgumentError("Eora database directory does not exist: $path"))
    required_files = ["T.txt", "labels_T.txt", "VA.txt", "labels_VA.txt", "FD.txt", "labels_FD.txt", "Q.txt", "labels_Q.txt"]
    missing_files = filter(f -> !isfile(joinpath(path, f)), required_files)
    isempty(missing_files) || throw(ArgumentError("Eora database directory $path is missing required files: $(join(missing_files, ", "))"))
    # Parallel file reading using Threads.@spawn
    t_task = Threads.@spawn CSV.read(joinpath(path, "T.txt"), Tables.matrix, header = false)
    t_indices_task = Threads.@spawn @chain read_csv(joinpath(path, "labels_T.txt"), delim = "\t", col_names = false) begin
        @select(CountryCode = Column2, Industry = Column3, Sector = Column4)
    end

    v_task = Threads.@spawn CSV.read(joinpath(path, "VA.txt"), Tables.matrix, header = false)
    v_colnames_task = Threads.@spawn @chain read_csv(joinpath(path, "labels_VA.txt"), delim = "\t", col_names = false) begin
        @select(PrimaryInput = Column2)
    end

    y_task = Threads.@spawn CSV.read(joinpath(path, "FD.txt"), Tables.matrix, header = false)
    y_indices_task = Threads.@spawn @chain read_csv(joinpath(path, "labels_FD.txt"), delim = "\t", col_names = false) begin
        @select(CountryCode = Column2, Industry = Column3, Category = Column4)
    end


    t_indices = fetch(t_indices_task)
    t_matrix = fetch(t_task)

    row_mask = t_indices.CountryCode .!= "ROW"
    t_indices_clean = t_indices[row_mask, :]
    t = MatrixEntry(t_matrix[row_mask, row_mask], t_indices_clean, t_indices_clean)


    y_indices = fetch(y_indices_task)

    y_mask = y_indices.CountryCode .!= "ROW"
    y_indices_clean = y_indices[y_mask, :]
    y = MatrixEntry(fetch(y_task)[row_mask, y_mask], y_indices_clean, t_indices_clean)
    x = calculate_total_output(t.data, y.data)
    a = calculate_technical_coefficients(t, x)
    l = calculate_leontief_factorization(a)
    v = fetch(v_task)
    v_colnames = fetch(v_colnames_task)
    size(v, 2) == size(t_matrix, 1) || throw(
        DimensionMismatch(
            "VA.txt has $(size(v, 2)) columns but T has $(size(t_matrix, 1)) rows; " *
                "VA.txt columns must align one-to-one with T rows (one column per T row, " *
                "including the ROW aggregate region, which is filtered out during loading)"
        )
    )

    return MRIO(
        a,
        t,
        MatrixEntry(v[:, row_mask], t_indices_clean, v_colnames),
        y,
        l,
        SeriesEntry(x, t_indices_clean),
        EnvironmentalExtension(path, x, t_indices_clean, row_mask),
    )
end

"""
    Gloria(path::String, version::Integer, year::Integer)

Load a complete GLORIA MRIO from a cache, ZIP file, or source directory.
For `.jld2`/`.jdl2` cache files, `version` and `year` are ignored because
the cache contains the complete serialized MRIO.
"""
function Gloria(path::String, version::Integer, year::Integer)
    if Parser.is_gloria_cache_path(path)
        return Parser.load_gloria_cache(path)
    elseif isfile(path)
        if Parser.is_gloria_zip_path(path)
            return Parser.parse_gloria(path, year; version = version)
        end
        throw(ArgumentError("Unsupported GLORIA file path (expected .zip, .jld2, or .jdl2): $path"))
    elseif !isdir(path) && !isempty(splitext(path)[2])
        throw(ArgumentError("Unsupported GLORIA file path (expected .zip, .jld2, or .jdl2): $path"))
    end
    return Parser.parse_gloria(path, year; version = version)
end

"""
    Gloria(path::String)

Load a complete GLORIA MRIO from a `.jld2` or `.jdl2` cache file. Source
directories and ZIP files require the three-argument constructor.
"""
function Gloria(path::String)
    Parser.is_gloria_cache_path(path) ||
        throw(ArgumentError("Gloria(path) accepts cache files only (.jld2 or .jdl2); provide version and year for GLORIA source data"))
    return Parser.load_gloria_cache(path)
end


"""
    calculate_technical_coefficients(T::MatrixEntry, x; warn_zero_output::Bool = true)

Compute technical coefficients `A = T ./ x'` (column-wise division by total
output). Zero-output sectors are guarded against `NaN`/`Inf` by dividing by
`1.0` instead of `0.0`, so their coefficients equal the raw monetary flows; a
single `@warn` per call reports how many such sectors exist (disable with
`warn_zero_output = false`, e.g. for deliberate placeholder outputs). The
numeric behavior is unchanged by the warning.
"""
function calculate_technical_coefficients(T::MatrixEntry, x; warn_zero_output::Bool = true)
    n_zero = count(v -> v == 0, x)
    if n_zero > 0 && warn_zero_output
        @warn "calculate_technical_coefficients: $n_zero of $(length(x)) sectors have zero total output; their coefficients equal raw monetary flows (division by zero guarded)"
    end
    return MatrixEntry(T.data ./ replace(x, 0.0 => 1.0)', T.col_indices, T.row_indices)
end
function calculate_total_output(t_data, y_data)
    x = Vector{Float64}(undef, size(t_data, 1))
    @inbounds for i in 1:size(t_data, 1)
        x[i] = sum(view(t_data, i, :)) + sum(view(y_data, i, :))
    end
    return x
end

"""
    MRIO(; Z::MatrixEntry, Y::MatrixEntry, VA::MatrixEntry)

Build an `MRIO` from transaction (`Z`), final demand (`Y`), and value added
(`VA`) matrices. Requires `size(Y.data, 1) == size(Z.data, 1)` (one final
demand row per transaction row) and `size(VA.data, 2) == size(Z.data, 2)` (one
value added column per transaction column); throws a `DimensionMismatch`
otherwise. Square systems get real technical coefficients, a real Leontief
factorization, and a real total output vector. Non-square systems get
`L = nothing` and a zero total output vector (so `A` equals the raw monetary
flows). The environmental extension is always `nothing`; use
`environmental_impact` with an explicit `EnvironmentalExtension`, or build via
`Eora`/`Gloria`/`parse_gloria`, for environmental analysis.
"""
function MRIO(; Z::MatrixEntry, Y::MatrixEntry, VA::MatrixEntry)
    size(Y.data, 1) == size(Z.data, 1) || throw(
        DimensionMismatch(
            "Y has $(size(Y.data, 1)) rows but Z has $(size(Z.data, 1)) rows; " *
                "Y must have one row per Z row"
        )
    )
    size(VA.data, 2) == size(Z.data, 2) || throw(
        DimensionMismatch(
            "VA has $(size(VA.data, 2)) columns but Z has $(size(Z.data, 2)) columns; " *
                "VA must have one column per Z column"
        )
    )
    if size(Z.data, 1) == size(Z.data, 2)
        x = calculate_total_output(Z.data, Y.data)
        a = calculate_technical_coefficients(Z, x)
        l = calculate_leontief_factorization(a)
        return MRIO(
            a,
            Z,
            VA,
            Y,
            l,
            SeriesEntry(x, Z.row_indices),
            nothing
        )
    else
        x = zeros(size(Z.data, 2))
        a = calculate_technical_coefficients(Z, x; warn_zero_output = false)
        return MRIO(
            a,
            Z,
            VA,
            Y,
            nothing,
            SeriesEntry(x, Z.col_indices),
            nothing
        )
    end
end

function Base.getproperty(eora::MRIO, sym::Symbol)
    if sym === :Z
        return getfield(eora, :T)
    elseif sym === :Y
        return getfield(eora, :FD)
    else
        return getfield(eora, sym)
    end
end

# Convenience query functions

"""
    countries(df::DataFrame)
    countries(m::AbstractMatrixEntry)
    countries(s::SeriesEntry)
    countries(env::EnvironmentalExtension)
    countries(mrio::MRIO)

Return a vector of unique country codes/names available in the data or database.
"""
function countries(df::DataFrame)
    if "CountryCode" in names(df)
        return unique(df.CountryCode)
    elseif "Country" in names(df)
        return unique(df.Country)
    else
        return String[]
    end
end

function countries(m::AbstractMatrixEntry)
    c_rows = countries(m.row_indices)
    c_cols = countries(m.col_indices)
    return unique(vcat(c_rows, c_cols))
end

countries(s::SeriesEntry) = countries(s.col_indices)
countries(env::EnvironmentalExtension) = countries(env.F)
countries(mrio::MRIO) = countries(mrio.Z)

const country = countries

"""
    sectors(df::DataFrame)
    sectors(m::AbstractMatrixEntry)
    sectors(s::SeriesEntry)
    sectors(env::EnvironmentalExtension)
    sectors(mrio::MRIO)

Return sector names as an ordered `Vector{String}`.

Sector metadata from both Eora and Gloria is exposed through the canonical
`Sector` column. Duplicate names (for example, the same sector repeated for
several countries) are returned once, in order of first appearance. Missing
metadata values are ignored. For compatibility with generic data frames that
only contain an `Industry` column, that column is used as a fallback.

# Examples
```julia
sector_names = sectors(mrio)
```
"""
function _unique_strings(values)::Vector{String}
    result = String[]
    seen = Set{String}()
    for value in skipmissing(values)
        string_value = string(value)
        if string_value ∉ seen
            push!(seen, string_value)
            push!(result, string_value)
        end
    end
    return result
end

function sectors(df::DataFrame)
    if "Sector" in names(df)
        return _unique_strings(df.Sector)
    elseif "Industry" in names(df)
        return _unique_strings(df.Industry)
    else
        return String[]
    end
end

function sectors(m::AbstractMatrixEntry)
    s_rows = sectors(m.row_indices)
    s_cols = sectors(m.col_indices)
    return unique(vcat(s_rows, s_cols))
end

sectors(s::SeriesEntry) = sectors(s.col_indices)
sectors(env::EnvironmentalExtension) = sectors(env.F)
sectors(mrio::MRIO) = sectors(mrio.Z)

const sector = sectors

"""
    stressors(df::DataFrame)
    stressors(m::AbstractMatrixEntry)
    stressors(s::SeriesEntry)
    stressors(env::EnvironmentalExtension)
    stressors(mrio::MRIO)

Return a vector of unique stressor names available in the database.
"""
function stressors(df::DataFrame)
    if "Stressor" in names(df)
        return unique(df.Stressor)
    else
        return String[]
    end
end

function stressors(m::AbstractMatrixEntry)
    s_rows = stressors(m.row_indices)
    s_cols = stressors(m.col_indices)
    return unique(vcat(s_rows, s_cols))
end

stressors(s::SeriesEntry) = stressors(s.col_indices)
stressors(env::EnvironmentalExtension) = stressors(env.F.row_indices)
stressors(mrio::MRIO) = mrio.env === nothing ? String[] : stressors(mrio.env)

const stressor = stressors
