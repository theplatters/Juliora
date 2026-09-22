abstract type AbstractMatrixEntry end
"""
	MatrixEntry

A structure that combines a numerical matrix with labeled row and column indices,
optimized for economic input-output analysis.

# Fields
- `data::Matrix{Float64}`: The numerical matrix data
- `col_indices::DataFrame`: DataFrame containing column labels and metadata
- `row_indices::DataFrame`: DataFrame containing row labels and metadata  
- `row_lookup::Dict{NamedTuple, Int}`: Hash table for fast row index lookups
- `col_lookup::Dict{NamedTuple, Int}`: Hash table for fast column index lookups

# Constructor
	MatrixEntry(data, col_indices, row_indices)

Creates a MatrixEntry with automatic validation and lookup table generation.

# Arguments
- `data`: Matrix of numerical values
- `col_indices`: DataFrame with column labels (one row per column)
- `row_indices`: DataFrame with row labels (one row per row)

# Examples
```jldoctest
julia> using DataFrames

julia> data = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3 rows × 2 columns
3×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0
 5.0  6.0

julia> row_df = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Agr", "Man", "Ser"]);

julia> col_df = DataFrame(Country=["USA", "CHN"], Sector=["Agr", "Man"]);

julia> matrix_entry = MatrixEntry(data, col_df, row_df);

julia> size(matrix_entry.data)
(3, 2)

julia> matrix_entry.row_indices.Country
3-element Vector{String}:
 "USA"
 "CHN"
 "DEU"
```
"""
mutable struct MatrixEntry{T <: AbstractMatrix{Float64}} <: AbstractMatrixEntry
    data::T
    col_indices::DataFrame
    row_indices::DataFrame
    row_lookup::Dict{NamedTuple, Int}
    col_lookup::Dict{NamedTuple, Int}
end

function MatrixEntry(data::T, col_indices::DataFrame, row_indices::DataFrame) where {T <: AbstractMatrix{Float64}}
    expected_size = (nrow(row_indices), nrow(col_indices))
    if size(data) != expected_size
        throw(DimensionMismatch("data $(size(data)) dimensions must match index DataFrames $expected_size"))
    end
    row_lookup = _build_lookup(row_indices, "row")
    col_lookup = _build_lookup(col_indices, "column")
    return MatrixEntry{T}(data, col_indices, row_indices, row_lookup, col_lookup)
end

function MatrixEntry(data::AbstractMatrix, col_indices::DataFrame, row_indices::DataFrame)
    data_float = convert(AbstractMatrix{Float64}, data)
    return MatrixEntry(data_float, col_indices, row_indices)
end

function MatrixEntry(data::AbstractMatrix, col_indices, row_indices)
    return MatrixEntry(data, safe_dataframe(col_indices), safe_dataframe(row_indices))
end


"""
	Base.getindex(m::MatrixEntry, row_key::NamedTuple, col_key::NamedTuple)

Retrieve a single value from the matrix using labeled row and column keys.

# Arguments
- `m::MatrixEntry`: The matrix entry to index
- `row_key::NamedTuple`: Named tuple identifying the row (e.g., `(Country="USA", Sector="Manufacturing")`)
- `col_key::NamedTuple`: Named tuple identifying the column

# Returns
- `Float64`: The value at the specified row and column

# Throws
- `BoundsError`: If the row or column key is not found in the indices

# Examples
```jldoctest
julia> using DataFrames

julia> data = [1.0 2.0; 3.0 4.0; 5.0 6.0];

julia> row_df = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Agr", "Man", "Ser"]);

julia> col_df = DataFrame(Country=["USA", "CHN"], Sector=["Agr", "Man"]);

julia> matrix_entry = MatrixEntry(data, col_df, row_df);

julia> matrix_entry[(Country="USA", Sector="Agr"), (Country="USA", Sector="Agr")]
1.0

julia> matrix_entry[(Country="CHN", Sector="Man"), (Country="CHN", Sector="Man")]
4.0
```
"""
function Base.getindex(m::AbstractMatrixEntry, row_key::NamedTuple, col_key::NamedTuple)
    row_idx = get(m.row_lookup, row_key, nothing)
    col_idx = get(m.col_lookup, col_key, nothing)

    isnothing(row_idx) && throw(BoundsError(m, row_key))
    isnothing(col_idx) && throw(BoundsError(m, col_key))

    return m.data[row_idx, col_idx]
end

"""
	Base.getindex(m::AbstractMatrixEntry, row_key::NamedTuple, ::Colon)

Retrieve a subset of rows matching a partial or full row key, and all columns.

# Arguments
- `m::AbstractMatrixEntry`: The matrix entry to index
- `row_key::NamedTuple`: Named tuple identifying matching row(s) (e.g., `(Country="USA", Sector="Manufacturing")` or `(Country="USA",)`)
- `::Colon`: Colon indicating all columns

# Returns
- `SeriesEntry` or `MatrixEntry`: If only one row matches, returns a `SeriesEntry`. Otherwise, returns a `MatrixEntry` containing the matching rows.

# Throws
- `BoundsError`: If no rows match the key
"""
function Base.getindex(m::AbstractMatrixEntry, row_key::NamedTuple, ::Colon)
    # Find all rows that match the partial key
    row_indices_set = Set{Int64}()
    for (full_key, idx) in m.row_lookup
        if _partial_key_match(full_key, row_key)
            push!(row_indices_set, idx)
        end
    end

    isempty(row_indices_set) && throw(BoundsError(m, row_key))
    row_indices = sort(collect(row_indices_set))
    if length(row_indices) == 1
        return SeriesEntry(m.data[row_indices[1], :], m.col_indices)
    else
        return MatrixEntry(m.data[row_indices, :], m.col_indices, m.row_indices[row_indices, :])
    end
end

"""
	Base.getindex(m::AbstractMatrixEntry, ::Colon, col_key::NamedTuple)

Retrieve a subset of columns matching a partial or full column key, and all rows.

# Arguments
- `m::AbstractMatrixEntry`: The matrix entry to index
- `::Colon`: Colon indicating all rows
- `col_key::NamedTuple`: Named tuple identifying matching column(s) (e.g., `(Country="USA", Sector="Manufacturing")` or `(Country="USA",)`)

# Returns
- `SeriesEntry` or `MatrixEntry`: If only one column matches, returns a `SeriesEntry`. Otherwise, returns a `MatrixEntry` containing the matching columns.

# Throws
- `BoundsError`: If no columns match the key
"""
function Base.getindex(m::AbstractMatrixEntry, ::Colon, col_key::NamedTuple)
    # Find all columns that match the partial key
    col_indices_set = Set{Int64}()
    for (full_key, idx) in m.col_lookup
        if _partial_key_match(full_key, col_key)
            push!(col_indices_set, idx)
        end
    end

    isempty(col_indices_set) && throw(BoundsError(m, col_key))

    col_indices = sort(collect(col_indices_set))
    if length(col_indices) == 1
        return SeriesEntry(m.data[:, col_indices[1]], m.row_indices)
    else
        return MatrixEntry(m.data[:, col_indices], m.col_indices[col_indices, :], m.row_indices)
    end
end

"""
	Base.getindex(m::AbstractMatrixEntry, ::Colon, col_key::AbstractArray{T}) where T <: NamedTuple

Retrieve a subset of columns matching any of the partial or full column keys in the array, and all rows.

# Arguments
- `m::AbstractMatrixEntry`: The matrix entry to index
- `::Colon`: Colon indicating all rows
- `col_key::AbstractArray{T}`: Array of named tuples identifying matching column(s)

# Returns
- `SeriesEntry` or `MatrixEntry`: If only one column matches, returns a `SeriesEntry`. Otherwise, returns a `MatrixEntry` containing the matching columns.

# Throws
- `BoundsError`: If no columns match any of the keys in the array
"""
function Base.getindex(m::AbstractMatrixEntry, ::Colon, col_key::AbstractArray{T}) where {T <: NamedTuple}
    col_indices_set = Set{Int64}()
    for key in col_key
        for (full_key, idx) in m.col_lookup
            if _partial_key_match(full_key, key)
                push!(col_indices_set, idx)
            end
        end
    end


    isempty(col_indices_set) && throw(BoundsError(m, col_key))

    col_indices = sort(collect(col_indices_set))
    if length(col_indices) == 1
        return SeriesEntry(m.data[:, col_indices[1]], m.row_indices)
    else
        return MatrixEntry(m.data[:, col_indices], m.col_indices[col_indices, :], m.row_indices)
    end
end

"""
	Base.getindex(m::AbstractMatrixEntry, row_key::AbstractArray{T}, ::Colon) where T <: NamedTuple

Retrieve a subset of rows matching any of the partial or full row keys in the array, and all columns.

# Arguments
- `m::AbstractMatrixEntry`: The matrix entry to index
- `row_key::AbstractArray{T}`: Array of named tuples identifying matching row(s)
- `::Colon`: Colon indicating all columns

# Returns
- `SeriesEntry` or `MatrixEntry`: If only one row matches, returns a `SeriesEntry`. Otherwise, returns a `MatrixEntry` containing the matching rows.

# Throws
- `BoundsError`: If no rows match any of the keys in the array
"""
function Base.getindex(m::AbstractMatrixEntry, row_key::AbstractArray{T}, ::Colon) where {T <: NamedTuple}
    row_indices_set = Set{Int64}()
    for key in row_key
        for (full_key, idx) in m.row_lookup
            if _partial_key_match(full_key, key)
                push!(row_indices_set, idx)
            end
        end
    end

    isempty(row_indices_set) && throw(BoundsError(m, row_key))

    row_indices = sort(collect(row_indices_set))
    if length(row_indices) == 1
        return SeriesEntry(m.data[row_indices[1], :], m.col_indices)
    else
        return MatrixEntry(m.data[row_indices, :], m.col_indices, m.row_indices[row_indices, :])
    end
end


"""
	Base.getindex(m::AbstractMatrixEntry, row_mask::AbstractVector{Bool}, col_mask::AbstractVector{Bool})

Filter both rows and columns using boolean masks, returning a new MatrixEntry.

# Arguments
- `m::MatrixEntry`: The matrix entry to filter
- `row_mask::AbstractVector{Bool}`: Boolean vector for row selection (length must match number of rows)
- `col_mask::AbstractVector{Bool}`: Boolean vector for column selection (length must match number of columns)

# Returns
- `MatrixEntry`: New MatrixEntry with filtered data and corresponding indices

# Examples
```jldoctest
julia> using DataFrames

julia> data = [1.0 2.0; 3.0 4.0; 5.0 6.0];

julia> row_df = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Agr", "Man", "Ser"]);

julia> col_df = DataFrame(Country=["USA", "CHN"], Sector=["Agr", "Man"]);

julia> matrix_entry = MatrixEntry(data, col_df, row_df);

julia> usa_rows = matrix_entry.row_indices.Country .== "USA";

julia> agr_cols = matrix_entry.col_indices.Sector .== "Agr";

julia> filtered = matrix_entry[usa_rows, agr_cols];

julia> size(filtered.data)
(1, 1)

julia> filtered.data[1, 1]
1.0
```
"""
function Base.getindex(m::AbstractMatrixEntry, row_mask::AbstractVector{Bool}, col_mask::AbstractVector{Bool})
    if length(row_mask) != size(m.data, 1)
        throw(DimensionMismatch("row mask length must match number of rows"))
    end
    if length(col_mask) != size(m.data, 2)
        throw(DimensionMismatch("column mask length must match number of columns"))
    end

    new_data = m.data[row_mask, col_mask]
    new_row_indices = m.row_indices[row_mask, :]
    new_col_indices = m.col_indices[col_mask, :]

    return MatrixEntry(new_data, new_col_indices, new_row_indices)
end

"""
	Base.getindex(m::AbstractMatrixEntry, row_mask::AbstractVector{Bool}, ::Colon)

Filter rows using a boolean mask while keeping all columns.

# Arguments
- `m::AbstractMatrixEntry`: The matrix entry to filter
- `row_mask::AbstractVector{Bool}`: Boolean vector for row selection
- `::Colon`: Indicates all columns should be kept

# Returns
- `MatrixEntry`: New MatrixEntry with filtered rows and all original columns

# Examples
```jldoctest
julia> using DataFrames

julia> data = [1.0 2.0; 3.0 4.0; 5.0 6.0];

julia> row_df = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Agr", "Man", "Ser"]);

julia> col_df = DataFrame(Country=["USA", "CHN"], Sector=["Agr", "Man"]);

julia> matrix_entry = MatrixEntry(data, col_df, row_df);

julia> developed_countries = ["USA", "DEU"];

julia> developed_mask = [country in developed_countries for country in matrix_entry.row_indices.Country];

julia> developed_data = matrix_entry[developed_mask, :];

julia> size(developed_data.data)
(2, 2)

julia> developed_data.row_indices.Country
2-element Vector{String}:
 "USA"
 "DEU"
```
"""
function Base.getindex(m::AbstractMatrixEntry, row_mask::AbstractVector{Bool}, ::Colon)
    if length(row_mask) != size(m.data, 1)
        throw(DimensionMismatch("row mask length must match number of rows"))
    end

    new_data = m.data[row_mask, :]
    new_row_indices = m.row_indices[row_mask, :]

    return MatrixEntry(new_data, m.col_indices, new_row_indices)
end

"""
	Base.getindex(m::MatrixEntry, ::Colon, col_mask::AbstractVector{Bool})

Filter columns using a boolean mask while keeping all rows.

# Arguments
- `m::MatrixEntry`: The matrix entry to filter
- `::Colon`: Indicates all rows should be kept
- `col_mask::AbstractVector{Bool}`: Boolean vector for column selection

# Returns
- `MatrixEntry`: New MatrixEntry with all original rows and filtered columns

# Examples
```jldoctest
julia> using DataFrames

julia> data = [1.0 2.0; 3.0 4.0; 5.0 6.0];

julia> row_df = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Agr", "Man", "Ser"]);

julia> col_df = DataFrame(Country=["USA", "CHN"], Sector=["Agr", "Man"]);

julia> matrix_entry = MatrixEntry(data, col_df, row_df);

julia> usa_cols = matrix_entry.col_indices.Country .== "USA";

julia> usa_data = matrix_entry[:, usa_cols];

julia> size(usa_data.data)
(3, 1)

julia> usa_data.col_indices.Country
1-element Vector{String}:
 "USA"
```
"""
function Base.getindex(m::AbstractMatrixEntry, ::Colon, col_mask::AbstractVector{Bool})
    if length(col_mask) != size(m.data, 2)
        throw(DimensionMismatch("column mask length must match number of columns"))
    end

    new_data = m.data[:, col_mask]
    new_col_indices = m.col_indices[col_mask, :]

    return MatrixEntry(new_data, new_col_indices, m.row_indices)
end

# Boolean indexing with functions on row/column indices

"""
	filter_rows(m::AbstractMatrixEntry, condition_func)

Filter rows based on a condition function applied to row indices.

# Arguments
- `m::AbstractMatrixEntry`: The matrix entry to filter
- `condition_func`: Function that takes a NamedTuple (row) and returns Bool

# Returns
- `MatrixEntry`: New MatrixEntry with filtered rows

# Examples
```jldoctest
julia> using DataFrames

julia> data = [1.0 2.0; 3.0 4.0; 5.0 6.0];

julia> row_df = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Agr", "Man", "Ser"]);

julia> col_df = DataFrame(Country=["USA", "CHN"], Sector=["Agr", "Man"]);

julia> matrix_entry = MatrixEntry(data, col_df, row_df);

julia> developed = filter_rows(matrix_entry, row -> row.Country in ["USA", "DEU"]);

julia> size(developed.data)
(2, 2)

julia> developed.row_indices.Country
2-element Vector{String}:
 "USA"
 "DEU"

julia> manufacturing = filter_rows(matrix_entry, row -> row.Sector == "Man");

julia> size(manufacturing.data)
(1, 2)
```
"""
function filter_rows(m::AbstractMatrixEntry, condition_func)
    row_mask = [condition_func(NamedTuple(row)) for row in eachrow(m.row_indices)]
    return m[row_mask, :]
end

"""
	filter_cols(m::MatrixEntry, condition_func)

Filter columns based on a condition function applied to column indices.

# Arguments
- `m::MatrixEntry`: The matrix entry to filter
- `condition_func`: Function that takes a NamedTuple (column) and returns Bool

# Returns
- `MatrixEntry`: New MatrixEntry with filtered columns

# Examples
```jldoctest
julia> using DataFrames

julia> data = [1.0 2.0; 3.0 4.0; 5.0 6.0];

julia> row_df = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Agr", "Man", "Ser"]);

julia> col_df = DataFrame(Country=["USA", "CHN"], Sector=["Agr", "Man"]);

julia> matrix_entry = MatrixEntry(data, col_df, row_df);

julia> china_cols = filter_cols(matrix_entry, col -> col.Country == "CHN");

julia> size(china_cols.data)
(3, 1)

julia> china_cols.col_indices.Country
1-element Vector{String}:
 "CHN"
```
"""
function filter_cols(m::AbstractMatrixEntry, condition_func)
    col_mask = [condition_func(NamedTuple(row)) for row in eachrow(m.col_indices)]
    return m[:, col_mask]
end

function Base.filter(fun::Function, m::AbstractMatrixEntry; dims::Int = 1)
    if dims != 1 && dims != 2
        throw(ArgumentError("dims must be 1 or 2, got $dims"))
    end
    if dims == 1
        return filter_rows(m, fun)
    else
        return filter_cols(m, fun)
    end
end

struct GroupedMatrixEntry <: AbstractMatrixEntry
    original::MatrixEntry
    grouped::GroupedDataFrame
    dims::Int
    cols::Union{Symbol, Vector{Symbol}}
end

function groupby(m::MatrixEntry, cols; dims::Int = 1)
    if dims == 1
        grouped = DataFrames.groupby(m.row_indices, cols)
    elseif dims == 2
        grouped = DataFrames.groupby(m.col_indices, cols)
    else
        throw(ArgumentError("dims must be 1 (rows) or 2 (columns)"))
    end

    return GroupedMatrixEntry(m, grouped, dims, cols)
end

function aggregate(gm::GroupedMatrixEntry, func::Function = sum)
    ind = groupindices(gm.grouped)
    groups = unique(ind)
    if isempty(groups)
        # Aggregating zero groups: return a correctly-typed empty result that
        # preserves the container invariants instead of throwing on reduce.
        if gm.dims == 1
            new_data_matrix = zeros(Float64, 0, size(gm.original.data, 2))
            new_row_indices = select(gm.original.row_indices, groupcols(gm.grouped))[Int[], :]
            return MatrixEntry(new_data_matrix, gm.original.col_indices, new_row_indices)
        else
            new_data_matrix = zeros(Float64, size(gm.original.data, 1), 0)
            new_col_indices = select(gm.original.col_indices, groupcols(gm.grouped))[Int[], :]
            return MatrixEntry(new_data_matrix, new_col_indices, gm.original.row_indices)
        end
    end
    if gm.dims == 1
        new_data_matrix = reduce(vcat, [func(gm.original.data[ind .== g, :], dims = 1) for g in groups])
        new_row_indices = unique(select(gm.original.row_indices, groupcols(gm.grouped)))
        return MatrixEntry(new_data_matrix, gm.original.col_indices, new_row_indices)
    else
        new_data_matrix = reduce(hcat, [func(gm.original.data[:, ind .== g], dims = 2) for g in groups])
        new_col_indices = unique(select(gm.original.col_indices, groupcols(gm.grouped)))

        return MatrixEntry(new_data_matrix, new_col_indices, gm.original.row_indices)
    end
end

"""
    _check_invariants(m::MatrixEntry)

Internal helper: verify `size(m.data) == (nrow(m.row_indices), nrow(m.col_indices))`
after a mutating operation, so aliasing or lookup bugs surface immediately.
"""
function _check_invariants(m::MatrixEntry)
    expected = (nrow(m.row_indices), nrow(m.col_indices))
    if size(m.data) != expected
        throw(ErrorException("MatrixEntry invariant violated: data size $(size(m.data)) does not match index dimensions $expected"))
    end
    return true
end

function drop(m::AbstractMatrixEntry, indices::T; dims = 1) where {T <: NamedTuple}
    if dims != 1 && dims != 2
        throw(ArgumentError("dims must be 1 or 2, got $dims"))
    end

    keep = trues(size(m.data, dims))

    matched = false
    lookup = dims == 1 ? m.row_lookup : m.col_lookup
    for (full_key, idx) in lookup
        if _partial_key_match(full_key, indices)
            keep[idx] = false
            matched = true
        end
    end
    matched || throw(BoundsError(m, indices))

    if dims == 1
        return m[keep, :]
    else
        return m[:, keep]
    end
end


function drop(m::AbstractMatrixEntry, row_key::AbstractArray{T}; dims = 1) where {T <: NamedTuple}
    if dims != 1 && dims != 2
        throw(ArgumentError("dims must be 1 or 2, got $dims"))
    end

    keep = trues(size(m.data, dims))

    lookup = dims == 1 ? m.row_lookup : m.col_lookup
    matched_any = false
    unmatched = NamedTuple[]
    for key in row_key
        matched_key = false
        for (full_key, idx) in lookup
            if _partial_key_match(full_key, key)
                keep[idx] = false
                matched_key = true
            end
        end
        if matched_key
            matched_any = true
        else
            push!(unmatched, key)
        end
    end
    matched_any || throw(BoundsError(m, row_key))
    if !isempty(unmatched)
        @warn "drop: some keys matched nothing and were ignored" unmatched_keys = unmatched
    end

    if dims == 1
        return m[keep, :]
    else
        return m[:, keep]
    end
end

function drop!(m::MatrixEntry, indices::T; dims = 1) where {T <: NamedTuple}
    # NB: drop! is intentionally defined only for the mutable MatrixEntry.
    # LeontiefFactorization is immutable (and its data is a cached inverse),
    # so in-place dropping is not meaningful for it; use non-mutating drop,
    # which accepts any AbstractMatrixEntry, instead.
    if dims != 1 && dims != 2
        throw(ArgumentError("dims must be 1 or 2, got $dims"))
    end

    # Find indices to keep (opposite of drop)
    keep = trues(size(m.data, dims))

    lookup = dims == 1 ? m.row_lookup : m.col_lookup
    matched = false
    for (full_key, idx) in lookup
        if _partial_key_match(full_key, indices)
            keep[idx] = false
            matched = true
        end
    end
    matched || throw(BoundsError(m, indices))

    # Replace (do not mutate) data and index frames: derived entries share
    # index DataFrames by reference, so in-place `deleteat!` would corrupt
    # parents and siblings. Rebinding keeps every other entry intact.
    if dims == 1
        # Drop rows
        m.data = m.data[keep, :]
        m.row_indices = m.row_indices[keep, :]
        m.row_lookup = _build_lookup(m.row_indices, "row")
    else
        # Drop columns
        m.data = m.data[:, keep]
        m.col_indices = m.col_indices[keep, :]
        m.col_lookup = _build_lookup(m.col_indices, "column")
    end
    _check_invariants(m)

    return m
end

function drop!(m::MatrixEntry, row_key::AbstractArray{T}; dims = 1) where {T <: NamedTuple}
    if dims != 1 && dims != 2
        throw(ArgumentError("dims must be 1 or 2, got $dims"))
    end

    keep = trues(size(m.data, dims))

    lookup = dims == 1 ? m.row_lookup : m.col_lookup
    matched_any = false
    unmatched = NamedTuple[]
    for key in row_key
        matched_key = false
        for (full_key, idx) in lookup
            if _partial_key_match(full_key, key)
                keep[idx] = false
                matched_key = true
            end
        end
        if matched_key
            matched_any = true
        else
            push!(unmatched, key)
        end
    end
    matched_any || throw(BoundsError(m, row_key))
    if !isempty(unmatched)
        @warn "drop!: some keys matched nothing and were ignored" unmatched_keys = unmatched
    end

    if dims == 1
        m.data = m.data[keep, :]
        m.row_indices = m.row_indices[keep, :]
        m.row_lookup = _build_lookup(m.row_indices, "row")
    else
        m.data = m.data[:, keep]
        m.col_indices = m.col_indices[keep, :]
        m.col_lookup = _build_lookup(m.col_indices, "column")
    end
    _check_invariants(m)

    return m
end
