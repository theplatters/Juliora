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
struct MatrixEntry
	data::Matrix{Float64}
	col_indices::DataFrame
	row_indices::DataFrame
	row_lookup::Dict{NamedTuple, Int}
	col_lookup::Dict{NamedTuple, Int}
	function MatrixEntry(data, col_indices, row_indices)
		@assert size(data) == (size(row_indices)[1], size(col_indices)[1]) "Data $(size(data)) dimensions must match index DataFrames $(size(row_indices)[1]), $(size(col_indices)[1])"

		row_lookup = Dict(NamedTuple(row) => i for (i, row) in enumerate(eachrow(row_indices)))
		col_lookup = Dict(NamedTuple(row) => i for (i, row) in enumerate(eachrow(col_indices)))
		new(data, col_indices, row_indices, row_lookup, col_lookup)
	end
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
function Base.getindex(m::MatrixEntry, row_key::NamedTuple, col_key::NamedTuple)
    row_idx = get(m.row_lookup, row_key, nothing)
    col_idx = get(m.col_lookup, col_key, nothing)

    isnothing(row_idx) && throw(BoundsError(m, row_key))
    isnothing(col_idx) && throw(BoundsError(m, col_key))

    return m.data[row_idx, col_idx]
end

function Base.getindex(m::MatrixEntry, row_key::NamedTuple, ::Colon)
    # Find all rows that match the partial key
    row_indices_set = Set{Int64}()
    for (full_key, idx) in m.row_lookup
        if all(k -> haskey(full_key, k) && full_key[k] == row_key[k], keys(row_key))
            push!(row_indices_set, idx)
        end
    end
    
    isempty(row_indices) && throw(BoundsError(m, row_key))
    row_indices = collect(row_indices_set)    
    if length(row_indices) == 1
        return SeriesEntry(m.data[row_indices[1], :], m.col_indices)
    else
        return MatrixEntry(m.data[row_indices, :], m.col_indices, m.row_indices[row_indices, :])
    end
end

function Base.getindex(m::MatrixEntry, ::Colon, col_key::NamedTuple)
    # Find all columns that match the partial key
    col_indices_set = Set{Int64}()
    for (full_key, idx) in m.col_lookup
        if all(k -> haskey(full_key, k) && full_key[k] == col_key[k], keys(col_key))
            push!(col_indices_set, idx)
        end
    end

    isempty(col_indices_set) && throw(BoundsError(m, col_key))

    col_indices = collect(col_indices_set)
    if length(col_indices) == 1
        return SeriesEntry(m.data[:, col_indices[1]], m.row_indices)
    else
        return MatrixEntry(m.data[:, col_indices], m.col_indices[col_indices, :], m.row_indices)
    end
end

function Base.getindex(m::MatrixEntry, ::Colon, col_key::AbstractArray{T}) where T <: NamedTuple
    col_indices_set = Set{Int64}()
    for key in col_key
        for (full_key, idx) in m.col_lookup
            if all(k -> haskey(full_key, k) && full_key[k] == key[k], keys(key))
                push!(col_indices_set, idx)
            end
        end
    end


    isempty(col_indices_set) && throw(BoundsError(m, col_key))
    
    col_indices = collect(col_indices_set)
    if length(col_indices) == 1
        return SeriesEntry(m.data[:, col_indices[1]], m.row_indices)
    else
        return MatrixEntry(m.data[:, col_indices], m.col_indices[col_indices, :], m.row_indices)
    end
end

function Base.getindex(m::MatrixEntry, row_key::AbstractArray{T}, ::Colon) where T <: NamedTuple
    row_indices_set = Set{Int64}()
    for key in row_key
        for (full_key, idx) in m.row_lookup
            if all(k -> haskey(full_key, k) && full_key[k] == key[k], keys(key))
                push!(row_indices_set, idx)
            end
        end
    end

    isempty(row_indices_set) && throw(BoundsError(m, row_key))

    row_indices = collect(row_indices_set)
    if length(row_indices) == 1
        return SeriesEntry(m.data[row_indices[1], :], m.col_indices)
    else
        return MatrixEntry(m.data[row_indices, :], m.col_indices, m.row_indices[row_indices, :])
    end
end


  

"""
	Base.getindex(m::MatrixEntry, row_mask::AbstractVector{Bool}, col_mask::AbstractVector{Bool})

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
function Base.getindex(m::MatrixEntry, row_mask::AbstractVector{Bool}, col_mask::AbstractVector{Bool})
	@assert length(row_mask) == size(m.data, 1) "Row mask length must match number of rows"
	@assert length(col_mask) == size(m.data, 2) "Column mask length must match number of columns"

	new_data = m.data[row_mask, col_mask]
	new_row_indices = m.row_indices[row_mask, :]
	new_col_indices = m.col_indices[col_mask, :]

	return MatrixEntry(new_data, new_col_indices, new_row_indices)
end

"""
	Base.getindex(m::MatrixEntry, row_mask::AbstractVector{Bool}, ::Colon)

Filter rows using a boolean mask while keeping all columns.

# Arguments
- `m::MatrixEntry`: The matrix entry to filter
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
function Base.getindex(m::MatrixEntry, row_mask::AbstractVector{Bool}, ::Colon)
	@assert length(row_mask) == size(m.data, 1) "Row mask length must match number of rows"

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
function Base.getindex(m::MatrixEntry, ::Colon, col_mask::AbstractVector{Bool})
	@assert length(col_mask) == size(m.data, 2) "Column mask length must match number of columns"

	new_data = m.data[:, col_mask]
	new_col_indices = m.col_indices[col_mask, :]

	return MatrixEntry(new_data, new_col_indices, m.row_indices)
end

# Boolean indexing with functions on row/column indices

"""
    filter_rows(m::MatrixEntry, condition_func)

Filter rows based on a condition function applied to row indices.

# Arguments
- `m::MatrixEntry`: The matrix entry to filter
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
function filter_rows(m::MatrixEntry, condition_func)
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
function filter_cols(m::MatrixEntry, condition_func)
	col_mask = [condition_func(NamedTuple(row)) for row in eachrow(m.col_indices)]
	return m[:, col_mask]
end

function Base.filter(fun::Function, m::MatrixEntry, dims=1)
    if dims > 2 || dims < 1 
        throw(BoundsError("Dimension not  supported, dims should either be 1 or 2, $dims was given"))
    end
    if dims == 1 ? filter_rows(m, fun) : filter_cols(m, fun)
        return filter_rows(m, fun)
    else
        return filter_cols(m, fun)
    end
end

