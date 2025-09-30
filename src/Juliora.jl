"""
    Juliora

A Julia package for economic input-output analysis using the Eora global MRIO database.

Juliora provides efficient data structures and methods for working with large-scale
multi-region input-output (MRIO) tables, with integrated support for environmental
extensions and TidierData operations.

# Key Features
- Memory-efficient `MatrixEntry` structure combining numerical matrices with labeled indices
- Boolean indexing and filtering capabilities
- Seamless integration with TidierData for data manipulation
- Built-in support for Eora global MRIO database
- Environmental impact analysis capabilities
- Economic analysis functions (multipliers, linkages, etc.)

# Main Types
- `MatrixEntry`: Core structure for labeled matrices
- `SeriesEntry`: Structure for labeled vectors  
- `Eora`: Complete MRIO database structure
- `EnvironmentalExtension`: Environmental impact data

# Example Usage
```julia
using Juliora

# Load Eora database
eora = Eora("path/to/eora/data/")

# Analyze trade flows
usa_exports = filter_rows(eora.T, row -> row.CountryCode == "USA")
eu_trade = sum_by_country(eora.T; dimension=:both)

# Environmental analysis  
co2_intensity = filter_rows(eora.env.A, row -> row.Stressor == "CO2")

# TidierData integration
result = @tidier_matrix eora.A begin
    @filter(value > 0.01)
    @group_by(row_Sector, col_Sector)
    @summarize(avg_coefficient = mean(value))
end
```

See individual function documentation for detailed usage information.
"""
module Juliora

using Tidier
using TidierFiles
using LinearAlgebra

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
```julia
data = rand(3, 2)
row_df = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Agr", "Man", "Ser"])
col_df = DataFrame(Country=["USA", "CHN"], Sector=["Agr", "Man"])
matrix_entry = MatrixEntry(data, col_df, row_df)
```
"""
struct MatrixEntry
	data::Matrix{Float64}
	col_indices::DataFrame
	row_indices::DataFrame
	row_lookup::Dict{NamedTuple, Int}
	col_lookup::Dict{NamedTuple, Int}
	function MatrixEntry(data, col_indices, row_indices)
		@assert size(data) == (size(row_indices)[1], size(col_indices)[1])

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
```julia
value = matrix_entry[(Country="USA", Sector="Manufacturing"), (Country="CHN", Sector="Services")]
```
"""
function Base.getindex(m::MatrixEntry, row_key::NamedTuple, col_key::NamedTuple)
	row_idx = get(m.row_lookup, row_key, nothing)
	col_idx = get(m.col_lookup, col_key, nothing)

	isnothing(row_idx) && throw(BoundsError(m, row_key))
	isnothing(col_idx) && throw(BoundsError(m, col_key))

	return m.data[row_idx, col_idx]
end

# Boolean indexing methods

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
```julia
usa_rows = matrix_entry.row_indices.Country .== "USA"
manufacturing_cols = matrix_entry.col_indices.Sector .== "Manufacturing"
filtered = matrix_entry[usa_rows, manufacturing_cols]
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
```julia
eu_countries = ["DEU", "FRA", "ITA"]
eu_mask = [country in eu_countries for country in matrix_entry.row_indices.Country]
eu_data = matrix_entry[eu_mask, :]
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
```julia
primary_sectors = matrix_entry.col_indices.Sector .== "Primary"
primary_data = matrix_entry[:, primary_sectors]
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
```julia
# Filter for developed countries
developed = filter_rows(matrix_entry, row -> row.Country in ["USA", "DEU", "JPN"])

# Filter for manufacturing sectors
manufacturing = filter_rows(matrix_entry, row -> row.Sector == "Manufacturing")
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
```julia
# Filter for Asian countries in columns
asian = filter_cols(matrix_entry, col -> col.Country in ["CHN", "JPN", "KOR"])
```
"""
function filter_cols(m::MatrixEntry, condition_func)
	col_mask = [condition_func(NamedTuple(row)) for row in eachrow(m.col_indices)]
	return m[:, col_mask]
end

"""
    filter_matrix(m::MatrixEntry, row_condition, col_condition)

Filter both rows and columns using separate condition functions.

# Arguments
- `m::MatrixEntry`: The matrix entry to filter
- `row_condition`: Function for filtering rows
- `col_condition`: Function for filtering columns

# Returns
- `MatrixEntry`: New MatrixEntry with both dimensions filtered

# Examples
```julia
# Get EU to NAFTA trade flows
eu_to_nafta = filter_matrix(matrix_entry,
    row -> row.Country in ["DEU", "FRA", "ITA"],
    col -> col.Country in ["USA", "CAN", "MEX"])
```
"""
function filter_matrix(m::MatrixEntry, row_condition, col_condition)
	row_mask = [row_condition(NamedTuple(row)) for row in eachrow(m.row_indices)]
	col_mask = [col_condition(NamedTuple(row)) for row in eachrow(m.col_indices)]
	return m[row_mask, col_mask]
end

# TidierData integration methods

"""
    to_long_dataframe(m::MatrixEntry; value_name::String="value")

Convert a MatrixEntry to long-form DataFrame suitable for TidierData operations.

This function transforms the matrix into a "tidy" format where each row represents
one observation (matrix cell) with separate columns for row indices, column indices,
and the value.

# Arguments
- `m::MatrixEntry`: The matrix entry to convert
- `value_name::String="value"`: Name for the column containing matrix values

# Returns
- `DataFrame`: Long-form DataFrame with prefixed index columns and values

# Column Structure
- `row_*`: Columns from row indices (prefixed with "row_")
- `col_*`: Columns from column indices (prefixed with "col_")
- `value_name`: Column containing the matrix values

# Examples
```julia
long_df = to_long_dataframe(matrix_entry; value_name="trade_flow")
# Result has columns: row_Country, row_Sector, col_Country, col_Sector, trade_flow
```
"""
function to_long_dataframe(m::MatrixEntry; value_name::String="value")
    n_rows, n_cols = size(m.data)
    
    # Create expanded indices
    row_indices_expanded = repeat(1:n_rows, n_cols)
    col_indices_expanded = repeat(1:n_cols, inner=n_rows)
    
    # Build the long DataFrame
    df = DataFrame()
    
    # Add row index columns with prefix
    for col_name in names(m.row_indices)
        df[!, Symbol("row_" * string(col_name))] = m.row_indices[row_indices_expanded, col_name]
    end
    
    # Add column index columns with prefix  
    for col_name in names(m.col_indices)
        df[!, Symbol("col_" * string(col_name))] = m.col_indices[col_indices_expanded, col_name]
    end
    
    # Add the values
    df[!, Symbol(value_name)] = vec(m.data)
    
    return df
end

"""
    from_long_dataframe(df::DataFrame; value_col="value", row_prefix="row_", col_prefix="col_")

Convert a long-form DataFrame back to MatrixEntry format.

This function reverses the `to_long_dataframe` operation, reconstructing the matrix
and indices from a tidy DataFrame.

# Arguments
- `df::DataFrame`: Long-form DataFrame to convert
- `value_col::String="value"`: Name of column containing values
- `row_prefix::String="row_"`: Prefix for row index columns
- `col_prefix::String="col_"`: Prefix for column index columns

# Returns
- `MatrixEntry`: Reconstructed MatrixEntry with matrix data and indices

# Examples
```julia
# After manipulating the long DataFrame
processed_df = @chain long_df begin
    @filter(trade_flow > 1000)
    @group_by(row_Country, col_Country) 
    @summarize(total_flow = sum(trade_flow))
end

matrix_result = from_long_dataframe(processed_df; value_col="total_flow")
```
"""
function from_long_dataframe(df::DataFrame; 
                             value_col::String="value",
                             row_prefix::String="row_",
                             col_prefix::String="col_")
    
    # Extract row and column index columns
    row_cols = filter(name -> startswith(string(name), row_prefix), names(df))
    col_cols = filter(name -> startswith(string(name), col_prefix), names(df))
    
    # Create clean column names (remove prefixes)
    clean_row_cols = [Symbol(replace(string(col), row_prefix => "")) for col in row_cols]
    clean_col_cols = [Symbol(replace(string(col), col_prefix => "")) for col in col_cols]
    
    # Get unique row and column indices
    row_df = unique(df[!, row_cols])
    col_df = unique(df[!, col_cols])
    
    # Rename columns to remove prefixes
    rename!(row_df, Dict(zip(row_cols, clean_row_cols)))
    rename!(col_df, Dict(zip(col_cols, clean_col_cols)))
    
    # Create matrix
    n_rows, n_cols = nrow(row_df), nrow(col_df)
    data_matrix = zeros(Float64, n_rows, n_cols)
    
    # Fill matrix with values
    for row in eachrow(df)
        # Find row and column indices
        row_vals = NamedTuple(row[col] for col in row_cols)
        col_vals = NamedTuple(row[col] for col in col_cols)
        
        row_idx = findfirst(r -> NamedTuple(r) == row_vals, eachrow(row_df))
        col_idx = findfirst(c -> NamedTuple(c) == col_vals, eachrow(col_df))
        
        if !isnothing(row_idx) && !isnothing(col_idx)
            data_matrix[row_idx, col_idx] = row[Symbol(value_col)]
        end
    end
    
    return MatrixEntry(data_matrix, col_df, row_df)
end

"""
    @tidier_matrix(matrix_expr, tidier_operations...)

Macro for applying TidierData operations directly to MatrixEntry objects.

This macro automatically converts the MatrixEntry to long form, applies the specified
TidierData operations, and attempts to convert back to MatrixEntry format.

# Arguments
- `matrix_expr`: Expression that evaluates to a MatrixEntry
- `tidier_operations...`: TidierData operations to apply

# Returns
- `MatrixEntry`: If conversion back is successful
- `DataFrame`: If the result cannot be converted back to MatrixEntry

# Examples
```julia
# Filter and summarize in one operation
result = @tidier_matrix eora.T begin
    @filter(value > 1000)
    @group_by(row_Country, col_Country)
    @summarize(total_trade = sum(value))
end

# Complex transformations
processed = @tidier_matrix matrix_entry begin
    @mutate(log_value = log(value + 1))
    @filter(row_Sector == "Manufacturing")
    @select(row_Country, col_Country, log_value)
end
```
"""
macro tidier_matrix(matrix_expr, tidier_operations...)
    quote
        # Convert to DataFrame
        df = to_long_dataframe($(esc(matrix_expr)))
        
        # Apply TidierData operations
        result_df = @chain df begin
            $(esc.(tidier_operations)...)
        end
        
        # Convert back to MatrixEntry if possible, otherwise return DataFrame
        try
            from_long_dataframe(result_df)
        catch
            result_df  # Return DataFrame if conversion back fails
        end
    end
end

# Convenience methods for common TidierData operations

"""
    groupby_matrix(m::MatrixEntry, grouping_cols...; agg_func=sum, rows=true, value_name="value")

Group and aggregate matrix data by specified index columns.

# Arguments
- `m::MatrixEntry`: The matrix entry to group
- `grouping_cols...`: Column names to group by (as symbols or strings)
- `agg_func=sum`: Aggregation function to apply (sum, mean, maximum, etc.)
- `rows=true`: Whether to group by row indices (true) or column indices (false)
- `value_name="value"`: Name for the value column in output

# Returns
- `DataFrame`: Grouped and aggregated results

# Examples
```julia
# Sum by country (grouping row indices)
country_sums = groupby_matrix(matrix_entry, :Country; agg_func=sum, rows=true)

# Average by sector (grouping column indices)  
sector_means = groupby_matrix(matrix_entry, :Sector; agg_func=mean, rows=false)

# Group by multiple dimensions
complex_groups = groupby_matrix(matrix_entry, :Country, :Sector; agg_func=maximum)
```
"""
function groupby_matrix(m::MatrixEntry, grouping_cols...; 
                       agg_func=sum, 
                       rows=true, 
                       value_name="value")
    df = to_long_dataframe(m; value_name=value_name)
    
    # Determine which columns to group by
    group_cols = if rows
        [Symbol("row_" * string(col)) for col in grouping_cols]
    else
        [Symbol("col_" * string(col)) for col in grouping_cols]
    end
    
    # Apply grouping and aggregation
    result_df = @chain df begin
        @group_by($(group_cols...))
        @summarize(!!Symbol(value_name) := agg_func(!!Symbol(value_name)))
        @ungroup()
    end
    
    return result_df
end

"""
    sum_by_country(m::MatrixEntry; dimension=:both)

Sum matrix values by country codes. Common operation in input-output analysis.

# Arguments
- `m::MatrixEntry`: The matrix entry to aggregate
- `dimension=:both`: Which dimension to sum over
  - `:rows`: Sum across rows (by row countries)
  - `:cols`: Sum across columns (by column countries)
  - `:both`: Create country-to-country summary matrix

# Returns
- `DataFrame`: Country-level aggregated data

# Examples
```julia
# Total exports by country (sum rows)
exports = sum_by_country(eora.T; dimension=:rows)

# Total imports by country (sum columns)
imports = sum_by_country(eora.T; dimension=:cols)

# Country-to-country trade matrix
bilateral = sum_by_country(eora.T; dimension=:both)
```
"""
function sum_by_country(m::MatrixEntry; dimension=:both)
    if dimension == :rows
        return groupby_matrix(m, :CountryCode; rows=true)
    elseif dimension == :cols  
        return groupby_matrix(m, :CountryCode; rows=false)
    else  # both
        df = to_long_dataframe(m)
        return @chain df begin
            @group_by(row_CountryCode, col_CountryCode)
            @summarize(value = sum(value))
            @ungroup()
        end
    end
end

"""
    sum_by_sector(m::MatrixEntry; dimension=:both)

Sum matrix values by sector codes. Useful for sectoral analysis in IO models.

# Arguments
- `m::MatrixEntry`: The matrix entry to aggregate
- `dimension=:both`: Which dimension to sum over
  - `:rows`: Sum across rows (by row sectors)
  - `:cols`: Sum across columns (by column sectors)
  - `:both`: Create sector-to-sector summary matrix

# Returns
- `DataFrame`: Sector-level aggregated data

# Examples
```julia
# Total output by sector
output = sum_by_sector(eora.T; dimension=:rows)

# Total input requirements by sector
input = sum_by_sector(eora.T; dimension=:cols)

# Intersectoral linkages
linkages = sum_by_sector(eora.A; dimension=:both)
```
"""
function sum_by_sector(m::MatrixEntry; dimension=:both)
    if dimension == :rows
        return groupby_matrix(m, :Sector; rows=true)
    elseif dimension == :cols
        return groupby_matrix(m, :Sector; rows=false)  
    else  # both
        df = to_long_dataframe(m)
        return @chain df begin
            @group_by(row_Sector, col_Sector)
            @summarize(value = sum(value))
            @ungroup()
        end
    end
end

"""
    SeriesEntry

A structure for storing a vector of numerical data with labeled indices,
optimized for economic time series or cross-sectional data.

# Fields
- `data::Vector{Float64}`: The numerical vector data
- `col_indices::DataFrame`: DataFrame containing labels and metadata for each element

# Examples
```julia
data = [100.0, 200.0, 150.0]
indices = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Total", "Total", "Total"])
series = SeriesEntry(data, indices)
```
"""
struct SeriesEntry
	data::Vector{Float64}
	col_indices::DataFrame
end

# Additional TidierData convenience methods

"""
    Base.:|>(m::MatrixEntry, f::Function)

Pipe operator for MatrixEntry to work seamlessly with functions expecting DataFrames.

Automatically converts the MatrixEntry to long-form DataFrame before applying the function.

# Arguments
- `m::MatrixEntry`: The matrix entry to pipe
- `f::Function`: Function to apply to the converted DataFrame

# Returns
- Result of applying `f` to the long-form DataFrame

# Examples
```julia
# Pipe directly to TidierData operations
result = matrix_entry |> df -> @chain df begin
    @filter(value > 100)
    @group_by(row_Country)
    @summarize(total = sum(value))
end
```
"""
function Base.:|>(m::MatrixEntry, f::Function)
    return f(to_long_dataframe(m))
end

"""
    tidier_filter_rows(m::MatrixEntry, condition_expr)

Filter MatrixEntry rows using TidierData syntax and return a new MatrixEntry.

# Arguments
- `m::MatrixEntry`: The matrix entry to filter
- `condition_expr`: TidierData-style filter expression

# Returns
- `MatrixEntry`: Filtered matrix entry

# Examples
```julia
# Filter for high-value flows
filtered = tidier_filter_rows(matrix_entry, value > 1000)

# Filter by country
usa_data = tidier_filter_rows(matrix_entry, row_Country == "USA")
```
"""
function tidier_filter_rows(m::MatrixEntry, condition_expr)
    df = to_long_dataframe(m)
    
    # Apply filter to the long DataFrame
    filtered_df = @chain df begin
        @filter($(condition_expr))
    end
    
    # Convert back to MatrixEntry
    return from_long_dataframe(filtered_df)
end

"""
    add_calculated_column(m::MatrixEntry, col_name::Symbol, calculation_func; to_rows=true)

Add a calculated column to row or column indices based on existing index values.

# Arguments
- `m::MatrixEntry`: The matrix entry to modify
- `col_name::Symbol`: Name of the new column to add
- `calculation_func`: Function that takes a NamedTuple (row) and returns a value
- `to_rows=true`: Whether to add to row indices (true) or column indices (false)

# Returns
- `MatrixEntry`: New MatrixEntry with added calculated column

# Examples
```julia
# Add region classification to rows
with_regions = add_calculated_column(matrix_entry, :Region, 
    row -> row.Country in ["USA", "CAN", "MEX"] ? "NAFTA" : "Other")

# Add development status to columns
with_development = add_calculated_column(matrix_entry, :Development,
    col -> col.Country in ["USA", "DEU", "JPN"] ? "Developed" : "Developing",
    to_rows=false)
```
"""
function add_calculated_column(m::MatrixEntry, col_name::Symbol, calculation_func; to_rows=true)
    if to_rows
        new_row_indices = copy(m.row_indices)
        new_row_indices[!, col_name] = [calculation_func(NamedTuple(row)) for row in eachrow(m.row_indices)]
        return MatrixEntry(m.data, m.col_indices, new_row_indices)
    else
        new_col_indices = copy(m.col_indices) 
        new_col_indices[!, col_name] = [calculation_func(NamedTuple(row)) for row in eachrow(m.col_indices)]
        return MatrixEntry(m.data, new_col_indices, m.row_indices)
    end
end

"""
    pivot_matrix_to_wide(m::MatrixEntry, row_vars, col_var, value_var="value")

Pivot the matrix data to wide format for analysis or visualization.

# Arguments
- `m::MatrixEntry`: The matrix entry to pivot
- `row_vars`: Variables to use as row identifiers (vector of symbols/strings)
- `col_var`: Variable to use for column names (symbol/string)
- `value_var="value"`: Name for the value column

# Returns
- `DataFrame`: Wide-format DataFrame suitable for analysis or export

# Examples
```julia
# Create country-by-sector matrix
wide_format = pivot_matrix_to_wide(matrix_entry, 
    [:Country], :Sector, "trade_value")

# Multi-dimensional pivot
complex_pivot = pivot_matrix_to_wide(matrix_entry,
    [:Country, :Region], :Sector)
```
"""
function pivot_matrix_to_wide(m::MatrixEntry, row_vars, col_var, value_var="value")
    df = to_long_dataframe(m; value_name=value_var)
    
    # Create row identifier
    row_id_cols = [Symbol("row_" * string(var)) for var in row_vars]
    col_id_col = Symbol("col_" * string(col_var))
    
    return @chain df begin
        @select($(row_id_cols...), !!col_id_col, !!Symbol(value_var))
        @pivot_wider(names_from = !!col_id_col, values_from = !!Symbol(value_var))
    end
end

"""
    matrix_summary(m::MatrixEntry)

Generate comprehensive summary statistics for the matrix data.

# Arguments
- `m::MatrixEntry`: The matrix entry to summarize

# Returns
- `DataFrame`: Single-row DataFrame with summary statistics

# Statistics Included
- `total`: Sum of all values
- `mean`: Average value
- `median`: Median value
- `std`: Standard deviation
- `min_val`: Minimum value
- `max_val`: Maximum value
- `n_nonzero`: Count of non-zero values
- `n_total`: Total number of values

# Examples
```julia
stats = matrix_summary(eora.A)
println("Matrix density: ", stats.n_nonzero[1] / stats.n_total[1])
```
"""
function matrix_summary(m::MatrixEntry)
    df = to_long_dataframe(m)
    
    return @chain df begin
        @summarize(
            total = sum(value),
            mean = mean(value), 
            median = median(value),
            std = std(value),
            min_val = minimum(value),
            max_val = maximum(value),
            n_nonzero = sum(value .!= 0),
            n_total = length(value)
        )
    end
end

"""
    country_summary(m::MatrixEntry)

Generate country-by-country flow summary for bilateral analysis.

Creates a summary of flows between all country pairs, useful for analyzing
trade patterns, economic linkages, and bilateral relationships.

# Arguments
- `m::MatrixEntry`: The matrix entry to analyze

# Returns
- `DataFrame`: Country-pair summary with columns:
  - `row_CountryCode`: Origin country
  - `col_CountryCode`: Destination country
  - `total_flow`: Sum of all flows between the countries
  - `mean_flow`: Average flow value
  - `n_sectors`: Number of sectors in the relationship

# Examples
```julia
# Analyze trade relationships
trade_summary = country_summary(eora.T)

# Find largest trade partners
top_partners = first(trade_summary, 10)

# Analyze environmental flows
co2_flows = country_summary(eora.env.F)
```
"""
function country_summary(m::MatrixEntry)
    df = to_long_dataframe(m)
    
    return @chain df begin
        @group_by(row_CountryCode, col_CountryCode)
        @summarize(
            total_flow = sum(value),
            mean_flow = mean(value),
            n_sectors = length(value)
        )
        @ungroup()
        @arrange(desc(total_flow))
    end
end

"""
    EnvironmentalExtension

Structure containing environmental impact data for input-output analysis.

# Fields
- `A::MatrixEntry`: Environmental multipliers matrix (impacts per unit of output)
- `F::MatrixEntry`: Direct environmental impacts matrix (absolute impacts)

# Description
The environmental extension links economic activity to environmental impacts
such as CO2 emissions, water use, land use, etc. The `F` matrix contains
direct impacts while `A` contains impact intensities (impacts per dollar of output).
"""
struct EnvironmentalExtension
	A::MatrixEntry
	F::MatrixEntry
end

"""
    EnvironmentalExtension(path::String, x)

Construct environmental extension from Eora database files.

# Arguments
- `path::String`: Directory path containing Eora environmental files
- `x`: Vector of total output by sector (for calculating intensities)

# Required Files
- `Q.txt`: Environmental impacts matrix
- `labels_T.txt`: Sector labels (for matching with economic data)
- `labels_Q.txt`: Environmental stressor labels

# Returns
- `EnvironmentalExtension`: Complete environmental extension with both direct impacts and intensities

# Examples
```julia
# Load environmental data
env_ext = EnvironmentalExtension("data/2017/", total_output_vector)

# Access CO2 intensities
co2_intensity = filter_rows(env_ext.A, row -> row.Stressor == "CO2")
```
"""
function EnvironmentalExtension(path::String, x)
	f = Matrix(read_csv(path * "Q.txt", col_names = false, delim = "\t"))
	t_indices = @chain read_csv(path * "labels_T.txt", delim = "\t", col_names = false) begin
		@select(CountryCode = Column2, Industry = Column3, Sector = Column4)
	end
	f_indices = @chain read_csv(path * "labels_Q.txt", delim = "\t", col_names = false) begin
		@select(Stressor = Column1, Source = Column2)
	end

	EnvironmentalExtension(MatrixEntry(f, t_indices, f_indices), MatrixEntry(f ./ x', t_indices, f_indices))
end

"""
    Eora

Complete Eora global multi-region input-output (MRIO) database structure.

# Fields
- `A::MatrixEntry`: Technical coefficients matrix (intermediate inputs per unit output)
- `T::MatrixEntry`: Intermediate transaction matrix (monetary flows between sectors)
- `VA::MatrixEntry`: Value added matrix (primary inputs by sector)
- `FD::MatrixEntry`: Final demand matrix (consumption, investment, government, exports)
- `L::MatrixEntry`: Leontief inverse matrix (total requirements matrix)
- `X::SeriesEntry`: Total output vector by sector
- `env::EnvironmentalExtension`: Environmental impact data

# Description
The Eora database provides a complete picture of the global economy with detailed
sectoral and country-level data. This structure contains all the key matrices
needed for input-output analysis, multiplier calculations, and environmental
impact assessments.

# Matrix Dimensions
All matrices share consistent country-sector dimensions, typically:
- Rows/Columns: Countries × Sectors (e.g., 189 countries × 26 sectors)
- Environmental: Stressors × (Countries × Sectors)
"""
struct Eora
	A::MatrixEntry
	T::MatrixEntry
	VA::MatrixEntry
	FD::MatrixEntry
	L::MatrixEntry
	X::SeriesEntry
	env::EnvironmentalExtension
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

# Returns
- `Eora`: Complete MRIO database with all matrices and environmental data

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
	t = Matrix(read_csv(path * "T.txt", col_names = false, delim = "\t"))
	t_indices = @chain read_csv(path * "labels_T.txt", delim = "\t", col_names = false) begin
		@select(CountryCode = Column2, Industry = Column3, Sector = Column4)
	end


	v = Matrix(read_csv(path * "VA.txt", col_names = false, delim = "\t"))
	v_colnames = @chain read_csv(path * "labels_VA.txt", delim = "\t", col_names = false) begin
		@select(PrimaryInput = Column2)
	end

	y = Matrix(read_csv(path * "FD.txt", col_names = false, delim = "\t"))
	y_indices = @chain read_csv(path * "labels_FD.txt", delim = "\t", col_names = false) begin
		@select(CountryCode = Column2, Industry = Column3, Category = Column4)
	end


	x = vec(sum(t, dims = 2) + sum(y, dims = 2))
	a = t ./ replace(x, 0.0 => 1.0)

	Eora(
		MatrixEntry(a, t_indices, t_indices),
		MatrixEntry(t, t_indices, t_indices),
		MatrixEntry(v, t_indices, v_colnames),
		MatrixEntry(y, y_indices, t_indices),
		MatrixEntry(inv(I - a), t_indices, t_indices),
		SeriesEntry(x, t_indices),
		EnvironmentalExtension(path, x),
	)
end

end # module Juliora

