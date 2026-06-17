# Analysis and TidierData Integration Functions

using DataFrames
using Statistics: mean, median, std

"""
    filter_matrix(m::AbstractMatrixEntry, row_condition, col_condition)

Filter both rows and columns using separate condition functions.
"""
function filter_matrix(m::AbstractMatrixEntry, row_condition, col_condition)
  row_mask = [row_condition(NamedTuple(row)) for row in eachrow(m.row_indices)]
  col_mask = [col_condition(NamedTuple(row)) for row in eachrow(m.col_indices)]
  return m[row_mask, col_mask]
end

"""
    to_long_dataframe(m::AbstractMatrixEntry; value_name::String="value")

Convert a MatrixEntry to long-form DataFrame suitable for TidierData operations.
"""
function to_long_dataframe(m::AbstractMatrixEntry; value_name::String="value")
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
"""
function from_long_dataframe(
  df::DataFrame;
  value_col::String="value",
  row_prefix::String="row_",
  col_prefix::String="col_"
)

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
    groupby_matrix(m::AbstractMatrixEntry, grouping_cols...; agg_func=sum, rows=true, value_name="value")

Group and aggregate matrix data by specified index columns.
"""
function groupby_matrix(
  m::AbstractMatrixEntry, grouping_cols...;
  agg_func=sum,
  rows=true,
  value_name="value"
)
  df = to_long_dataframe(m; value_name=value_name)

  # Determine which columns to group by
  group_cols = if rows
    [Symbol("row_" * string(col)) for col in grouping_cols]
  else
    [Symbol("col_" * string(col)) for col in grouping_cols]
  end

  # Apply grouping and aggregation using pure DataFrames operations
  return DataFrames.combine(DataFrames.groupby(df, group_cols), Symbol(value_name) => agg_func => Symbol(value_name))
end

"""
    sum_by_country(m::AbstractMatrixEntry; dimension=:both)

Sum matrix values by country codes.
"""
function sum_by_country(m::AbstractMatrixEntry; dimension=:both)
  if dimension == :rows
    return groupby_matrix(m, :CountryCode; rows=true)
  elseif dimension == :cols
    return groupby_matrix(m, :CountryCode; rows=false)
  else  # both
    df = to_long_dataframe(m)
    return DataFrames.combine(DataFrames.groupby(df, [:row_CountryCode, :col_CountryCode]), :value => sum => :value)
  end
end

"""
    sum_by_sector(m::AbstractMatrixEntry; dimension=:both)

Sum matrix values by sector codes.
"""
function sum_by_sector(m::AbstractMatrixEntry; dimension=:both)
  if dimension == :rows
    return groupby_matrix(m, :Sector; rows=true)
  elseif dimension == :cols
    return groupby_matrix(m, :Sector; rows=false)
  else  # both
    df = to_long_dataframe(m)
    return DataFrames.combine(DataFrames.groupby(df, [:row_Sector, :col_Sector]), :value => sum => :value)
  end
end

"""
    Base.:|>(m::AbstractMatrixEntry, f::Function)

Pipe operator for MatrixEntry to work seamlessly with functions expecting DataFrames.
"""
function Base.:|>(m::AbstractMatrixEntry, f::Function)
  return f(to_long_dataframe(m))
end

"""
    add_calculated_column(m::AbstractMatrixEntry, col_name::Symbol, calculation_func; to_rows=true)

Add a calculated column to row or column indices based on existing index values.
"""
function add_calculated_column(m::AbstractMatrixEntry, col_name::Symbol, calculation_func; to_rows=true)
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
    pivot_matrix_to_wide(m::AbstractMatrixEntry, row_vars, col_var, value_var="value")

Pivot the matrix data to wide format for analysis or visualization.
"""
function pivot_matrix_to_wide(m::AbstractMatrixEntry, row_vars, col_var, value_var="value")
  df = to_long_dataframe(m; value_name=value_var)
  row_id_cols = [Symbol("row_" * string(var)) for var in row_vars]
  col_id_col = Symbol("col_" * string(col_var))
  return DataFrames.unstack(df, row_id_cols, col_id_col, Symbol(value_var))
end

"""
    matrix_summary(m::AbstractMatrixEntry)

Generate comprehensive summary statistics for the matrix data.
"""
function matrix_summary(m::AbstractMatrixEntry)
  df = to_long_dataframe(m)
  val = df.value
  return DataFrame(
    total=sum(val),
    mean=mean(val),
    median=median(val),
    std=std(val),
    min_val=minimum(val),
    max_val=maximum(val),
    n_nonzero=sum(val .!= 0),
    n_total=length(val)
  )
end

"""
    country_summary(m::AbstractMatrixEntry)

Generate country-by-country flow summary for bilateral analysis.
"""
function country_summary(m::AbstractMatrixEntry)
  df = to_long_dataframe(m)
  res = DataFrames.combine(
    DataFrames.groupby(df, [:row_CountryCode, :col_CountryCode]),
    :value => sum => :total_flow,
    :value => mean => :mean_flow,
    :value => length => :n_sectors
  )
  return sort!(res, :total_flow, rev=true)
end

"""
    induced_production(mrio::MRIO; consumer_countries::Vector{String}=String[], producer_countries::Vector{String}=String[])

Calculate the production induced by the final demand of specified consumer countries 
on specified producer countries using the Leontief Inverse matrix.

If `consumer_countries` is empty, final demand from all countries is included.
If `producer_countries` is empty, output for all producing countries is returned.
"""
function induced_production(
  mrio::MRIO;
  consumer_countries::AbstractVector=String[],
  producer_countries::AbstractVector=String[]
)

  # 1. Identify columns in Y corresponding to the consumer countries
  y_cols = mrio.Y.col_indices.CountryCode

  # If consumer_countries is empty, include all consumers
  consumer_mask = if isempty(consumer_countries)
    trues(length(y_cols))
  else
    [c in consumer_countries for c in y_cols]
  end

  # 2. Get the final demand submatrix and sum rows to get a vector
  y_demand = sum(mrio.Y.data[:, consumer_mask], dims=2)[:]

  # 3. Calculate induced production: x = L * y_demand (solving linear system)
  x_induced = mrio.L.factorization \ y_demand

  # 4. Filter for producer countries
  row_indices = mrio.L.row_indices

  # If producer_countries is empty, include all producers
  producer_mask = if isempty(producer_countries)
    trues(size(row_indices, 1))
  else
    [c in producer_countries for c in row_indices.CountryCode]
  end

  # 5. Build and return DataFrame
  df = DataFrame(
    CountryCode=row_indices.CountryCode[producer_mask],
    Sector=row_indices.Sector[producer_mask],
    InducedProduction=x_induced[producer_mask]
  )
  return df
end

function induced_production(
  mrio::MRIO,
  consumer_countries::AbstractVector,
  producer_countries::AbstractVector
)
  return induced_production(
    mrio;
    consumer_countries=Vector{String}(consumer_countries),
    producer_countries=Vector{String}(producer_countries)
  )
end

function induced_production(mrio::MRIO, consumer_countries::String, producer_countries::AbstractVector)
  return induced_production(mrio, [consumer_countries], producer_countries)
end

function induced_production(mrio::MRIO, consumer_countries::AbstractVector, producer_countries::String)
  return induced_production(mrio, consumer_countries, [producer_countries])
end

function induced_production(mrio::MRIO, consumer_countries::String, producer_countries::String)
  return induced_production(mrio, [consumer_countries], [producer_countries])
end
