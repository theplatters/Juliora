# Analysis and TidierData Integration Functions

using DataFrames
using Statistics: mean, median, std

"""
    _pick_col(colnames, candidates...)

Return the first of `candidates` present in `colnames` (compared by string
value, preserving the stored name type). Throws a descriptive `ArgumentError`
when none of the candidates is found. Used to support dual-naming conventions
(e.g. `CountryCode`/`Country`, `Sector`/`Industry`) across analysis helpers.
"""
function _pick_col(colnames, candidates...)
    wanted = string.(candidates)
    strnames = string.(colnames)
    for (stored, strname) in zip(colnames, strnames)
        if strname in wanted
            return stored
        end
    end
    throw(
        ArgumentError(
            "None of the expected columns [$(join(wanted, ", "))] found; " *
                "available columns: [$(join(strnames, ", "))]"
        )
    )
end

"""
    _country_mask(values, requested, kind)

Build a boolean mask selecting entries of `values` whose string form appears
in `requested`. An empty `requested` selects everything. When `requested` is
non-empty but nothing matches, throws an `ArgumentError` listing the unmatched
names; when only some match, emits a `@warn` listing the unmatched names.
`kind` labels the role (`"consumer"`/`"producer"`) in messages.
"""
function _country_mask(values, requested, kind::AbstractString)
    isempty(requested) && return trues(length(values))
    requested_str = string.(requested)
    values_str = string.(values)
    unmatched = unique(filter(r -> r ∉ values_str, requested_str))
    if !isempty(unmatched) && !any(v -> v in requested_str, values_str)
        throw(
            ArgumentError(
                "No $kind countries matched; unmatched $kind names: [$(join(unmatched, ", "))]. " *
                    "Available: [$(join(unique(values_str), ", "))]"
            )
        )
    elseif !isempty(unmatched)
        @warn "Some $kind countries did not match any data and were ignored: [$(join(unmatched, ", "))]"
    end
    return [v in requested_str for v in values_str]
end

"""
    _require_env(mrio::MRIO)

Return `mrio.env`, throwing a clear `ArgumentError` when the MRIO was built
without environmental data.
"""
function _require_env(mrio::MRIO)
    mrio.env === nothing && throw(
        ArgumentError(
            "MRIO has no environmental extension; construct it with environmental data " *
                "(e.g. parse_gloria, Eora) or pass an EnvironmentalExtension directly"
        )
    )
    return mrio.env
end

"""
    _require_leontief(mrio::MRIO)

Return `mrio.L`, throwing a clear `ArgumentError` for non-square systems that
have no Leontief factorization.
"""
function _require_leontief(mrio::MRIO)
    mrio.L === nothing && throw(
        ArgumentError("MRIO has no Leontief factorization (non-square system)")
    )
    return mrio.L
end

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
function to_long_dataframe(m::AbstractMatrixEntry; value_name::String = "value")
    n_rows, n_cols = size(m.data)

    # Create expanded indices
    row_indices_expanded = repeat(1:n_rows, n_cols)
    col_indices_expanded = repeat(1:n_cols, inner = n_rows)

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

Duplicate `(row, column)` pairs are aggregated with `sum`. Rows whose value
column is missing raise an error; data rows are always recognized since the
index tables are derived from `df` itself, but any row that cannot be matched
to the derived indices is counted and reported with a `@warn` (it is skipped).
Runs in O(N) time via dictionary lookups.

Inverse of `to_long_dataframe` (up to row/column ordering and duplicate
summation).
"""
function from_long_dataframe(
        df::DataFrame;
        value_col::String = "value",
        row_prefix::String = "row_",
        col_prefix::String = "col_"
    )

    # Extract row and column index columns
    row_cols = filter(name -> startswith(string(name), row_prefix), names(df))
    col_cols = filter(name -> startswith(string(name), col_prefix), names(df))

    value_sym = Symbol(value_col)
    value_sym in Symbol.(names(df)) || throw(
        ArgumentError(
            "Value column :$value_col not found in DataFrame; " *
                "available columns: [$(join(string.(names(df)), ", "))]"
        )
    )

    # Create clean column names (remove prefixes)
    clean_row_cols = [Symbol(replace(string(col), row_prefix => "")) for col in row_cols]
    clean_col_cols = [Symbol(replace(string(col), col_prefix => "")) for col in col_cols]

    # Get unique row and column indices
    row_df = unique(df[!, row_cols])
    col_df = unique(df[!, col_cols])

    # Rename columns to remove prefixes
    rename!(row_df, Dict(zip(row_cols, clean_row_cols)))
    rename!(col_df, Dict(zip(col_cols, clean_col_cols)))

    # Build dictionary lookups: key tuple -> matrix position (single pass each)
    row_lookup = Dict{Tuple, Int}()
    for (i, row) in enumerate(eachrow(row_df))
        row_lookup[Tuple(row[col] for col in clean_row_cols)] = i
    end
    col_lookup = Dict{Tuple, Int}()
    for (j, col) in enumerate(eachrow(col_df))
        col_lookup[Tuple(col[c] for c in clean_col_cols)] = j
    end

    # Create matrix
    n_rows, n_cols = nrow(row_df), nrow(col_df)
    data_matrix = zeros(Float64, n_rows, n_cols)

    # Fill matrix with values in a single pass, summing duplicates
    n_duplicates = 0
    n_dropped = 0
    seen = Set{Tuple{Int, Int}}()
    for row in eachrow(df)
        row_key = Tuple(row[col] for col in row_cols)
        col_key = Tuple(row[col] for col in col_cols)

        row_idx = get(row_lookup, row_key, nothing)
        col_idx = get(col_lookup, col_key, nothing)

        if isnothing(row_idx) || isnothing(col_idx)
            n_dropped += 1
        else
            if (row_idx, col_idx) in seen
                n_duplicates += 1
            else
                push!(seen, (row_idx, col_idx))
            end
            data_matrix[row_idx, col_idx] += row[value_sym]
        end
    end

    if n_dropped > 0 || n_duplicates > 0
        @warn "from_long_dataframe: skipped $n_dropped rows with unrecognized keys; summed $n_duplicates duplicate (row, column) pairs"
    end

    return MatrixEntry(data_matrix, col_df, row_df)
end

"""
    groupby_matrix(m::AbstractMatrixEntry, grouping_cols...; agg_func=sum, rows=true, value_name="value")

Group and aggregate matrix data by specified index columns.

`agg_func` may be a function or a string naming one (resolved via
`string_to_func`, e.g. `"sum"`, `"mean"`, `"statistics.mean"`, `"base.sum"`).
The string form is the path used by the R bindings.
"""
function groupby_matrix(
        m::AbstractMatrixEntry, grouping_cols...;
        agg_func = sum,
        rows = true,
        value_name = "value"
    )
    isempty(grouping_cols) && throw(
        ArgumentError("groupby_matrix requires at least one grouping column")
    )
    # Resolve string aggregation names (R-binding path) to functions.
    func = agg_func isa AbstractString ? string_to_func(agg_func) : agg_func
    df = to_long_dataframe(m; value_name = value_name)

    # Determine which columns to group by
    group_cols = if rows
        [Symbol("row_" * string(col)) for col in grouping_cols]
    else
        [Symbol("col_" * string(col)) for col in grouping_cols]
    end

    # Apply grouping and aggregation using pure DataFrames operations
    return DataFrames.combine(DataFrames.groupby(df, group_cols), Symbol(value_name) => func => Symbol(value_name))
end

# Convenience method accepting the grouping columns as a single vector. This is
# the path used by the R bindings, which pass one Vector{Symbol} argument rather
# than splicing varargs. The positional signature differs from the varargs
# method above (AbstractVector vs Vararg), so the two coexist without conflict.
function groupby_matrix(
        m::AbstractMatrixEntry, grouping_cols::AbstractVector;
        agg_func = sum,
        rows = true,
        value_name = "value"
    )
    return groupby_matrix(m, grouping_cols...; agg_func = agg_func, rows = rows, value_name = value_name)
end

"""
    sum_by_country(m::AbstractMatrixEntry; dimension=:both)

Sum matrix values by country codes. Accepts either the `CountryCode` or the
`Country` index column on each side (resolved independently for rows and
columns).
"""
function sum_by_country(m::AbstractMatrixEntry; dimension = :both)
    row_cc = _pick_col(names(m.row_indices), :CountryCode, :Country)
    col_cc = _pick_col(names(m.col_indices), :CountryCode, :Country)
    if dimension == :rows
        return groupby_matrix(m, Symbol(row_cc); rows = true)
    elseif dimension == :cols
        return groupby_matrix(m, Symbol(col_cc); rows = false)
    else  # both
        df = to_long_dataframe(m)
        return DataFrames.combine(
            DataFrames.groupby(df, [Symbol("row_" * string(row_cc)), Symbol("col_" * string(col_cc))]),
            :value => sum => :value
        )
    end
end

"""
    sum_by_sector(m::AbstractMatrixEntry; dimension=:both)

Sum matrix values by sector codes. Accepts either the `Sector` or the
`Industry` index column on each side (resolved independently for rows and
columns).
"""
function sum_by_sector(m::AbstractMatrixEntry; dimension = :both)
    row_sc = _pick_col(names(m.row_indices), :Sector, :Industry)
    col_sc = _pick_col(names(m.col_indices), :Sector, :Industry)
    if dimension == :rows
        return groupby_matrix(m, Symbol(row_sc); rows = true)
    elseif dimension == :cols
        return groupby_matrix(m, Symbol(col_sc); rows = false)
    else  # both
        df = to_long_dataframe(m)
        return DataFrames.combine(
            DataFrames.groupby(df, [Symbol("row_" * string(row_sc)), Symbol("col_" * string(col_sc))]),
            :value => sum => :value
        )
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
function add_calculated_column(m::AbstractMatrixEntry, col_name::Symbol, calculation_func; to_rows = true)
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

Requires unique `(row_vars..., col_var)` key combinations: duplicate pivot
keys throw an informative `ArgumentError`. Aggregate first (e.g. with
`groupby_matrix`) when the long-form data holds several values per key.
"""
function pivot_matrix_to_wide(m::AbstractMatrixEntry, row_vars, col_var, value_var = "value")
    row_vars = row_vars isa Symbol ? [row_vars] : collect(row_vars)
    df = to_long_dataframe(m; value_name = value_var)
    row_id_cols = [Symbol("row_" * string(var)) for var in row_vars]
    col_id_col = Symbol("col_" * string(col_var))
    key_cols = vcat(row_id_cols, [col_id_col])
    if nrow(unique(df[!, key_cols])) != nrow(df)
        throw(
            ArgumentError(
                "pivot_matrix_to_wide requires unique (row, column) key combinations, " *
                    "but duplicate entries were found for row_vars=$(collect(string.(row_vars))) " *
                    "and col_var=$(string(col_var)); aggregate first (e.g. with groupby_matrix)"
            )
        )
    end
    return DataFrames.unstack(df, row_id_cols, col_id_col, Symbol(value_var))
end

"""
    matrix_summary(m::AbstractMatrixEntry)

Generate comprehensive summary statistics for the matrix data.

Throws an `ArgumentError` for empty (zero-element) matrices. Note that `std`
is `NaN` for a single element (Statistics.std normalization); that behavior
is kept as-is.
"""
function matrix_summary(m::AbstractMatrixEntry)
    isempty(m.data) && throw(
        ArgumentError("matrix_summary requires a non-empty matrix (got size $(size(m.data)))")
    )
    df = to_long_dataframe(m)
    val = df.value
    return DataFrame(
        total = sum(val),
        mean = mean(val),
        median = median(val),
        std = std(val),
        min_val = minimum(val),
        max_val = maximum(val),
        n_nonzero = sum(val .!= 0),
        n_total = length(val)
    )
end

"""
    country_summary(m::AbstractMatrixEntry)

Generate country-by-country flow summary for bilateral analysis. Accepts
either the `CountryCode` or the `Country` index column on each side.
"""
function country_summary(m::AbstractMatrixEntry)
    df = to_long_dataframe(m)
    row_cc = _pick_col(names(m.row_indices), :CountryCode, :Country)
    col_cc = _pick_col(names(m.col_indices), :CountryCode, :Country)
    res = DataFrames.combine(
        DataFrames.groupby(df, [Symbol("row_" * string(row_cc)), Symbol("col_" * string(col_cc))]),
        :value => sum => :total_flow,
        :value => mean => :mean_flow,
        :value => length => :n_sectors
    )
    return sort!(res, :total_flow, rev = true)
end

function _check_environmental_production_length(env::EnvironmentalExtension, n::Integer)
    expected = size(env.A.data, 2)
    if n != expected
        throw(DimensionMismatch("production has length $n, expected $expected"))
    end
    return nothing
end

function _scenario_indices(n::Integer)::DataFrame
    return DataFrame(Scenario = collect(1:n))
end

"""
    environmental_impact(env::EnvironmentalExtension, production)
    environmental_impact(mrio::MRIO, production)

Calculate environmental impacts from a production vector or matrix.

For vector input, returns a `SeriesEntry` indexed by environmental stressors.
For matrix input, columns are treated as production scenarios and a `MatrixEntry`
is returned with stressors as rows.
"""
function environmental_impact(env::EnvironmentalExtension, production::AbstractVector{<:Real})
    _check_environmental_production_length(env, length(production))

    impact = Vector{Float64}(undef, size(env.A.data, 1))
    mul!(impact, env.A.data, production)

    return SeriesEntry(impact, env.A.row_indices)
end

function environmental_impact(mrio::MRIO, production::AbstractVector{<:Real})
    return environmental_impact(_require_env(mrio), production)
end

function environmental_impact(env::EnvironmentalExtension, production::SeriesEntry)
    return environmental_impact(env, production.data)
end

function environmental_impact(mrio::MRIO, production::SeriesEntry)
    return environmental_impact(_require_env(mrio), production)
end

function environmental_impact(env::EnvironmentalExtension, production::AbstractMatrix{<:Real})
    _check_environmental_production_length(env, size(production, 1))

    impact = Matrix{Float64}(undef, size(env.A.data, 1), size(production, 2))
    mul!(impact, env.A.data, production)

    return MatrixEntry(impact, _scenario_indices(size(production, 2)), env.A.row_indices)
end

function environmental_impact(mrio::MRIO, production::AbstractMatrix{<:Real})
    return environmental_impact(_require_env(mrio), production)
end

function environmental_impact(env::EnvironmentalExtension, production::MatrixEntry)
    _check_environmental_production_length(env, size(production.data, 1))

    impact = Matrix{Float64}(undef, size(env.A.data, 1), size(production.data, 2))
    mul!(impact, env.A.data, production.data)

    return MatrixEntry(impact, production.col_indices, env.A.row_indices)
end

function environmental_impact(mrio::MRIO, production::MatrixEntry)
    return environmental_impact(_require_env(mrio), production)
end

"""
    induced_production(mrio::MRIO; consumer_countries::Vector{String}=String[], producer_countries::Vector{String}=String[])

Calculate the production induced by the final demand of specified consumer countries
on specified producer countries using the Leontief Inverse matrix.

If `consumer_countries` is empty, final demand from all countries is included.
If `producer_countries` is empty, output for all producing countries is returned.

Country matching accepts any values convertible with `string` (e.g. integer
codes). When a non-empty request matches nothing, an `ArgumentError` listing
the unmatched names is thrown; partial matches emit a `@warn` listing the
unmatched names. Throws an `ArgumentError` when the MRIO has no Leontief
factorization (non-square system). Accepts either the `CountryCode`/`Country`
and `Sector`/`Industry` index columns. The returned `DataFrame` columns are
always named `CountryCode` and `Sector`, even when the underlying index
columns are named `Country`/`Industry`.
"""
function induced_production(
        mrio::MRIO;
        consumer_countries::AbstractVector = String[],
        producer_countries::AbstractVector = String[]
    )
    L = _require_leontief(mrio)
    consumer_requested = string.(consumer_countries)
    producer_requested = string.(producer_countries)

    # 1. Identify columns in Y corresponding to the consumer countries
    y_country_col = _pick_col(names(mrio.Y.col_indices), :CountryCode, :Country)
    y_cols = mrio.Y.col_indices[!, y_country_col]

    # If consumer_countries is empty, include all consumers
    consumer_mask = _country_mask(y_cols, consumer_requested, "consumer")

    # 2. Get the final demand submatrix and sum rows to get a vector
    y_demand = sum(mrio.Y.data[:, consumer_mask], dims = 2)[:]

    # 3. Calculate induced production: x = L * y_demand (solving linear system)
    x_induced = L.factorization \ y_demand

    # 4. Filter for producer countries
    row_indices = L.row_indices
    row_country_col = _pick_col(names(row_indices), :CountryCode, :Country)
    row_sector_col = _pick_col(names(row_indices), :Sector, :Industry)

    # If producer_countries is empty, include all producers
    producer_mask = _country_mask(row_indices[!, row_country_col], producer_requested, "producer")

    # 5. Build and return DataFrame
    df = DataFrame(
        CountryCode = row_indices[producer_mask, row_country_col],
        Sector = row_indices[producer_mask, row_sector_col],
        InducedProduction = x_induced[producer_mask]
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
        consumer_countries = string.(consumer_countries),
        producer_countries = string.(producer_countries)
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
