"""
	SeriesEntry

A structure for storing a vector of numerical data with labeled indices,
optimized for economic time series or cross-sectional data.

# Fields
- `data::Vector{Float64}`: The numerical vector data
- `col_indices::DataFrame`: DataFrame containing labels and metadata for each element

# Examples
```jldoctest
julia> using DataFrames

julia> data = [100.0, 200.0, 150.0];

julia> indices = DataFrame(Country=["USA", "CHN", "DEU"], Sector=["Total", "Total", "Total"]);

julia> series = SeriesEntry(data, indices);

julia> series.data
3-element Vector{Float64}:
 100.0
 200.0
 150.0

julia> series.col_indices.Country
3-element Vector{String}:
 "USA"
 "CHN"
 "DEU"
```
"""
struct SeriesEntry{T}
    data::T
    col_indices::DataFrame
    col_lookup::Dict{NamedTuple, Int}
end

function SeriesEntry(data::T, col_indices::DataFrame) where {T}
    @assert length(data) == nrow(col_indices)
    col_lookup = Dict(NamedTuple(row) => i for (i, row) in enumerate(eachrow(col_indices)))
    return SeriesEntry{T}(data, col_indices, col_lookup)
end

function SeriesEntry(data, col_indices)
    return SeriesEntry(data, safe_dataframe(col_indices))
end


function Base.getindex(m::SeriesEntry, col_key::NamedTuple)
    col_idx = get(m.col_lookup, col_key, nothing)

    isnothing(col_idx) && throw(BoundsError(m, col_key))

    return m.data[col_idx]
end

struct GroupedSeriesEntry
    original::SeriesEntry
    grouped::GroupedDataFrame
    cols
end

function groupby(s::SeriesEntry, cols)
    grouped = DataFrames.groupby(s.col_indices, cols)
    return GroupedSeriesEntry(s, grouped, cols)
end

function aggregate(gs::GroupedSeriesEntry, func::Function = sum)
    ind = groupindices(gs.grouped)
    groups = unique(ind)
    new_data = [func(gs.original.data[ind .== g]) for g in groups]
    new_col_indices = unique(select(gs.original.col_indices, groupcols(gs.grouped)))
    return SeriesEntry(new_data, new_col_indices)
end
