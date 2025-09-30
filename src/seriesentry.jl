
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
struct SeriesEntry
	data::Vector{Float64}
	col_indices::DataFrame
    col_lookup::Dict{NamedTuple, Int}
	function SeriesEntry(data, col_indices)
        @assert length(data) == nrow(col_indices)
		col_lookup = Dict(NamedTuple(row) => i for (i, row) in enumerate(eachrow(col_indices)))
		new(data, col_indices, col_lookup)
	end
end

function Base.getindex(m::SeriesEntry, col_key::NamedTuple)
	col_idx = get(m.col_lookup, col_key, nothing)

	isnothing(col_idx) && throw(BoundsError(m, col_key))

	return m.data[col_idx]
end

