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
    F::MatrixEntry
    A::MatrixEntry
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
# Load environmental data (requires actual data files)
env_ext = EnvironmentalExtension("data/2017/", total_output_vector)

# Access CO2 intensities
co2_intensity = filter_rows(env_ext.A, row -> row.Stressor == "CO2")
```

```jldoctest
julia> using DataFrames

# Create mock environmental data for testing
julia> f_data = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3 stressors × 2 sectors
3×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0
 5.0  6.0

julia> sector_indices = DataFrame(CountryCode=["USA", "CHN"], Industry=["Agr", "Man"], Sector=["Primary", "Secondary"]);

julia> stressor_indices = DataFrame(Stressor=["CO2", "Water", "Land"], Source=["Fossil", "Fresh", "Arable"]);

julia> x_output = [10.0, 20.0];  # Total output by sector

# Create environmental extension manually (bypassing file loading)
julia> f_matrix = MatrixEntry(f_data, sector_indices, stressor_indices);

julia> a_matrix = MatrixEntry(f_data ./ x_output', sector_indices, stressor_indices);

julia> env_ext = EnvironmentalExtension(a_matrix, f_matrix);

julia> size(env_ext.F.data)
(3, 2)
```
"""
function EnvironmentalExtension(path::String, x, t_indices, row_mask)
    f_data = CSV.read(path * "Q.txt", Tables.matrix, header = false)
    f_indices = @chain read_csv(path * "labels_Q.txt", delim = "\t", col_names = false) begin
        @select(Stressor = Column1, Source = Column2)
    end
    f = MatrixEntry(f_data[:, row_mask], t_indices, f_indices)

    return EnvironmentalExtension(f, calculate_technical_coefficients(f, x))
end
