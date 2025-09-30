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
using DataFrames

include("seriesentry.jl")
include("matrixentry.jl")



# Additional TidierData convenience methods


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

