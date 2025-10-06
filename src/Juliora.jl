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
using CSV


include("seriesentry.jl")
include("matrixentry.jl")
include("LeontiefFactorization.jl")
include("environmental_extension.jl")
include("eora.jl")
include("aggregation.jl")


# Additional TidierData convenience methods



end # module Juliora

