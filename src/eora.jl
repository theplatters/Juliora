

"""
	Eora

Complete Eora global multi-region input-output (MRIO) database structure.

# Fields
- `A::MatrixEntry`: Technical coefficients matrix (intermediate inputs per unit output)
- `T::MatrixEntry`: Intermediate transaction matrix (monetary flows between sectors)
- `VA::MatrixEntry`: Value added matrix (primary inputs by sector)
- `FD::MatrixEntry`: Final demand matrix (consumption, investment, government, exports)
- `L::LeontiefFactorization`: Leontief inverse matrix (total requirements matrix)
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
	L::LeontiefFactorization
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
	# Parallel file reading using Threads.@spawn
	t_task = Threads.@spawn CSV.read(path * "T.txt", Tables.matrix, header = false)
	t_indices_task = Threads.@spawn @chain read_csv(path * "labels_T.txt", delim = "\t", col_names = false) begin
		@select(CountryCode = Column2, Industry = Column3, Sector = Column4)
	end

	v_task = Threads.@spawn CSV.read(path * "VA.txt", Tables.matrix, header = false)
	v_colnames_task = Threads.@spawn @chain read_csv(path * "labels_VA.txt", delim = "\t", col_names = false) begin
		@select(PrimaryInput = Column2)
	end

	y_task = Threads.@spawn CSV.read(path * "FD.txt", Tables.matrix, header = false)
	y_indices_task = Threads.@spawn @chain read_csv(path * "labels_FD.txt", delim = "\t", col_names = false) begin
		@select(CountryCode = Column2, Industry = Column3, Category = Column4)
	end


	t_indices = fetch(t_indices_task)
	t_matrix = fetch(t_task)

	row_mask = t_indices.CountryCode .!= "ROW"
	t_indices_clean = t_indices[row_mask, :]
	t = MatrixEntry(t_matrix[row_mask, row_mask], t_indices_clean, t_indices_clean)



	y_indices = fetch(y_indices_task)

	y_mask = y_indices.CountryCode .!= "ROW"
	y_indices_clean = y_indices[y_mask, :]
	y = MatrixEntry(fetch(y_task)[row_mask, y_mask], y_indices_clean, t_indices_clean)
	x = calculate_total_output(t.data, y.data)
	a = calculate_technical_coefficients(t, x)
	l = calculate_leontief_factorization(a)
	v = fetch(v_task)
	v_colnames = fetch(v_colnames_task)

	Eora(
		a,
		t,
		MatrixEntry(v[:, row_mask], t_indices_clean, v_colnames),
		y,
		l,
		SeriesEntry(x, t_indices_clean),
		EnvironmentalExtension(path, x, t_indices_clean, row_mask),
	)
end


calculate_leontief_inverse(a::MatrixEntry) = MatrixEntry(inv(I - a.data), a.col_indices, a.row_indices)
calculate_technical_coefficients(T::MatrixEntry, x) = MatrixEntry(T.data ./ replace(x, 0.0 => 1.0)', T.col_indices, T.row_indices)
function calculate_total_output(t_data, y_data)
	x = Vector{Float64}(undef, size(t_data, 1))
	@inbounds for i in 1:size(t_data, 1)
		x[i] = sum(view(t_data, i, :)) + sum(view(y_data, i, :))
	end
	return x
end

function Eora(; Z::MatrixEntry, Y::MatrixEntry, VA::MatrixEntry)
	if size(Z.data, 1) == size(Z.data, 2)
		x = calculate_total_output(Z.data, Y.data)
		a = calculate_technical_coefficients(Z, x)
		l = calculate_leontief_factorization(a)
		dummy_env = EnvironmentalExtension(Z, a)
		return Eora(
			a,
			Z,
			VA,
			Y,
			l,
			SeriesEntry(x, Z.row_indices),
			dummy_env
		)
	else
		x = zeros(size(Z.data, 2))
		a = calculate_technical_coefficients(Z, x)
		l = LeontiefFactorization(lu(Matrix{Float64}(I, 1, 1)), Z.col_indices, Z.row_indices)
		dummy_env = EnvironmentalExtension(Z, a)
		return Eora(
			a,
			Z,
			VA,
			Y,
			l,
			SeriesEntry(x, Z.col_indices),
			dummy_env
		)
	end
end

function Base.getproperty(eora::Eora, sym::Symbol)
	if sym === :Z
		return getfield(eora, :T)
	elseif sym === :Y
		return getfield(eora, :FD)
	else
		return getfield(eora, sym)
	end
end
