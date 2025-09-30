module Juliora

using Tidier
using TidierFiles
using LinearAlgebra

struct MatrixEntry
  data::Matrix{Float64}
  col_indices::DataFrame
  row_indices::DataFrame
  function MatrixEntry(data, col_indices, row_indices)
    @assert size(data) == (size(row_indices)[1], size(col_indices)[1])
    new(data, col_indices, row_indices)
  end
end

Base.getindex(m::MatrixEntry,col)

struct SeriesEntry
  data::Vector{Float64}
  col_indices::DataFrame
end

struct EnvironmentalExtension
  A::MatrixEntry
  F::MatrixEntry
end

function EnvironmentalExtension(path::String, x)
  f = Matrix(read_csv(path * "Q.txt", col_names=false, delim="\t"))
  t_indices = @chain read_csv(path * "labels_T.txt", delim="\t", col_names=false) begin
    @select(CountryCode = Column2, Industry = Column3, Sector = Column4)
  end
  f_indices = @chain read_csv(path * "labels_Q.txt", delim="\t", col_names=false) begin
    @select(Stressor = Column1, Source = Column2)
  end

  EnvironmentalExtension(MatrixEntry(f, t_indices, f_indices), MatrixEntry(f ./ x', t_indices, f_indices))
end

struct Eora
  A::MatrixEntry
  T::MatrixEntry
  VA::MatrixEntry
  FD::MatrixEntry
  L::MatrixEntry
  X::SeriesEntry
  env::EnvironmentalExtension
end

function Eora(path::String)
  t = Matrix(read_csv(path * "T.txt", col_names=false, delim="\t"))
  t_indices = @chain read_csv(path * "labels_T.txt", delim="\t", col_names=false) begin
    @select(CountryCode = Column2, Industry = Column3, Sector = Column4)
  end


  v = Matrix(read_csv(path * "VA.txt", col_names=false, delim="\t"))
  v_colnames = @chain read_csv(path * "labels_VA.txt", delim="\t", col_names=false) begin
    @select(PrimaryInput = Column2)
  end

  y = Matrix(read_csv(path * "FD.txt", col_names=false, delim="\t"))
  y_indices = @chain read_csv(path * "labels_FD.txt", delim="\t", col_names=false) begin
    @select(CountryCode = Column2, Industry = Column3, Category = Column4)
  end


  x = vec(sum(t, dims=2) + sum(y, dims=2))
  a = t ./ replace(x, 0.0 => 1.0)

  Eora(
    MatrixEntry(a, t_indices, t_indices),
    MatrixEntry(t, t_indices, t_indices),
    MatrixEntry(v, t_indices, v_colnames),
    MatrixEntry(y, y_indices, t_indices),
    MatrixEntry(inv(I - a), t_indices, t_indices),
    SeriesEntry(x, t_indices),
    EnvironmentalExtension(path, x)
  )
end

end # module Juliora

