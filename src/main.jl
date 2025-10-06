using Juliora
using BenchmarkTools
using Profile
Juliora.Eora("data/2017/")

@profview Juliora.Eora("data/2017/")
norm(a.A.data)

ag = Juliora.groupby(a.A, [:Sector], dims = 2)
count(sum(a.A.data, dims = 2) .> 1.0)

a.T

Juliora.drop(a.FD,(CountryCode="ROW", ); dims=2)
Juliora.aggregate(ag, sum)
