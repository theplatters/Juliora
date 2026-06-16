using Juliora
using BenchmarkTools
using Profile
Juliora.Eora("data/2017/")

a = Juliora.Eora("data/2017/");
@time Juliora.Eora("data/2017/");
@profview Juliora.Eora("data/2017/")
norm(a.A.data)

ag = Juliora.groupby(a.A, [:Sector], dims = 2)
count(sum(a.A.data, dims = 1) .> 1.0)


Juliora.aggregate(ag, sum)

sum(ones(3, 3), dims = 2)
