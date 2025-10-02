using Juliora
using DataFrames
using CSV
using TidierFiles

a = Juliora.Eora("data/2017/");
a

Base.summarysize(a)
ag = Juliora.groupby(a.A, [:Sector], dims = 2)

Juliora.aggregate(ag, sum)
file

@time CSV.read("data/2017/FD.txt", Tables.matrix, header=false)
@time Matrix(read_csv("data/2017/FD.txt", col_names = false, delim = "\t"))