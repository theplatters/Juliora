module JulioraGraphsExt

using Juliora
using Graphs
using DataFrames
using LinearAlgebra
using Random
using SparseArrays
using Statistics

import Juliora: mrio_graph, graph_summary, to_simple_graph, communities, community_table, community_summary, pagerank_scores, node_similarity, similarity_graph, compare_networks, compare_partitions

include("graph/dense_kernels.jl")
include("graph/mrio_graph.jl")
include("graph/pagerank.jl")
include("graph/similarity.jl")
include("graph/communities.jl")
include("graph/louvain_leiden.jl")
include("graph/spectral.jl")
include("graph/compare.jl")

end
