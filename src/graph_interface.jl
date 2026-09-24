"""
    CommunityResult{TM}

Result of a community-detection run over an MRIO graph.

# Fields
- `membership::Vector{Int32}`: community label per node (position `i` is the
  community of node `i`). Labels are contiguous `1:k` in order of first node
  appearance.
- `modularity::Float64`: modularity of `membership` under `resolution`.
- `algorithm::Symbol`: detection algorithm
  (`:louvain`, `:leiden`, `:label_propagation` or `:spectral`).
- `resolution::Float64`: resolution parameter the partition was found with.
- `seed::Union{Nothing,Int}`: RNG seed used, or `nothing` for a random run.
- `nodes::DataFrame`: shared reference to the node metadata of the graph the
  partition was computed on (no copy — mutating it mutates the graph's table).
- `weights::TM`: shared reference to the dense weight matrix the partition was
  computed on (`===` the graph's matrix; never copied). The parameter
  `TM <: AbstractMatrix` keeps every dense element type working (a
  `Matrix{Float32}` graph yields a `CommunityResult{Matrix{Float32}}`).
- `filter::Tuple{Float64,Float64,Bool,Float64}`: the graph's
  `(threshold, min_share, self_loops, scale)` filter, with the same
  effective-weight semantics as `mrio_graph`.
- `directed::Bool`: directedness of the graph the partition was computed on.

This type lives in the main `Juliora` module so it is available without
loading the Graphs.jl package extension. The `communities` implementations
that produce it load with `using Graphs` (Graphs must be installed in the
active environment).
"""
struct CommunityResult{TM <: AbstractMatrix}
    membership::Vector{Int32}
    modularity::Float64
    algorithm::Symbol
    resolution::Float64
    seed::Union{Nothing, Int}
    nodes::DataFrame
    weights::TM
    filter::Tuple{Float64, Float64, Bool, Float64}
    directed::Bool
end

"""
    _graph_extension_error(fname::Symbol)

Throw the standard "extension not loaded" error for graph API stub `fname`.
"""
function _graph_extension_error(fname::Symbol)
    error(
        "$fname requires the Graphs.jl package extension, which is not loaded. " *
            "Run `using Graphs` (Graphs must be installed in the active environment) to enable graph features.",
    )
end

"""
    mrio_graph(mrio; source=nothing, weights=nothing, direction=:directed, threshold=0.0, min_share=0.0, self_loops=false)
    mrio_graph(W::AbstractMatrix, nodes::DataFrame; direction=:directed, threshold=0.0, min_share=0.0, self_loops=false)

Build a zero-copy graph view over a dense MRIO matrix.

An edge `i → j` means "monetary flow supplied by node `i` to buyer `j`"
(`Z[i, j]` of the wrapped transactions matrix). Self-flows are dropped by
default (`self_loops=false`). `threshold` is an absolute cutoff on `|w|` and
`min_share` is a minimum share of `Σ|W|`; both are applied on the fly inside
the graph kernels, never as a materialized pruned matrix.

Returns an `MRIOGraph` (a `Graphs.AbstractGraph`). The wrapped matrix is
stored by reference, never copied.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
mrio_graph(args...; kwargs...) = _graph_extension_error(:mrio_graph)

"""
    graph_summary(g) -> DataFrame

Summarize an `MRIOGraph` in a one-row `DataFrame` with columns `nodes`,
`edges` (post-filter edge count), `directed`, `threshold`, `min_share`,
`self_loops`, `total_weight`, `retained_weight`, `retained_share` and
`memory_bytes` (the memory the graph references, shared with the MRIO;
construction itself allocates no copy).

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
graph_summary(args...; kwargs...) = _graph_extension_error(:graph_summary)

"""
    to_simple_graph(g; threshold=0.0, topk=nothing) -> (simple_graph, distmx)

Opt-in extraction of an `MRIOGraph` into a `Graphs.SimpleDiGraph`/`SimpleGraph`
plus a sparse `distmx` (`SparseMatrixCSC{Float32}`) holding the extracted edge
weights for Graphs.jl weight-aware generics. `threshold` is an additional
absolute cutoff applied on top of the graph filter; `topk` keeps, per source
node (directed) or per incident endpoint (undirected), only the `topk`
heaviest out-/incident edges by `|w|`.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
to_simple_graph(args...; kwargs...) = _graph_extension_error(:to_simple_graph)

"""
    communities(g; algorithm=:louvain, resolution=1.0, nruns=1, seed=nothing, ncommunities=nothing) -> CommunityResult
    communities(mrio; source=nothing, weights=nothing, direction=:undirected, threshold=0.0, min_share=0.0, self_loops=false, algorithm=:louvain, resolution=1.0, nruns=1, seed=nothing, ncommunities=nothing) -> CommunityResult

Detect communities in an MRIO network. Returns a [`CommunityResult`](@ref).

Community detection runs on the symmetrized (undirected) flow weights
`w(i, j) = filt(W[i, j] + W[j, i])` by default — the MRIO method's
`direction` defaults to `:undirected`. Pass `direction = :directed` (or build
the graph yourself with `mrio_graph`) to detect on the directed flows under
directed (Leicht–Newman) modularity instead. The graph method always operates
on the effective weights of `g` exactly as built.

# Keyword arguments
- `algorithm`: one of `:louvain`, `:leiden`, `:label_propagation`,
  `:spectral` (default `:louvain`). `:label_propagation` is weight-blind: it
  runs `Graphs.label_propagation` on the unweighted topology extracted via
  `to_simple_graph` (which preserves the graph filter exactly).
- `resolution::Real` (default `1.0`): modularity resolution parameter `γ`;
  must be `> 0` (`ArgumentError` otherwise).
- `nruns::Integer` (default `1`): number of independent runs; must be `≥ 1`.
  The partition with the largest modularity is kept (ties → the earlier run).
- `seed::Union{Nothing,Integer}` (default `nothing`): run `r` uses
  `Random.MersenneTwister(Int(seed) + r)`; `nothing` uses
  `Random.default_rng()`.
- `ncommunities::Union{Nothing,Integer}` (default `nothing`): target community
  count, used by `algorithm = :spectral` only — passing it with any other
  algorithm throws an `ArgumentError`.
- MRIO method only: `source`, `weights`, `direction`, `threshold`, `min_share`,
  `self_loops` select the wrapped matrix and graph filter exactly as in
  `mrio_graph` (which the MRIO method builds and forwards to). Note that
  `direction` defaults to `:undirected` here (unlike `mrio_graph` and
  `pagerank_scores`): community detection defaults to the symmetrized flow
  weights.

!!! warning
    Graph code must never touch `LeontiefFactorization.data`: that accessor
    materializes a dense n×n matrix inverse. Community detection (and all
    other graph routines) operate on the wrapped transactions/technical
    matrix only; graph-only workflows can use `mrio_graph(W, nodes)` to skip
    the Leontief factorization entirely.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
communities(args...; kwargs...) = _graph_extension_error(:communities)

"""
    community_table(result::CommunityResult) -> DataFrame

Return the node → community mapping of a [`CommunityResult`](@ref) joined
with the node metadata.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
community_table(args...; kwargs...) = _graph_extension_error(:community_table)

"""
    community_summary(result::CommunityResult) -> DataFrame

Summarize a [`CommunityResult`](@ref) with one row per community (sorted by
`community` ascending) and columns `community::Int32`, `size::Int`,
`internal_flow::Float64`, `external_flow::Float64`,
`internal_share::Float64`, plus — when the corresponding metadata column
exists in `result.nodes` — `n_countries::Int`, `top_country::String`,
`top_country_share::Float64` (from `CountryCode`, falling back to `Country`)
and `n_sectors::Int`, `top_sector::String`, `top_sector_share::Float64`
(from `Sector`, falling back to `Industry`).

Flows use the graph's effective weights `A`: `internal_flow(c)` is the double
sum `Σ_{i,j ∈ c} A_ij` (a surviving diagonal contributes once) and
`external_flow(c) = Σ_{i∈c,j∉c} A_ij + Σ_{i∉c,j∈c} A_ij`;
`internal_share = internal / (internal + external)` (`0.0` when both are
zero). `top_country`/`top_sector` are the most frequent metadata values among
the community's members (ties → first in node order) and the `*_share`
columns are their fractions of the community size.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
community_summary(args...; kwargs...) = _graph_extension_error(:community_summary)

"""
    pagerank_scores(g::MRIOGraph; damping=0.85, weighted=true, tol=1.0e-6, max_iter=100) -> SeriesEntry
    pagerank_scores(mrio::MRIO; source=nothing, weights=nothing, direction=:directed, threshold=0.0, min_share=0.0, self_loops=false, damping=0.85, weighted=true, tol=1.0e-6, max_iter=100) -> SeriesEntry

Compute (flow-weighted) PageRank scores over the directed MRIO graph.
Returns a `SeriesEntry` over the node metadata (the graph's shared node
`DataFrame` reference, never copied).

Weighted path (`weighted=true`, default): power iteration on the effective
weights `w(i, j)` of the graph (directed: filtered flows; undirected:
symmetrized weights) with out-strength normalization, uniform teleportation,
and dangling-node (zero out-strength) mass redistributed uniformly. Scores
sum to 1 and are non-negative. Effective weights must be finite and
non-negative (an `ArgumentError` covering negative values, NaN and ±Inf and
naming the first offending pair `(i, j)` is thrown otherwise; values pruned
by the filter cannot violate this); non-convergence throws an
`ErrorException`.

Unweighted path (`weighted=false`): delegates to
`Graphs.pagerank(simple, damping, max_iter, tol)` on the extracted simple
graph (undirected self-loops are dropped, consistent with
`to_simple_graph`); non-convergence raises Graphs' `ErrorException`.

Requires `0 < damping < 1`, `tol > 0`, `max_iter >= 1` and a non-empty graph
(`ArgumentError` otherwise). The MRIO method builds
`mrio_graph(mrio; source, weights, direction, threshold, min_share,
self_loops)` and forwards to the graph method.

!!! warning
    Graph code must never touch `LeontiefFactorization.data`: that accessor
    materializes a dense n×n matrix inverse. PageRank (and all other graph
    routines) operate on the wrapped transactions/technical matrix only;
    graph-only workflows can use `mrio_graph(W, nodes)` to skip the Leontief
    factorization entirely.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
pagerank_scores(args...; kwargs...) = _graph_extension_error(:pagerank_scores)

"""
    node_similarity(g_or_mrio; method=:cosine, on=:out, k=10) -> MRIOGraph

Compute top-k structural node similarity without ever allocating an n×n
result (row-blocked accumulation). Returns a compact kNN-weight `MRIOGraph`.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
node_similarity(args...; kwargs...) = _graph_extension_error(:node_similarity)

"""
    similarity_graph(g_or_mrio; k=10, symmetrize=:max, ...) -> MRIOGraph

Build a symmetric kNN similarity graph intended as input to `communities`.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
similarity_graph(args...; kwargs...) = _graph_extension_error(:similarity_graph)

"""
    compare_networks(g1, g2; match=:keys) -> DataFrame

Compare two MRIO networks on their shared nodes (edge overlap, strength
correlations, density/scale ratios). Returns a one-row `DataFrame`.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
compare_networks(args...; kwargs...) = _graph_extension_error(:compare_networks)

"""
    compare_partitions(c1, c2) -> DataFrame

Compare two community partitions (adjusted Rand index, normalized mutual
information, community counts, modularities). Returns a one-row `DataFrame`.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
compare_partitions(args...; kwargs...) = _graph_extension_error(:compare_partitions)
