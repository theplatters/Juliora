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
    node_similarity(g::MRIOGraph; method=:cosine, on=:out, k=10, sources=nothing, damping=0.85, tol=1.0e-6, max_iter=100) -> MRIOGraph
    node_similarity(mrio::MRIO; source=nothing, weights=nothing, direction=:directed, threshold=0.0, min_share=0.0, self_loops=false, method=:cosine, on=:out, k=10, sources=nothing, damping=0.85, tol=1.0e-6, max_iter=100) -> MRIOGraph

Top-k structural node similarity over the graph's effective weights `A` —
the same filtered semantics as everything else in the library (directed
`g`: `A[i, j] = filtered_weight(W, i, j, cutoff, self_loops)`; undirected
`g`: `A[i, j] = sym_weight(W, i, j, cutoff, self_loops)`, symmetric): the
`threshold`/`min_share` filter applies (self-flows dropped by default,
`self_loops=false`).

# Methods

- `:cosine` (default): `S[i, j] = dot(p_i, p_j) / (‖p_i‖₂ · ‖p_j‖₂)` over
  the profiles `p_i` (`S[i, j] = 0` if either norm is `0`).
- `:jaccard`: `S[i, j] = |supp(p_i) ∩ supp(p_j)| / |supp(p_i) ∪ supp(p_j)|`
  on the binarized supports `b(x) = (x != 0)` (`0` if the union is empty).
- `:random_walk`: PPR-style relatedness for selected source nodes only —
  for each `s` in `sources`, personalized PageRank on the effective-weight
  transitions of `g` as built with teleport to `δ_s` and dangling-node mass
  redistributed to the personalization vector (standard
  Andersen–Chung–Lang PPR). The score vector `p_s` (visit probabilities,
  summing to 1) is the similarity of every node to `s`. Documented cost:
  one solve per source.

# Profiles (`on`; validated always, ignored by `:random_walk`)

- `:out` (default): row `i` of `A` (supplier profiles).
- `:in`: column `i` of `A` (buyer profiles).
- `:both`: the concatenation `[row_i; col_i]` (dots and supports add;
  never materialized).

# Top-k rule

Per row, excluding `j == i` (self-similarity never becomes an edge):
strictly positive values only (exact zeros are never edges), ranked by
descending similarity with ties broken towards the smaller `j`, keeping
the first `k` (`k` clamped to `n - 1`; rows with no positive candidates
get no out-edges). `:random_walk` keeps the top-k of each source score
vector only — other rows carry no edges.

# Keyword arguments

- `method`: one of `:cosine`, `:jaccard`, `:random_walk` (default
  `:cosine`; `ArgumentError` otherwise).
- `on`: one of `:out`, `:in`, `:both` (default `:out`; `ArgumentError`
  otherwise).
- `k`: neighbors kept per row — an Int-range `Integer` (`Bool` excluded)
  `≥ 1` (default `10`; `ArgumentError` otherwise).
- `sources`: `nothing` (default) unless `method === :random_walk`, which
  requires a non-empty vector of in-range node indices (duplicates dedupe
  to first occurrences, e.g. `[2, 1, 2]` → `[2, 1]`); passing `sources`
  with any other method throws an `ArgumentError`.
- `damping`: teleportation damping factor, `0 < damping < 1`
  (`ArgumentError` otherwise; validated exactly as in `pagerank_scores`,
  used by `:random_walk` only).
- `tol`/`max_iter`: convergence tolerance (`> 0`) and iteration cap
  (`≥ 1`) for `:random_walk` (defaults `1.0e-6`/`100`, as in
  `pagerank_scores`); non-convergence throws an `ErrorException`.
- MRIO method only: `source`, `weights`, `direction`, `threshold`,
  `min_share`, `self_loops` select the wrapped matrix and graph filter
  exactly as in `mrio_graph` (which the MRIO method builds and forwards
  to). Note that `direction` defaults to `:directed` here (profiles of
  the directed supply/purchase flows, `on = :out` = rows) — unlike
  `communities`, whose default is `:undirected`.

Effective weights must be finite and non-negative (an `ArgumentError`
naming the first offending pair `(i, j)` is thrown otherwise; values
pruned by the filter cannot violate this).

# Returns

A directed (`D = true`) `MRIOGraph` over a compact
`SparseMatrixCSC{Float32, Int32}` (n×n, at most `k` stored entries per
row) with `weights[i, j] = Float32(S[i, j])` per kept directed edge
`i → j`, built through `mrio_graph(W, nodes; direction = :directed)`.
`result.nodes === g.nodes` (shared reference — zero-copy contract).

# Memory

Row-blocked accumulation (block ≈ 64 rows, panel ≈ 64): no n×n result is
ever allocated (R6) — scratch is O((block + n)·panel + block·n) `Float64`
plus O(n) profile statistics and the O(k·n) compact result. The
implementation uses no RNG: repeated calls return identical results.

!!! warning
    Graph code must never touch `LeontiefFactorization.data`: that accessor
    materializes a dense n×n matrix inverse. Similarity (and all other graph
    routines) operate on the wrapped transactions/technical matrix only;
    graph-only workflows can use `mrio_graph(W, nodes)` to skip the Leontief
    factorization entirely.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
node_similarity(args...; kwargs...) = _graph_extension_error(:node_similarity)

"""
    similarity_graph(g::MRIOGraph; method=:cosine, on=:out, k=10, sources=nothing, damping=0.85, tol=1.0e-6, max_iter=100, symmetrize=:max) -> MRIOGraph
    similarity_graph(mrio::MRIO; source=nothing, weights=nothing, direction=:directed, threshold=0.0, min_share=0.0, self_loops=false, method=:cosine, on=:out, k=10, sources=nothing, damping=0.85, tol=1.0e-6, max_iter=100, symmetrize=:max) -> MRIOGraph

Symmetric kNN similarity graph over the graph's effective weights: run
[`node_similarity`](@ref) (same methods, profiles, top-k rule, validation
and compact directed result — see its documentation), then symmetrize the
kept directed edges. The undirected pair set is the union of kept directed
edges as unordered pairs `{i, j}` (`i ≠ j`); the pair weight `s` is the
maximum (`symmetrize = :max`, default) or the arithmetic mean
(`symmetrize = :mean`) over the present directed values among `S[i, j]`
and `S[j, i]` — a one-sided pair keeps its value under both rules
(`ArgumentError` for any other `symmetrize` value).

# Returns

An undirected (`D = false`) `MRIOGraph` over a compact
`SparseMatrixCSC{Float32, Int32}` with one-sided upper-triangular storage:
`weights[i, j] = Float32(s)` for `i < j`, `0` elsewhere, so the pair
weight reads back as `W[i, j] + W[j, i] = s` — read pair weights via
`Graphs.weights(g)[i, j]` (the established undirected fixture
convention). `result.nodes === g.nodes` (shared reference).

This is the reusable builder intended as input to `communities`:
`communities(similarity_graph(g; k = k); algorithm = :louvain)` clusters
nodes with similar flow profiles. Memory and determinism follow
`node_similarity` (row-blocked, no n×n result — R6; no RNG).

# Keyword arguments

All [`node_similarity`](@ref) keywords apply unchanged (including the
`:random_walk` `sources` requirement and per-source cost, and the MRIO
method's `direction = :directed` default), plus `symmetrize` (`:max` or
`:mean`, default `:max`).

!!! warning
    Graph code must never touch `LeontiefFactorization.data`: that accessor
    materializes a dense n×n matrix inverse. Similarity (and all other graph
    routines) operate on the wrapped transactions/technical matrix only;
    graph-only workflows can use `mrio_graph(W, nodes)` to skip the Leontief
    factorization entirely.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
similarity_graph(args...; kwargs...) = _graph_extension_error(:similarity_graph)

"""
    compare_networks(g1, g2; match=:keys, pagerank=false, damping=0.85) -> DataFrame

Compare two MRIO networks on their shared (matched) nodes. Graphs only:
filter choices are per-graph, so build graphs first with `mrio_graph`
(there is no MRIO pass-through).

All metrics are computed on the matched node set — the key intersection
ordered by `g1`'s node order; unmatched nodes contribute to nothing, not
even the densities or scales. `match === :keys` (default) uses every column
present in both `nodes` tables (zero shared columns throw an
`ArgumentError`); `match::Vector{Symbol}` names the key columns explicitly
(non-empty, each present in both tables); anything else throws an
`ArgumentError`. Keys must uniquely identify nodes in each graph (duplicate
keys throw an `ArgumentError` naming the key); key tuples compare with
`isequal` (missing-safe).

With `V_g[i, j]` the effective pair weight of graph `g` on the matched set
(read on the fly — never materialized) over all ordered matched pairs
`(i, j)` (diagonal included): weighted edge overlap
`Σ min(V_1, V_2) / Σ max(V_1, V_2)` (`NaN` when `Σ max == 0`); Pearson
(`Statistics.cor`) and Spearman (Pearson on average tied ranks) correlation
of the matched-set out/in strengths (`NaN` when degenerate); nonzero-pair
densities over `m^2` and total-weight scales with their ratios (Julia `/`
semantics: `NaN`/`Inf` propagate). `pagerank = true` (a `Bool`) adds
`pearson_pagerank`/`spearman_pagerank` from a PageRank solve per matched
subgraph (uniform teleport, `damping` with `0 < damping < 1`, the
`pagerank_scores` iteration defaults; `m == 0` gives `NaN` columns).

Returns a one-row `DataFrame` with columns (in order) `n_matched`, `n_1`,
`n_2`, `edge_overlap`, `pearson_out`, `spearman_out`, `pearson_in`,
`spearman_in`, `density_1`, `density_2`, `density_ratio`, `scale_1`,
`scale_2`, `scale_ratio` (`n_*` are `Int`, the rest `Float64`), plus
`pearson_pagerank`, `spearman_pagerank` when `pagerank = true`. Effective
weights of both graphs must be finite and non-negative (`ArgumentError`
naming the first offending pair otherwise). Comparing graphs built with
different `direction` mixes pair-weight conventions — use the same
`direction` for interpretable ratios. Elementwise passes over the dense
matrices: O(n²) time, O(1) extra memory (the matched set is accessed
through index maps and a lazy view — no submatrix is ever materialized).

!!! warning
    Graph code must never touch `LeontiefFactorization.data`: that accessor
    materializes a dense n×n matrix inverse. Comparison (and all other graph
    routines) operate on the wrapped transactions/technical matrices only;
    graph-only workflows can use `mrio_graph(W, nodes)` to skip the Leontief
    factorization entirely.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
compare_networks(args...; kwargs...) = _graph_extension_error(:compare_networks)

"""
    compare_partitions(c1, c2; match=:keys) -> DataFrame
    compare_partitions(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) -> DataFrame

Compare two community partitions (adjusted Rand index, normalized mutual
information, community counts, modularities). Membership labels may be
arbitrary integers. Returns a one-row `DataFrame` with columns (in order)
`n_matched`, `n_1`, `n_2`, `ari`, `nmi`, `n_communities_1`,
`n_communities_2`, `modularity_1`, `modularity_2` (counts `Int`, the rest
`Float64`).

`CommunityResult` inputs align via the node key columns exactly as in
`compare_networks` (`match === :keys` uses every shared column; explicit
`match::Vector{Symbol}` otherwise; uniqueness required; matched items are
the key intersection in first-input order; counts are over matched items,
`n_1`/`n_2` the full `nrow(c.nodes)` totals). Raw label vectors compare
positionally (equal lengths required; no `match` keyword; `Integer` eltype
with `Bool` excluded), as do the mixed `CommunityResult` × vector forms
(`modularity` is `NaN` on a vector side).

ARI is Hubert–Arabie over the contingency table
(`(ΣC(n,2) − ΣC(a)·ΣC(b)/P) / (0.5·(ΣC(a) + ΣC(b)) − ΣC(a)·ΣC(b)/P)`, zero
denominator → `1.0`); NMI is `2·I / (H_1 + H_2)` with natural logs (`H_1 +
H_2 == 0` → `1.0`); fewer than 2 aligned items give `(1.0, 1.0)`.
Contingency-table counting runs in O(n) memory; the matched items are
accessed through index maps — no submatrix is ever materialized.

!!! warning
    Graph code must never touch `LeontiefFactorization.data`: that accessor
    materializes a dense n×n matrix inverse. Comparison (and all other graph
    routines) operate on the wrapped transactions/technical matrices only;
    graph-only workflows can use `mrio_graph(W, nodes)` to skip the Leontief
    factorization entirely.

This is a stub: the implementation loads with `using Graphs` (Graphs must be
installed in the active environment).
"""
compare_partitions(args...; kwargs...) = _graph_extension_error(:compare_partitions)
