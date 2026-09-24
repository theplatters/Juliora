# Community detection over `MRIOGraph`: dispatch, modularity, and summaries.
#
# Memory rules: partitions are scored and summarized by streaming the effective
# weights `w(i, j)` on the fly from the wrapped dense matrix — no
# pruned/symmetrized n×n copy is ever materialized. `CommunityResult` shares
# the graph's weight matrix and node table by reference. The only extraction
# path is the weight-blind `:label_propagation` plugin, which runs
# `Graphs.label_propagation` on `to_simple_graph(g)` (opt-in extraction, R4).
# Graph code never touches `LeontiefFactorization.data`, which would
# materialize a dense n×n inverse.

"""
    _COMMUNITY_ALGORITHMS

Algorithms accepted by [`communities`](@ref): `:louvain` and `:leiden`
(multilevel plugins in `louvain_leiden.jl`), `:label_propagation`
(weight-blind, via the extracted topology) and `:spectral` (in
`spectral.jl`).
"""
const _COMMUNITY_ALGORITHMS = (:louvain, :leiden, :label_propagation, :spectral)

"""
    _validate_community_args(algorithm, resolution, nruns, seed, ncommunities, n) -> Float64

Validate the shared `communities` keyword arguments, throwing an
`ArgumentError` with an informative message otherwise: `algorithm` must be one
of `_COMMUNITY_ALGORITHMS`; `resolution` must be finite and `> 0`; `nruns` must
be an Int-range `Integer` (Bool excluded) `≥ 1`; `ncommunities` must
be `nothing` unless `algorithm === :spectral` (in which case it must be an
Int-range `Integer` (Bool excluded) `≥ 1` when given); `seed` must be
`nothing` or an Int-range `Integer` (Bool excluded); the graph must be
non-empty (`n ≥ 1`). Returns `Float64(resolution)`.
"""
function _validate_community_args(algorithm, resolution, nruns, seed, ncommunities, n)
  algorithm in _COMMUNITY_ALGORITHMS || throw(
    ArgumentError("algorithm must be one of $(_COMMUNITY_ALGORITHMS), got $algorithm"),
  )
  gamma = Float64(resolution)
  (gamma > 0 && isfinite(gamma)) ||
    throw(ArgumentError("resolution must be finite and > 0, got $resolution"))
  (nruns isa Integer && !(nruns isa Bool) && typemin(Int) <= nruns <= typemax(Int) && Int(nruns) >= 1) ||
    throw(ArgumentError("nruns must be ≥ 1 (Bool excluded), got $nruns"))
  if ncommunities !== nothing
    if algorithm !== :spectral
      throw(
        ArgumentError(
          "ncommunities is only used by algorithm = :spectral, got ncommunities = $ncommunities with algorithm = $algorithm",
        ),
      )
    end
    (ncommunities isa Integer && !(ncommunities isa Bool) && typemin(Int) <= ncommunities <= typemax(Int)) ||
      throw(
        ArgumentError(
          "ncommunities must be an Int-range Integer (Bool excluded) ≥ 1, got $ncommunities",
        ),
      )
    Int(ncommunities) >= 1 ||
      throw(ArgumentError("ncommunities must be ≥ 1, got $ncommunities"))
  end
  seed === nothing ||
    (seed isa Integer && !(seed isa Bool) && typemin(Int) <= seed <= typemax(Int)) ||
    throw(ArgumentError("seed must be nothing or an Int-range Integer (Bool excluded), got $seed"))
  Int(n) >= 1 || throw(ArgumentError("communities requires a non-empty graph (nv(g) = $n)"))
  return gamma
end

"""
    _run_community_plugin(g::MRIOGraph, algorithm, gamma, rng, ncommunities) -> Vector{Int32}

Run one pass of the selected algorithm plugin and return its raw membership
vector (arbitrary positive `Int32` labels; the caller normalizes via
[`_relabel_first_appearance`](@ref)). `:louvain`/`:leiden`/`:spectral`
delegate to their plugin files; `:label_propagation` extracts the unweighted
topology with `to_simple_graph(g)` — which preserves the graph filter exactly
— and runs `Graphs.label_propagation(simple, 1000; rng = rng)`. The algorithm
is weight-blind and reads only `outneighbors`, so edge weights play no role on
this path (undirected self-loops are dropped, consistent with
`to_simple_graph`).
"""
function _run_community_plugin(g::MRIOGraph, algorithm, gamma, rng, ncommunities)
  if algorithm === :louvain
    return _louvain_partition(g, gamma, rng)
  elseif algorithm === :leiden
    return _leiden_partition(g, gamma, rng)
  elseif algorithm === :label_propagation
    simple, _ = to_simple_graph(g)
    labels, _ = Graphs.label_propagation(simple, 1000; rng=rng)
    return Vector{Int32}(labels)
  else
    return _spectral_partition(g, ncommunities, rng)
  end
end

"""
    communities(g::MRIOGraph; algorithm=:louvain, resolution=1.0, nruns=1, seed=nothing, ncommunities=nothing) -> CommunityResult

Detect communities in `g`, running the selected algorithm plugin `nruns` times
(run `r` with `Random.MersenneTwister(Int(seed) + r)`, or
`Random.default_rng()` when `seed === nothing`) and keeping the partition with
the largest [`_partition_modularity`](@ref) value (ties → the earlier run).
The kept partition is normalized with [`_relabel_first_appearance`](@ref) and
returned in a [`CommunityResult`](@ref) sharing `g`'s node table and weight
matrix by reference (no copies). Effective weights must be finite and
non-negative (an `ArgumentError` naming the first offending pair `(i, j)` is
thrown otherwise; values pruned by the filter cannot violate this). See the
`communities` stub in `src/graph_interface.jl` for the keyword contract.
"""
function communities(
  g::MRIOGraph{TV,TI,D};
  algorithm=:louvain,
  resolution::Real=1.0,
  nruns::Integer=1,
  seed=nothing,
  ncommunities::Union{Nothing,Integer}=nothing,
) where {TV,TI,D}
  n = Graphs.nv(g)
  gamma = _validate_community_args(algorithm, resolution, nruns, seed, ncommunities, n)
  _check_nonnegative_weights(g, "communities")
  ntarget = ncommunities === nothing ? nothing : Int(ncommunities)
  best_membership = Vector{Int32}()
  best_q = -Inf
  @inbounds for r in 1:Int(nruns)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(Int(seed) + r)
    raw = _run_community_plugin(g, algorithm, gamma, rng, ntarget)
    membership = _relabel_first_appearance(raw)
    q = _partition_modularity(g, membership, gamma)
    if q > best_q
      best_q = q
      best_membership = membership
    end
  end
  return Juliora.CommunityResult(
    best_membership,
    best_q,
    algorithm,
    gamma,
    seed === nothing ? nothing : Int(seed),
    g.nodes,
    g.weights,
    g.filter,
    D,
  )
end

"""
    communities(mrio::MRIO; source=nothing, weights=nothing, direction=:undirected, threshold=0.0, min_share=0.0, self_loops=false, algorithm=:louvain, resolution=1.0, nruns=1, seed=nothing, ncommunities=nothing) -> CommunityResult

Build `mrio_graph(mrio; source, weights, direction, threshold, min_share,
self_loops)` and forward to `communities(g; algorithm, resolution, nruns,
seed, ncommunities)`. The wrapped transactions/technical matrix is used by
reference; the Leontief factorization is never touched. Returns a
[`CommunityResult`](@ref) over the selected matrix's supplier-side
`row_indices` (shared reference). `direction` defaults to `:undirected`
(unlike `mrio_graph`/`pagerank_scores`): community detection defaults to the
symmetrized flow weights `w(i, j) = filt(W[i, j] + W[j, i])`; pass
`direction = :directed` for detection on the directed flows under directed
(Leicht–Newman) modularity. See the `MRIOGraph` method for the run semantics
and the keyword contract.
"""
function communities(
  mrio::Juliora.MRIO;
  source::Union{Nothing,Symbol}=nothing,
  weights::Union{Nothing,Symbol}=nothing,
  direction::Symbol=:undirected,
  threshold::Real=0.0,
  min_share::Real=0.0,
  self_loops::Bool=false,
  algorithm=:louvain,
  resolution::Real=1.0,
  nruns::Integer=1,
  seed=nothing,
  ncommunities::Union{Nothing,Integer}=nothing,
)
  g = mrio_graph(
    mrio;
    source=source,
    weights=weights,
    direction=direction,
    threshold=threshold,
    min_share=min_share,
    self_loops=self_loops,
  )
  return communities(
    g;
    algorithm=algorithm,
    resolution=resolution,
    nruns=nruns,
    seed=seed,
    ncommunities=ncommunities,
  )
end

"""
    _relabel_first_appearance(membership::AbstractVector{<:Integer}) -> Vector{Int32}

Map labels to contiguous `1:k` in order of first node appearance, e.g.
`[7, 7, 3, 5, 3] -> Int32[1, 1, 2, 3, 2]`. Throws an `ArgumentError` on empty
input or on labels `< 1`.
"""
function _relabel_first_appearance(membership::AbstractVector{<:Integer})
  isempty(membership) && throw(
    ArgumentError("_relabel_first_appearance requires a non-empty membership vector"),
  )
  for v in membership
    v >= 1 || throw(
      ArgumentError("_relabel_first_appearance requires labels ≥ 1, got $v"),
    )
  end
  remap = Dict{eltype(membership),Int32}()
  out = Vector{Int32}(undef, length(membership))
  next_label = Int32(1)
  for (i, v) in enumerate(membership)
    existing = get(remap, v, Int32(0))
    if existing == Int32(0)
      remap[v] = next_label
      out[i] = next_label
      next_label += Int32(1)
    else
      out[i] = existing
    end
  end
  return out
end

"""
    _compact_membership(membership::AbstractVector{<:Integer}, n) -> (idx, k)

Validate `membership` against a graph with `n` nodes (a `DimensionMismatch`
unless `length(membership) == n`; an `ArgumentError` on labels `< 1`) and map
its labels to compact row indices `1:k`, returning the per-node row vector
`idx` and the community count `k`.
"""
function _compact_membership(membership::AbstractVector{<:Integer}, n::Int)
  length(membership) == n || throw(
    DimensionMismatch(
      "membership has length $(length(membership)) but the graph has $n nodes",
    ),
  )
  for v in membership
    v >= 1 ||
      throw(ArgumentError("_compact_membership requires labels ≥ 1, got $v"))
  end
  remap = Dict{eltype(membership),Int}()
  idx = Vector{Int}(undef, n)
  k = 0
  for (i, v) in enumerate(membership)
    existing = get(remap, v, 0)
    if existing == 0
      k += 1
      remap[v] = k
      idx[i] = k
    else
      idx[i] = existing
    end
  end
  return idx, k
end

"""
    _partition_modularity(g::MRIOGraph, membership::AbstractVector{<:Integer}, γ::Float64) -> Float64

Modularity of `membership` on `g`'s effective weights `A_ij = w(i, j)` under
the resolution `γ`, with `k_i = Σ_j A_ij` (diagonal counted once) and
`k_j^in = Σ_i A_ij`:

- undirected (`D == false`): with `m2 = Σ_i k_i`, `Q = (Σ_c S_c − γ Σ_c K_c²
  / m2) / m2` where `S_c = Σ_{i,j ∈ c} A_ij` is the double sum (a surviving
  diagonal `A_ii` inside `c` contributes once) and `K_c = Σ_{i∈c} k_i`;
- directed: with `m1 = Σ_i k_i^out`, `Q = (Σ_c e_c − γ Σ_c K_c^out K_c^in /
  m1) / m1` where `e_c = Σ_{i,j ∈ c} A_ij`, `K_c^out = Σ_{i∈c} k_i^out` and
  `K_c^in = Σ_{i∈c} k_i^in`;
- `m == 0` returns `0.0`.

This equals `Graphs.modularity(g, membership; distmx = Graphs.weights(g), γ =
γ)` for `self_loops == false` (both directednesses) and for directed graphs
even with self-loops. Known divergence: for undirected graphs with self-loops
Graphs.jl double-counts diagonal weight in its total `m` (and, depending on
the Graphs.jl version, in its community sums), so it disagrees with the
mathematically clean formula above there; `_partition_modularity` always
counts each diagonal entry once.

Two O(n²) streaming passes over the dense matrix (O(n) memory; the directed
path runs a degree pass then a community-internal pass, the undirected path
two tile passes — the undirected path reuses
[`symmetric_tile_accumulate!`](@ref); the directed path uses plain
column-major loops in the style of `_check_nonnegative_weights`), plus O(k)
community accumulators. The two directedness branches live in the helpers
[`_directed_modularity`](@ref) and [`_undirected_modularity`](@ref) so that
each accumulator is a plain local (a single function capturing the undirected
`internal` accumulator across the `if`/`else` branches would box it and
heap-allocate per update).
"""
function _partition_modularity(
  g::MRIOGraph{TV,TI,D},
  membership::AbstractVector{<:Integer},
  γ::Float64,
) where {TV,TI,D}
  idx, k = _compact_membership(membership, size(g.weights, 1))
  return D ? _directed_modularity(g.weights, idx, k, g.filter, γ) :
         _undirected_modularity(g.weights, idx, k, g.filter, γ)
end

"""
    _directed_modularity(W, idx, k, filter, γ) -> Float64

Directed (Leicht–Newman) modularity for the compact per-node community rows
`idx` (`1:k`) on the effective weights `filtered_weight(W, i, j, cutoff,
self_loops)`: with `m1 = Σ_i k_i^out`, `Q = (Σ_c e_c − γ Σ_c K_c^out K_c^in /
m1) / m1` where `e_c = Σ_{i,j ∈ c} A_ij`, `K_c^out = Σ_{i∈c} k_i^out` and
`K_c^in = Σ_{i∈c} k_i^in`; `m == 0` returns `0.0`. Two O(n²) streaming
passes (degrees, then community internals), O(n) memory.
"""
function _directed_modularity(W, idx::Vector{Int}, k::Int, filter, γ::Float64)
  n = size(W, 1)
  f = filter
  cutoff = weight_cutoff(f)
  self_loops = f[3]
  kout = zeros(Float64, n)
  kin = zeros(Float64, n)
  @inbounds for j in 1:n
    acc = 0.0
    for i in 1:n
      w = Float64(filtered_weight(W, i, j, cutoff, self_loops))
      kout[i] += w
      acc += w
    end
    kin[j] = acc
  end
  m = sum(kout)
  m == 0.0 && return 0.0
  internal = zeros(Float64, k)
  kout_c = zeros(Float64, k)
  kin_c = zeros(Float64, k)
  @inbounds for i in 1:n
    c = idx[i]
    kout_c[c] += kout[i]
    kin_c[c] += kin[i]
  end
  @inbounds for j in 1:n
    cj = idx[j]
    for i in 1:n
      if idx[i] == cj
        internal[cj] += Float64(filtered_weight(W, i, j, cutoff, self_loops))
      end
    end
  end
  penalty = 0.0
  @inbounds for c in 1:k
    penalty += kout_c[c] * kin_c[c]
  end
  return (sum(internal) - γ * penalty / m) / m
end

"""
    _undirected_modularity(W, idx, k, filter, γ) -> Float64

Undirected modularity for the compact per-node community rows `idx` (`1:k`)
on the symmetrized effective weights `sym_weight`: with `m2 = Σ_i k_i`,
`Q = (Σ_c S_c − γ Σ_c K_c² / m2) / m2` where `S_c = Σ_{i,j ∈ c} A_ij` is the
double sum (a surviving diagonal `A_ii` inside `c` contributes once) and
`K_c = Σ_{i∈c} k_i`; `m == 0` returns `0.0`. Two O(n²) streaming tile
passes (strengths, then community internals), O(n) memory.
"""
function _undirected_modularity(W, idx::Vector{Int}, k::Int, filter, γ::Float64)
  n = size(W, 1)
  f = filter
  strengths = zeros(Float64, n)
  symmetric_strengths!(strengths, W, f)
  m = sum(strengths)
  m == 0.0 && return 0.0
  internal = zeros(Float64, k)
  symmetric_tile_accumulate!(
    (i, j, w) -> begin
      ci = idx[i]
      if ci == idx[j]
        internal[ci] += i == j ? Float64(w) : 2.0 * Float64(w)
      end
    end,
    W,
    f,
  )
  totals = zeros(Float64, k)
  @inbounds for i in 1:n
    totals[idx[i]] += strengths[i]
  end
  penalty = 0.0
  @inbounds for c in 1:k
    penalty += totals[c] * totals[c]
  end
  return (sum(internal) - γ * penalty / m) / m
end

"""
    community_table(result::CommunityResult) -> DataFrame

Return the node → community mapping of `result` joined with the node metadata:
all columns of `result.nodes` (in order) followed by `community::Int32` (the
`membership` values). Returns a fresh `DataFrame` with copied columns, so
mutating the table cannot corrupt `result`; `nrow` equals
`nrow(result.nodes)`.
"""
function community_table(result::Juliora.CommunityResult)
  table = DataFrame(result.nodes; copycols=true)
  table[!, :community] = copy(result.membership)
  return table
end

"""
    _metadata_column(nodes::DataFrame, candidates) -> Union{String,Nothing}

First candidate present in `names(nodes)`, or `nothing`. Used for the
`CountryCode` → `Country` and `Sector` → `Industry` fallbacks, mirroring
`countries()`/`sectors()` in `src/mrio.jl` but returning the column name.
"""
function _metadata_column(nodes::DataFrame, candidates)
  node_names = names(nodes)
  for candidate in candidates
    candidate in node_names && return candidate
  end
  return nothing
end

"""
    _top_value(values, members) -> (top, share)

Most frequent value of `values` among the member node indices `members` (given
in node order), with ties broken towards the value appearing first in node
order. Returns the value as a `String` and its fraction of
`length(members)`.
"""
function _top_value(values::AbstractVector, members::Vector{Int})
  counts = Dict{String,Int}()
  first_seen = Dict{String,Int}()
  rank = 0
  for i in members
    s = string(values[i])
    if !haskey(counts, s)
      counts[s] = 0
      rank += 1
      first_seen[s] = rank
    end
    counts[s] += 1
  end
  best = ""
  best_count = -1
  best_rank = typemax(Int)
  for (s, c) in counts
    r = first_seen[s]
    if c > best_count || (c == best_count && r < best_rank)
      best = s
      best_count = c
      best_rank = r
    end
  end
  return best, length(members) == 0 ? 0.0 : best_count / length(members)
end

"""
    community_summary(result::CommunityResult) -> DataFrame

Summarize `result` with one row per community label (sorted ascending) and
columns `community::Int32`, `size::Int`, `internal_flow::Float64`,
`external_flow::Float64`, `internal_share::Float64`, plus the composition
columns `n_countries::Int`, `top_country::String`,
`top_country_share::Float64` (added only when `result.nodes` has a
`CountryCode` column, falling back to `Country`) and `n_sectors::Int`,
`top_sector::String`, `top_sector_share::Float64` (added only when
`result.nodes` has a `Sector` column, falling back to `Industry`).

With `A_ij` the graph's effective weights (directed: `filtered_weight`;
undirected: `sym_weight`) over the `1:n` nodes, `internal_flow(c) = Σ_{i,j ∈
c} A_ij` (double sum over ordered pairs, diagonal once — the same `S_c`/`e_c`
as [`_partition_modularity`](@ref)), `external_flow(c) = Σ_{i∈c,j∉c} A_ij +
Σ_{i∉c,j∈c} A_ij`, and `internal_share(c) = internal / (internal + external)`
(`0.0` when the denominator is zero). Composition counts the most frequent
metadata value among the community's members (ties → first in node order) and
its share of the community size; `n_countries`/`n_sectors` count the distinct
values.

One O(n²) streaming pass over the weights plus O(n) work over the nodes; O(n)
memory.
"""
function community_summary(result::Juliora.CommunityResult)
  membership = result.membership
  n = length(membership)
  nrow(result.nodes) == n || throw(
    DimensionMismatch(
      "membership has length $n but the node table has $(nrow(result.nodes)) rows",
    ),
  )
  labels = sort!(unique(membership))
  k = length(labels)
  row_of = Dict{Int32,Int}()
  for (r, label) in enumerate(labels)
    row_of[label] = r
  end
  idx = Vector{Int}(undef, n)
  @inbounds for i in 1:n
    idx[i] = row_of[membership[i]]
  end
  members = [Int[] for _ in 1:k]
  for i in 1:n
    push!(members[idx[i]], i)
  end
  internal = zeros(Float64, k)
  external = zeros(Float64, k)
  W = result.weights
  f = result.filter
  if result.directed
    cutoff = weight_cutoff(f)
    self_loops = f[3]
    @inbounds for j in 1:n
      for i in 1:n
        w = Float64(filtered_weight(W, i, j, cutoff, self_loops))
        if w != 0.0
          ci = idx[i]
          cj = idx[j]
          if ci == cj
            internal[ci] += w
          else
            external[ci] += w
            external[cj] += w
          end
        end
      end
    end
  else
    symmetric_tile_accumulate!(
      (i, j, w) -> begin
        fw = Float64(w)
        ci = idx[i]
        cj = idx[j]
        if ci == cj
          internal[ci] += i == j ? fw : 2.0 * fw
        else
          external[ci] += 2.0 * fw
          external[cj] += 2.0 * fw
        end
      end,
      W,
      f,
    )
  end
  share = Vector{Float64}(undef, k)
  @inbounds for c in 1:k
    denom = internal[c] + external[c]
    share[c] = denom == 0.0 ? 0.0 : internal[c] / denom
  end
  out = DataFrame(
    :community => labels,
    :size => length.(members),
    :internal_flow => internal,
    :external_flow => external,
    :internal_share => share,
  )
  country_col = _metadata_column(result.nodes, ("CountryCode", "Country"))
  if country_col !== nothing
    country_values = result.nodes[!, country_col]
    n_countries = Vector{Int}(undef, k)
    top_country = Vector{String}(undef, k)
    top_country_share = Vector{Float64}(undef, k)
    for c in 1:k
      distinct = Set{String}()
      for i in members[c]
        push!(distinct, string(country_values[i]))
      end
      top, frac = _top_value(country_values, members[c])
      n_countries[c] = length(distinct)
      top_country[c] = top
      top_country_share[c] = frac
    end
    out[!, :n_countries] = n_countries
    out[!, :top_country] = top_country
    out[!, :top_country_share] = top_country_share
  end
  sector_col = _metadata_column(result.nodes, ("Sector", "Industry"))
  if sector_col !== nothing
    sector_values = result.nodes[!, sector_col]
    n_sectors = Vector{Int}(undef, k)
    top_sector = Vector{String}(undef, k)
    top_sector_share = Vector{Float64}(undef, k)
    for c in 1:k
      distinct = Set{String}()
      for i in members[c]
        push!(distinct, string(sector_values[i]))
      end
      top, frac = _top_value(sector_values, members[c])
      n_sectors[c] = length(distinct)
      top_sector[c] = top
      top_sector_share[c] = frac
    end
    out[!, :n_sectors] = n_sectors
    out[!, :top_sector] = top_sector
    out[!, :top_sector_share] = top_sector_share
  end
  return out
end
