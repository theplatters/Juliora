# Cross-network and cross-partition comparison (`compare_networks`,
# `compare_partitions`).
#
# Memory rules: every metric is computed on the matched node/item set, which
# is accessed through index maps and — for the PageRank option — a lazy
# `_MatchedView` over `Graphs.weights(g)`. No m×m submatrix is ever
# materialized: elementwise passes over the two dense matrices give O(n²)
# time with O(1) extra memory (plus O(m) index/strength/score vectors).
# Graph code never touches `LeontiefFactorization.data`, which would
# materialize a dense n×n inverse.

"""
    _compare_key_columns(nodes1::DataFrame, nodes2::DataFrame, match) -> Vector{Symbol}

Resolve the `match` argument to the key columns used for node alignment:

- `match === :keys` (default): every column present in both node tables, in
  the first table's column order. Zero shared columns throw an
  `ArgumentError`.
- `match::Vector{Symbol}`: exactly those columns (non-empty; each must exist
  in both tables, else `ArgumentError`).

Anything else throws an `ArgumentError`.
"""
function _compare_key_columns(nodes1::DataFrame, nodes2::DataFrame, match)
    names1 = propertynames(nodes1)
    in2 = Set(propertynames(nodes2))
    if match isa Symbol
        match === :keys || throw(
            ArgumentError("match must be :keys or a non-empty Vector{Symbol} of column names, got $match"),
        )
        shared = Symbol[c for c in names1 if c in in2]
        isempty(shared) && throw(
            ArgumentError(
                "match = :keys found no columns shared by both node tables " *
                    "(first table: $names1, second table: $(propertynames(nodes2)))",
            ),
        )
        return shared
    elseif match isa Vector{Symbol}
        isempty(match) && throw(
            ArgumentError("match must be :keys or a non-empty Vector{Symbol} of column names, got an empty vector"),
        )
        in1 = Set(names1)
        for c in match
            (c in in1 && c in in2) ||
                throw(ArgumentError("match column :$c must exist in both node tables"))
        end
        return copy(match)
    else
        throw(
            ArgumentError("match must be :keys or a non-empty Vector{Symbol} of column names, got $match"),
        )
    end
end

"""
    _compare_key(nodes::DataFrame, r::Int, cols::Vector{Symbol}) -> Tuple

Composite key of node row `r` over `cols`. Compared with `isequal`
(missing-safe) by the index maps.
"""
function _compare_key(nodes::DataFrame, r::Int, cols::Vector{Symbol})
    return ntuple(i -> nodes[r, cols[i]], length(cols))
end

"""
    _compare_match_maps(nodes1::DataFrame, nodes2::DataFrame, cols::Vector{Symbol}) -> (map1, map2)

Align the two node tables over the key `cols`: `map1[t]`/`map2[t]` are the
corresponding node rows of the `t`-th matched item. The matched items are
the key intersection ordered by the first table's row order. Keys must
uniquely identify nodes in each table (a duplicate key throws an
`ArgumentError` naming the offending key). Key tuples compare with
`isequal`, so `missing` entries match `missing`.
"""
function _compare_match_maps(nodes1::DataFrame, nodes2::DataFrame, cols::Vector{Symbol})
    index2 = Dict{Any, Int}()
    for r in 1:nrow(nodes2)
        k = _compare_key(nodes2, r, cols)
        haskey(index2, k) && throw(
            ArgumentError(
                "match columns $cols do not uniquely identify nodes in the second input (duplicate key $k)",
            ),
        )
        index2[k] = r
    end
    seen1 = Set{Any}()
    map1 = Int[]
    map2 = Int[]
    for r in 1:nrow(nodes1)
        k = _compare_key(nodes1, r, cols)
        k in seen1 && throw(
            ArgumentError(
                "match columns $cols do not uniquely identify nodes in the first input (duplicate key $k)",
            ),
        )
        push!(seen1, k)
        if haskey(index2, k)
            push!(map1, r)
            push!(map2, index2[k])
        end
    end
    return map1, map2
end

"""
    _tied_ranks(x::AbstractVector{<:Real}) -> Vector{Float64}

Average ranks of `x` for Spearman correlation: 1-based, ties share the mean
of their positions (e.g. `[1.0, 2.0, 2.0, 4.0]` gives
`[1.0, 2.5, 2.5, 4.0]`). Sorting gives O(m log m) time with one length-m
result vector.
"""
function _tied_ranks(x::AbstractVector{<:Real})
    n = length(x)
    order = sortperm(x)
    ranks = Vector{Float64}(undef, n)
    t = 1
    while t <= n
        u = t
        while u < n && x[order[u + 1]] == x[order[t]]
            u += 1
        end
        avg = (t + u) / 2.0
        for v in t:u
            ranks[order[v]] = avg
        end
        t = u + 1
    end
    return ranks
end

"""
    _safe_cor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) -> Float64

`Statistics.cor(x, y)` with degenerate inputs mapped to `NaN` instead of
throwing: lengths below 2, or either side constant (zero standard
deviation), give `NaN`.
"""
function _safe_cor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    (n == length(y) && n >= 2) || return NaN
    sx = Statistics.std(x)
    sy = Statistics.std(y)
    (isfinite(sx) && isfinite(sy) && sx > 0.0 && sy > 0.0) || return NaN
    return Statistics.cor(x, y)
end

"""
    _MatchedView{TF} <: AbstractMatrix{Float64}

Lazy matched-subset view over a parent pair-weight matrix (in practice the
`Graphs.weights(g)` view of a graph): `size(V) == (m, m)` with
`V[i, j] == parent[map[i], map[j]]`. Reads the matched pairs on the fly —
no submatrix is ever materialized.
"""
struct _MatchedView{TF <: AbstractMatrix{Float64}} <: AbstractMatrix{Float64}
    parent::TF
    map::Vector{Int}
end

Base.size(V::_MatchedView) = (length(V.map), length(V.map))
Base.IndexStyle(::Type{<:_MatchedView}) = IndexCartesian()
Base.eltype(::Type{<:_MatchedView}) = Float64

@inline function Base.getindex(V::_MatchedView, i::Int, j::Int)
    @boundscheck checkbounds(V, i, j)
    @inbounds return Float64(V.parent[V.map[i], V.map[j]])
end

Base.getindex(V::_MatchedView, i::Integer, j::Integer) = V[Int(i), Int(j)]

"""
    _matched_graph(g::MRIOGraph, map::Vector{Int}) -> MRIOGraph

Matched subgraph of `g` over the node rows `map`: a directed `MRIOGraph`
over the lazy [`_MatchedView`](@ref) of `Graphs.weights(g)` with the
identity filter `(0.0, 0.0, true, 0.0)`. The cutoff `0.0` keeps every value
and `self_loops = true` lets the diagonal pass through, so the pair
weights read back exactly as in the source graph
(`Graphs.weights(gmatched)[i, j] == Graphs.weights(g)[map[i], map[j]]`;
the source graph's own self-loop choices are already applied). The node
table is shared by reference. Nothing is copied.
"""
function _matched_graph(g::MRIOGraph{TV, TI, D}, map::Vector{Int}) where {TV, TI, D}
    V = _MatchedView(Graphs.weights(g), map)
    return MRIOGraph{Float64, TI, true, typeof(V)}(V, g.nodes, (0.0, 0.0, true, 0.0))
end

"""
    compare_networks(g1::MRIOGraph, g2::MRIOGraph; match=:keys, pagerank=false, damping=0.85) -> DataFrame

Compare two MRIO networks on their shared (matched) nodes. Graphs only:
filter choices are per-graph, so build graphs first with `mrio_graph`
(there is no MRIO pass-through).

All metrics are computed on the matched node set — the key intersection
ordered by `g1`'s node order (see below); unmatched nodes contribute to
nothing, not even the densities or scales.

# Node matching (`match`)

- `match === :keys` (default): the composite key is every column present in
  both `nodes` tables, in `g1.nodes` column order (zero shared columns throw
  an `ArgumentError`).
- `match::Vector{Symbol}`: exactly those columns (non-empty; each must exist
  in both tables, else `ArgumentError`).

Anything else throws an `ArgumentError`. Keys must uniquely identify nodes
in each graph (a duplicate key throws an `ArgumentError` naming the
offending key); key tuples compare with `isequal` (missing-safe).

# Metrics

With `V_g[i, j] = Graphs.weights(g)[map_g[i], map_g[j]]` the effective pair
weight of graph `g` on the matched set (read on the fly — never
materialized) and `m` the matched count, all sums run over the ordered
matched pairs `(i, j)` (diagonal included; each graph's self-loop semantics
are already baked into its pair weights):

- `edge_overlap = Σ min(V_1[i, j], V_2[i, j]) / Σ max(V_1[i, j], V_2[i, j])`
  (Julia `/` semantics: `NaN` when `Σ max == 0`).
- Matched-set strengths `sout_g[i] = Σ_j V_g[i, j]` and
  `sin_g[i] = Σ_j V_g[j, i]`; `pearson_out`/`pearson_in` are
  `Statistics.cor` of the two graphs' out/in strengths, `spearman_out`/
  `spearman_in` are Pearson correlations of their average tied ranks
  ([`_tied_ranks`](@ref)). Degenerate statistics (`m < 2`, zero variance)
  give `NaN`, never an exception.
- `density_g = |{(i, j): V_g[i, j] != 0}| / m^2` (nonzero ordered pairs over
  `m^2`) and `density_ratio = density_1 / density_2`.
- `scale_g = Σ V_g[i, j]` (total effective pair weight) and
  `scale_ratio = scale_1 / scale_2` (Julia `/`: `NaN`/`Inf` propagate).

# PageRank option (`pagerank`)

`pagerank = true` additionally solves PageRank on each matched subgraph
(uniform teleport, `damping`, via `_weighted_pagerank` with the
`pagerank_scores` defaults `tol = 1.0e-6`, `max_iter = 100`; one solve per
graph) and reports `pearson_pagerank`/`spearman_pagerank` over the two score
vectors. The matched subgraphs are lazy [`_MatchedView`](@ref)s (nothing
copied). When `m == 0` the solves are skipped and both columns hold `NaN`.
Non-convergence throws an `ErrorException` as in `pagerank_scores`.

Comparing graphs built with different `direction` mixes pair-weight
conventions (an undirected graph's pair weight enters both ordered slots),
so build both graphs with the same `direction` for interpretable ratios.

# Keyword arguments

- `match`: `:keys` or a non-empty `Vector{Symbol}` (default `:keys`).
- `pagerank`: `Bool` (default `false`; any non-`Bool` throws an
  `ArgumentError`).
- `damping`: teleportation damping factor, `0 < damping < 1`
  (`ArgumentError` otherwise; validated exactly as in `pagerank_scores`,
  always — even when `pagerank = false`).

Effective weights of both graphs must be finite and non-negative (an
`ArgumentError` naming the first offending pair `(i, j)` is thrown
otherwise, via `_check_nonnegative_weights(g, "compare_networks")` on each
graph; values pruned by a graph's filter cannot violate this).

# Returns

A one-row `DataFrame` with columns (in order) `n_matched::Int`,
`n_1::Int`, `n_2::Int` (matched count and each graph's full node count
`nv(g)`), `edge_overlap::Float64`, `pearson_out::Float64`,
`spearman_out::Float64`, `pearson_in::Float64`, `spearman_in::Float64`,
`density_1::Float64`, `density_2::Float64`, `density_ratio::Float64`,
`scale_1::Float64`, `scale_2::Float64`, `scale_ratio::Float64`, plus
`pearson_pagerank::Float64`, `spearman_pagerank::Float64` appended when
`pagerank = true` (absent otherwise).

# Memory

Elementwise passes over the two dense matrices: O(n²) time, O(1) extra
memory — the matched node set is accessed through index maps and a lazy
view, no submatrix is ever materialized (plus O(m) index/strength vectors,
and O(m) iteration vectors per PageRank solve when `pagerank = true`).

!!! warning
    Graph code must never touch `LeontiefFactorization.data`: that accessor
    materializes a dense n×n matrix inverse. Comparison (and all other graph
    routines) operate on the wrapped transactions/technical matrices only;
    graph-only workflows can use `mrio_graph(W, nodes)` to skip the Leontief
    factorization entirely.
"""
function compare_networks(g1::MRIOGraph, g2::MRIOGraph; match = :keys, pagerank = false, damping::Real = 0.85)
    pagerank isa Bool ||
        throw(ArgumentError("pagerank must be a Bool, got $pagerank"))
    0 < Float64(damping) < 1 ||
        throw(ArgumentError("damping must satisfy 0 < damping < 1, got $damping"))
    cols = _compare_key_columns(g1.nodes, g2.nodes, match)
    _check_nonnegative_weights(g1, "compare_networks")
    _check_nonnegative_weights(g2, "compare_networks")
    map1, map2 = _compare_match_maps(g1.nodes, g2.nodes, cols)
    m = length(map1)
    n1 = Graphs.nv(g1)
    n2 = Graphs.nv(g2)
    F1 = Graphs.weights(g1)
    F2 = Graphs.weights(g2)
    sout1 = zeros(Float64, m)
    sout2 = zeros(Float64, m)
    sin1 = zeros(Float64, m)
    sin2 = zeros(Float64, m)
    sum_min = 0.0
    sum_max = 0.0
    cnt1 = 0
    cnt2 = 0
    scale1 = 0.0
    scale2 = 0.0
    @inbounds for j in 1:m
        a1 = map1[j]
        a2 = map2[j]
        for i in 1:m
            v1 = F1[map1[i], a1]
            v2 = F2[map2[i], a2]
            sum_min += min(v1, v2)
            sum_max += max(v1, v2)
            sout1[i] += v1
            sout2[i] += v2
            sin1[j] += v1
            sin2[j] += v2
            v1 != 0.0 && (cnt1 += 1)
            v2 != 0.0 && (cnt2 += 1)
            scale1 += v1
            scale2 += v2
        end
    end
    edge_overlap = sum_min / sum_max
    pearson_out = _safe_cor(sout1, sout2)
    spearman_out = _safe_cor(_tied_ranks(sout1), _tied_ranks(sout2))
    pearson_in = _safe_cor(sin1, sin2)
    spearman_in = _safe_cor(_tied_ranks(sin1), _tied_ranks(sin2))
    msq = Float64(m)^2
    density1 = Float64(cnt1) / msq
    density2 = Float64(cnt2) / msq
    density_ratio = density1 / density2
    scale_ratio = scale1 / scale2
    base = (
        n_matched = m,
        n_1 = n1,
        n_2 = n2,
        edge_overlap = edge_overlap,
        pearson_out = pearson_out,
        spearman_out = spearman_out,
        pearson_in = pearson_in,
        spearman_in = spearman_in,
        density_1 = density1,
        density_2 = density2,
        density_ratio = density_ratio,
        scale_1 = scale1,
        scale_2 = scale2,
        scale_ratio = scale_ratio,
    )
    if pagerank
        if m == 0
            pearson_pr = NaN
            spearman_pr = NaN
        else
            gm1 = _matched_graph(g1, map1)
            gm2 = _matched_graph(g2, map2)
            teleport = fill(1.0 / m, m)
            p1 = _weighted_pagerank(gm1, damping, teleport, 1.0e-6, 100; fname = "compare_networks")
            p2 = _weighted_pagerank(gm2, damping, teleport, 1.0e-6, 100; fname = "compare_networks")
            pearson_pr = _safe_cor(p1, p2)
            spearman_pr = _safe_cor(_tied_ranks(p1), _tied_ranks(p2))
        end
        return DataFrame(; base..., pearson_pagerank = pearson_pr, spearman_pagerank = spearman_pr)
    else
        return DataFrame(; base...)
    end
end

"""
    _contingency_counts(a, b) -> (cells, rows, cols)

Contingency table of the aligned label vectors `a`/`b` (equal length) by
`Dict` counting in one pass (O(m) memory): `cells[(u, v)]` is `n_uv`,
`rows[u]`/`cols[v]` the margins. Labels may be arbitrary integers.
"""
function _contingency_counts(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    cells = Dict{Tuple{Any, Any}, Int}()
    rows = Dict{Any, Int}()
    cols = Dict{Any, Int}()
    for t in eachindex(a, b)
        u = a[t]
        v = b[t]
        key = (u, v)
        cells[key] = get(cells, key, 0) + 1
        rows[u] = get(rows, u, 0) + 1
        cols[v] = get(cols, v, 0) + 1
    end
    return cells, rows, cols
end

@inline _choose2(x::Real) = Float64(x) * (Float64(x) - 1.0) / 2.0

"""
    _ari_nmi(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) -> (ari, nmi)

Adjusted Rand index (Hubert–Arabie) and normalized mutual information
(arithmetic-mean normalization) of the aligned label vectors `a`/`b`
(equal length `m`), from one contingency table ([`_contingency_counts`](@ref)).

With `n_uv` the cells, `a_u`/`b_v` the margins, `C(x, 2) = x(x - 1)/2`
(computed exactly in `Float64`) and `P = C(m, 2)`:

- `ARI = (Σ C(n_uv, 2) − ΣC(a_u)·ΣC(b_v)/P) / (0.5·(ΣC(a_u) + ΣC(b_v)) − ΣC(a_u)·ΣC(b_v)/P)`;
  a zero denominator (both partitions all-singletons or both one-cluster)
  gives `1.0`.
- `NMI = 2·I / (H_1 + H_2)` with natural logs,
  `I = Σ p_uv·log(p_uv / (p_u·q_v))` over nonzero cells
  (`0·log(0) := 0`) and `H_1`, `H_2` the margin entropies;
  `H_1 + H_2 == 0` (both partitions trivial) gives `1.0`.

Fewer than 2 items give `(1.0, 1.0)` (vacuous agreement: no pair can
disagree).
"""
function _ari_nmi(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    m = length(a)
    if m < 2
        return 1.0, 1.0
    end
    cells, rows, cols = _contingency_counts(a, b)
    fm = Float64(m)
    sum_cells = 0.0
    for n in values(cells)
        sum_cells += _choose2(n)
    end
    sum_rows = 0.0
    for r in values(rows)
        sum_rows += _choose2(r)
    end
    sum_cols = 0.0
    for c in values(cols)
        sum_cols += _choose2(c)
    end
    P = _choose2(m)
    expected = sum_rows * sum_cols / P
    denom = 0.5 * (sum_rows + sum_cols) - expected
    ari = denom == 0.0 ? 1.0 : (sum_cells - expected) / denom
    I = 0.0
    for ((u, v), n) in cells
        I += (Float64(n) / fm) * log((Float64(n) * fm) / (Float64(rows[u]) * Float64(cols[v])))
    end
    H1 = 0.0
    for r in values(rows)
        H1 -= (Float64(r) / fm) * log(Float64(r) / fm)
    end
    H2 = 0.0
    for c in values(cols)
        H2 -= (Float64(c) / fm) * log(Float64(c) / fm)
    end
    nmi = (H1 + H2) == 0.0 ? 1.0 : 2.0 * I / (H1 + H2)
    return ari, nmi
end

"""
    _validate_partition_labels(v::AbstractVector, which::AbstractString)

Require `eltype(v) <: Integer` with `Bool` excluded (the established
strictness pattern: `Bool` is an `Integer` subtype but never a valid
community label), throwing an `ArgumentError` otherwise.
"""
function _validate_partition_labels(v::AbstractVector, which::AbstractString)
    (eltype(v) <: Integer && !(eltype(v) <: Bool)) || throw(
        ArgumentError("$which partition labels must be Integers (Bool excluded), got eltype $(eltype(v))"),
    )
    return nothing
end

"""
    _check_community_lengths(c::CommunityResult, which::AbstractString)

Require `length(c.membership) == nrow(c.nodes)` (position `i` of
`membership` is the community of node row `i`), throwing an
`ArgumentError` otherwise.
"""
function _check_community_lengths(c::Juliora.CommunityResult, which::AbstractString)
    length(c.membership) == nrow(c.nodes) || throw(
        ArgumentError(
            "$which CommunityResult has $(length(c.membership)) membership labels but $(nrow(c.nodes)) node rows",
        ),
    )
    return nothing
end

"""
    _partition_row(n_matched, n_1, n_2, ari, nmi, labels1, labels2, modularity_1, modularity_2) -> DataFrame

One-row result frame shared by the `compare_partitions` methods (columns in
order `n_matched`, `n_1`, `n_2`, `ari`, `nmi`, `n_communities_1`,
`n_communities_2`, `modularity_1`, `modularity_2`).
"""
function _partition_row(
        n_matched::Int,
        n_1::Int,
        n_2::Int,
        ari::Float64,
        nmi::Float64,
        labels1,
        labels2,
        modularity_1::Float64,
        modularity_2::Float64,
    )
    return DataFrame(
        n_matched = n_matched,
        n_1 = n_1,
        n_2 = n_2,
        ari = ari,
        nmi = nmi,
        n_communities_1 = length(unique(labels1)),
        n_communities_2 = length(unique(labels2)),
        modularity_1 = modularity_1,
        modularity_2 = modularity_2,
    )
end

"""
    compare_partitions(c1::CommunityResult, c2::CommunityResult; match=:keys) -> DataFrame
    compare_partitions(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) -> DataFrame
    compare_partitions(c::CommunityResult, v::AbstractVector) -> DataFrame
    compare_partitions(v::AbstractVector, c::CommunityResult) -> DataFrame

Compare two community partitions: adjusted Rand index (ARI), normalized
mutual information (NMI), community counts, and modularities. Membership
labels may be arbitrary integers (negative or large labels are accepted).

# Alignment

- `CommunityResult` × `CommunityResult`: items are aligned via the node key
  columns exactly as in `compare_networks` — `match === :keys` (default)
  uses every column present in both `nodes` tables (zero shared columns
  throw an `ArgumentError`); `match::Vector{Symbol}` names the columns
  explicitly (non-empty, each present in both tables); anything else throws
  an `ArgumentError`. Keys must uniquely identify nodes in each result
  (duplicate keys throw an `ArgumentError` naming the key); the matched
  items are the key intersection ordered by the first input's node order.
  Community counts are over the matched items; `n_1`/`n_2` are the full
  `nrow(c.nodes)` counts.
- Vector × vector: positional over the two label vectors (equal lengths
  required, else `ArgumentError`; no `match` keyword). Label vectors must
  hold Integers with `Bool` excluded (`ArgumentError` otherwise, as for any
  non-`Integer` eltype).
- Mixed `CommunityResult` × vector forms: positional with the same vector
  and length rules; the `modularity` column holds `NaN` on the vector side.

# Metrics (over the aligned items, `m = n_matched`)

ARI (Hubert–Arabie) and NMI (arithmetic-mean normalization, natural logs)
via one contingency table ([`_ari_nmi`](@ref)). A zero ARI denominator
(both partitions all-singletons or both one-cluster) gives `1.0`, as does
`H_1 + H_2 == 0` for NMI (both partitions trivial); fewer than 2 aligned
items give `(1.0, 1.0)`.

# Returns

A one-row `DataFrame` with columns (in order) `n_matched::Int`,
`n_1::Int`, `n_2::Int`, `ari::Float64`, `nmi::Float64`,
`n_communities_1::Int`, `n_communities_2::Int` (distinct labels among the
matched items), `modularity_1::Float64`, `modularity_2::Float64` (the stored
`CommunityResult` modularities; `NaN` for raw-vector sides).

# Memory

Contingency-table counting in one pass: O(m) memory. The matched items are
accessed through index maps — no submatrix is ever materialized.

!!! warning
    Graph code must never touch `LeontiefFactorization.data`: that accessor
    materializes a dense n×n matrix inverse. Comparison (and all other graph
    routines) operate on the wrapped transactions/technical matrices only;
    graph-only workflows can use `mrio_graph(W, nodes)` to skip the Leontief
    factorization entirely.
"""
function compare_partitions(c1::Juliora.CommunityResult, c2::Juliora.CommunityResult; match = :keys)
    cols = _compare_key_columns(c1.nodes, c2.nodes, match)
    _check_community_lengths(c1, "first")
    _check_community_lengths(c2, "second")
    map1, map2 = _compare_match_maps(c1.nodes, c2.nodes, cols)
    a = c1.membership[map1]
    b = c2.membership[map2]
    ari, nmi = _ari_nmi(a, b)
    return _partition_row(
        length(map1),
        nrow(c1.nodes),
        nrow(c2.nodes),
        ari,
        nmi,
        a,
        b,
        Float64(c1.modularity),
        Float64(c2.modularity),
    )
end

function compare_partitions(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    _validate_partition_labels(a, "first")
    _validate_partition_labels(b, "second")
    length(a) == length(b) || throw(
        ArgumentError(
            "partitions must have equal length for positional comparison, got $(length(a)) and $(length(b))",
        ),
    )
    ari, nmi = _ari_nmi(a, b)
    return _partition_row(length(a), length(a), length(b), ari, nmi, a, b, NaN, NaN)
end

function compare_partitions(a::AbstractVector, b::AbstractVector)
    throw(
        ArgumentError(
            "partitions must hold Integer labels (Bool excluded), got eltypes $(eltype(a)) and $(eltype(b))",
        ),
    )
end

function compare_partitions(c::Juliora.CommunityResult, v::AbstractVector)
    _check_community_lengths(c, "first")
    _validate_partition_labels(v, "second")
    length(c.membership) == length(v) || throw(
        ArgumentError(
            "partitions must have equal length for positional comparison, got $(length(c.membership)) and $(length(v))",
        ),
    )
    ari, nmi = _ari_nmi(c.membership, v)
    return _partition_row(
        length(v),
        nrow(c.nodes),
        length(v),
        ari,
        nmi,
        c.membership,
        v,
        Float64(c.modularity),
        NaN,
    )
end

function compare_partitions(v::AbstractVector, c::Juliora.CommunityResult)
    _validate_partition_labels(v, "first")
    _check_community_lengths(c, "second")
    length(v) == length(c.membership) || throw(
        ArgumentError(
            "partitions must have equal length for positional comparison, got $(length(v)) and $(length(c.membership))",
        ),
    )
    ari, nmi = _ari_nmi(v, c.membership)
    return _partition_row(
        length(v),
        length(v),
        nrow(c.nodes),
        ari,
        nmi,
        v,
        c.membership,
        NaN,
        Float64(c.modularity),
    )
end
