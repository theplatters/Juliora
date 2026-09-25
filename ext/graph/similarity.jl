# Structural node similarity and kNN similarity-graph construction.
#
# Memory rules: profile similarity never allocates an n×n result (R6). Each
# row block of the similarity matrix (block ≈ 64 rows) is accumulated in a
# reused block×n `Float64` scratch panel by panel (panel ≈ 64) via
# `mul!(S_block, X, transpose(Y))` (GEMM) over on-the-fly transformed
# effective weights — filtered `A` entries for cosine, `0/1` binarized
# filtered entries for jaccard — read through `_sim_raw`, so the
# `threshold`/`min_share` filter and the self-loop rule apply exactly as in
# every other graph kernel (R3: no materialized pruned copy). Scratch is
# O((block + n)·panel + block·n) `Float64` plus O(n) profile statistics;
# top-k per row uses a preallocated binary heap (no per-row allocation).
# `:random_walk` runs one personalized PageRank solve per source via
# `_weighted_pagerank`. Graph code never touches
# `LeontiefFactorization.data`, which would materialize a dense n×n inverse.

"""
    _SIMILARITY_METHODS

Methods accepted by [`node_similarity`](@ref)/[`similarity_graph`](@ref):
`:cosine` and `:jaccard` (profile similarity, all rows) and `:random_walk`
(PPR-style relatedness, selected source rows only).
"""
const _SIMILARITY_METHODS = (:cosine, :jaccard, :random_walk)

"""
    _SIMILARITY_ON

Profile selectors accepted by [`node_similarity`](@ref): `:out` (supplier
row profiles), `:in` (buyer column profiles), `:both` (concatenated
row/column profiles). Always validated; ignored by `:random_walk`.
"""
const _SIMILARITY_ON = (:out, :in, :both)

"""
    _SIMILARITY_SYMMETRIZE

Symmetrization rules accepted by [`similarity_graph`](@ref): `:max`
(maximum of the present directed values) and `:mean` (arithmetic mean over
the present directed values).
"""
const _SIMILARITY_SYMMETRIZE = (:max, :mean)

"""
    _SIMILARITY_BLOCK

Row-block height of the blocked similarity engine (see the file header).
"""
const _SIMILARITY_BLOCK = 64

"""
    _SIMILARITY_PANEL

Profile-dimension panel width of the blocked similarity engine (see the
file header).
"""
const _SIMILARITY_PANEL = 64

"""
    _validate_similarity_sources(method, sources, n) -> Union{Nothing,Vector{Int}}

Validate the `sources` keyword: it must be `nothing` unless
`method === :random_walk` (mirroring the `ncommunities`-only-with-`:spectral`
strictness of [`communities`](@ref)), and for `:random_walk` it must be a
non-empty vector of in-range node indices (Int-range `Integer`s, `Bool`
excluded, `1 ≤ s ≤ n`). Anything else throws an `ArgumentError`. Returns
`nothing`, or the deduplicated source list in first-occurrence order
(`[2, 1, 2]` dedupes to `[2, 1]`).
"""
function _validate_similarity_sources(method, sources, n)
    if method !== :random_walk
        sources === nothing || throw(
            ArgumentError(
                "sources is only used by method = :random_walk, got sources = $sources with method = $method",
            ),
        )
        return nothing
    end
    sources isa AbstractVector || throw(
        ArgumentError(
            "sources must be a non-empty vector of in-range node indices for method = :random_walk, got $sources",
        ),
    )
    isempty(sources) && throw(
        ArgumentError(
            "sources must be a non-empty vector of in-range node indices for method = :random_walk (got an empty vector)",
        ),
    )
    out = Int[]
    seen = Set{Int}()
    for s in sources
        (s isa Integer && !(s isa Bool) && typemin(Int) <= s <= typemax(Int) && 1 <= Int(s) <= Int(n)) ||
            throw(
            ArgumentError(
                "sources must hold node indices in 1:$n (Int-range Integers, Bool excluded), got $s",
            ),
        )
        si = Int(s)
        if si ∉ seen
            push!(seen, si)
            push!(out, si)
        end
    end
    return out
end

"""
    _validate_similarity_args(method, on, k, sources, damping, tol, max_iter, n) -> (Int, Union{Nothing,Vector{Int}})

Validate the shared `node_similarity`/`similarity_graph` keyword arguments,
throwing an `ArgumentError` with an informative message otherwise:
`method ∈ (:cosine, :jaccard, :random_walk)`; `on ∈ (:out, :in, :both)`
(always validated, even for `:random_walk` which ignores it); `k` an
Int-range `Integer` (`Bool` excluded) `≥ 1`; `sources` per
[`_validate_similarity_sources`](@ref); `damping` exactly as
`pagerank_scores` (`0 < damping < 1`); `tol > 0`; `max_iter` an Int-range
`Integer` (`Bool` excluded) `≥ 1`; the graph non-empty (`n ≥ 1`). Returns
`(Int(k), deduped sources)`.
"""
function _validate_similarity_args(method, on, k, sources, damping, tol, max_iter, n)
    method in _SIMILARITY_METHODS ||
        throw(ArgumentError("method must be one of $(_SIMILARITY_METHODS), got $method"))
    on in _SIMILARITY_ON || throw(ArgumentError("on must be one of $(_SIMILARITY_ON), got $on"))
    (k isa Integer && !(k isa Bool) && typemin(Int) <= k <= typemax(Int) && Int(k) >= 1) ||
        throw(ArgumentError("k must be an Int-range Integer (Bool excluded) ≥ 1, got $k"))
    srcs = _validate_similarity_sources(method, sources, n)
    0 < Float64(damping) < 1 ||
        throw(ArgumentError("damping must satisfy 0 < damping < 1, got $damping"))
    Float64(tol) > 0 || throw(ArgumentError("tol must be > 0, got $tol"))
    (max_iter isa Integer && !(max_iter isa Bool) && typemin(Int) <= max_iter <= typemax(Int) && Int(max_iter) >= 1) ||
        throw(ArgumentError("max_iter must be an Int-range Integer (Bool excluded) ≥ 1, got $max_iter"))
    Int(n) >= 1 || throw(ArgumentError("node_similarity requires a non-empty graph (nv(g) = $n)"))
    return Int(k), srcs
end

"""
    _sim_raw(W, i, j, cutoff, self_loops, ::Val{D})

Effective profile weight `A[i, j]`: directed `filtered_weight` for
`D == true`, undirected `sym_weight` for `D == false`. The `Val(D)`
dispatch keeps the directedness branch compile-time inside the panel fill
loops.
"""
@inline _sim_raw(W, i, j, cutoff, self_loops, ::Val{true}) =
    filtered_weight(W, i, j, cutoff, self_loops)
@inline _sim_raw(W, i, j, cutoff, self_loops, ::Val{false}) =
    sym_weight(W, i, j, cutoff, self_loops)

"""
    _profile_stats(g::MRIOGraph) -> (rowsq, colsq, rownnz, colnnz)

Stream the effective weights `A` once (O(n²) time, O(n) memory) and return
the per-node profile statistics: `rowsq[i] = Σ_p A[i, p]²` and
`colsq[j] = Σ_p A[p, j]²` (cosine squared norms), `rownnz[i]`/`colnnz[j]`
the corresponding binarized support sizes (jaccard). Self-loop and filter
semantics are exactly the graph's (via [`_sim_raw`](@ref)).
"""
function _profile_stats(g::MRIOGraph{TV, TI, D}) where {TV, TI, D}
    W = g.weights
    n = size(W, 1)
    cutoff = weight_cutoff(g.filter)
    self_loops = g.filter[3]
    dval = Val(D)
    rowsq = zeros(Float64, n)
    colsq = zeros(Float64, n)
    rownnz = zeros(Int, n)
    colnnz = zeros(Int, n)
    @inbounds for j in 1:n
        for i in 1:n
            x = Float64(_sim_raw(W, i, j, cutoff, self_loops, dval))
            if x != 0.0
                rowsq[i] += x * x
                colsq[j] += x * x
                rownnz[i] += 1
                colnnz[j] += 1
            end
        end
    end
    return rowsq, colsq, rownnz, colnnz
end

"""
    _accumulate_profile_dots!(Sv, Xfull, Yfull, g, rows, colpart, binarize) -> Sv

Accumulate one profile part (row profiles for `colpart == false`, column
profiles for `colpart == true`) of the dot products for block `rows` into
`Sv` (length(`rows`) × n): stream the profile dimension in panels of
`size(Yfull, 2)`, materialize the on-the-fly transformed values (filtered
`A` entries, or `0/1` binarized filtered entries for `binarize == true`)
for the block×panel slice into `Xfull` and the all-rows×panel slice into
`Yfull`, then `mul!(Sv, Xv, transpose(Yv), 1.0, 1.0)` (GEMM). For `:both`
the caller invokes this twice (row part, then column part) into the same
`Sv`, so dots and supports of the concatenated profiles add without ever
materializing them. `Xfull`/`Yfull` are caller-owned reusable scratch.
"""
function _accumulate_profile_dots!(
        Sv::AbstractMatrix{Float64},
        Xfull::Matrix{Float64},
        Yfull::Matrix{Float64},
        g::MRIOGraph{TV, TI, D},
        rows::UnitRange{Int},
        colpart::Bool,
        binarize::Bool,
    ) where {TV, TI, D}
    W = g.weights
    n = size(W, 1)
    nb = length(rows)
    npan = size(Yfull, 2)
    dval = Val(D)
    cutoff = weight_cutoff(g.filter)
    self_loops = g.filter[3]
    p = 1
    while p <= n
        q = min(p + npan - 1, n)
        np = q - p + 1
        Xv = view(Xfull, 1:nb, 1:np)
        Yv = view(Yfull, 1:n, 1:np)
        if !colpart
            @inbounds for t in 1:np
                pc = p + t - 1
                for a in 1:nb
                    x = _sim_raw(W, rows[a], pc, cutoff, self_loops, dval)
                    Xv[a, t] = binarize ? (x != zero(x) ? 1.0 : 0.0) : Float64(x)
                end
                for b in 1:n
                    x = _sim_raw(W, b, pc, cutoff, self_loops, dval)
                    Yv[b, t] = binarize ? (x != zero(x) ? 1.0 : 0.0) : Float64(x)
                end
            end
        else
            @inbounds for t in 1:np
                pc = p + t - 1
                for a in 1:nb
                    x = _sim_raw(W, pc, rows[a], cutoff, self_loops, dval)
                    Xv[a, t] = binarize ? (x != zero(x) ? 1.0 : 0.0) : Float64(x)
                end
                for b in 1:n
                    x = _sim_raw(W, pc, b, cutoff, self_loops, dval)
                    Yv[b, t] = binarize ? (x != zero(x) ? 1.0 : 0.0) : Float64(x)
                end
            end
        end
        LinearAlgebra.mul!(Sv, Xv, transpose(Yv), 1.0, 1.0)
        p = q + 1
    end
    return Sv
end

"""
    _sim_worse(v1, i1, v2, i2) -> Bool

Top-k heap order: candidate `(v1, i1)` is worse than `(v2, i2)` iff its
value is smaller, or the values tie and its index is larger (so ties keep
the smaller index, the pinned top-k tie rule).
"""
@inline _sim_worse(v1::Float64, i1::Int, v2::Float64, i2::Int) =
    v1 < v2 || (v1 == v2 && i1 > i2)

"""
    _sim_heap_topk!(hval, hidx, scores, n, k, skip) -> Int

Collect the top-k entries of `scores[1:n]` (strictly positive values only,
index `skip` excluded — the self row) into the preallocated `hval`/`hidx`
buffers (capacity `k ≥ 1`) via a binary min-heap ordered by
[`_sim_worse`](@ref) (O(n log k) time, zero allocation), then heapsort the
kept entries in place into rank order (descending value, ties → smaller
index). Returns the kept count (`≤ k`; rows with no positive candidates
keep nothing). Ranks `scores` scanned in ascending index order, so equal
values resolve to the smaller index.
"""
function _sim_heap_topk!(
        hval::Vector{Float64},
        hidx::Vector{Int},
        scores::AbstractVector{Float64},
        n::Int,
        k::Int,
        skip::Int,
    )
    hlen = 0
    @inbounds for j in 1:n
        if j == skip
            continue
        end
        s = scores[j]
        if !(s > 0.0)
            continue
        end
        if hlen < k
            hlen += 1
            pos = hlen
            while pos > 1
                par = pos ÷ 2
                if _sim_worse(s, j, hval[par], hidx[par])
                    hval[pos] = hval[par]
                    hidx[pos] = hidx[par]
                    pos = par
                else
                    break
                end
            end
            hval[pos] = s
            hidx[pos] = j
        elseif s > hval[1] || (s == hval[1] && j < hidx[1])
            pos = 1
            while true
                left = 2 * pos
                left > hlen && break
                child = left
                if left < hlen && _sim_worse(hval[left + 1], hidx[left + 1], hval[left], hidx[left])
                    child = left + 1
                end
                if _sim_worse(hval[child], hidx[child], s, j)
                    hval[pos] = hval[child]
                    hidx[pos] = hidx[child]
                    pos = child
                else
                    break
                end
            end
            hval[pos] = s
            hidx[pos] = j
        end
    end
    @inbounds for t in hlen:-1:2
        wv = hval[1]
        wi = hidx[1]
        lv = hval[t]
        li = hidx[t]
        pos = 1
        lim = t - 1
        while true
            left = 2 * pos
            left > lim && break
            child = left
            if left < lim && _sim_worse(hval[left + 1], hidx[left + 1], hval[left], hidx[left])
                child = left + 1
            end
            if _sim_worse(hval[child], hidx[child], lv, li)
                hval[pos] = hval[child]
                hidx[pos] = hidx[child]
                pos = child
            else
                break
            end
        end
        hval[pos] = lv
        hidx[pos] = li
        hval[t] = wv
        hidx[t] = wi
    end
    return hlen
end

"""
    _similarity_knn_profiles(g::MRIOGraph, method::Symbol, on::Symbol, k::Int) -> SparseMatrixCSC{Float32,Int32}

Blocked top-k profile similarity (`method ∈ (:cosine, :jaccard)`, `k ≥ 1`
already validated): per row block, accumulate the profile dots into reused
`Float64` scratch (row part and, for `on === :both`, column part into the
same block via [`_accumulate_profile_dots!`](@ref)), normalize each row
through the [`_profile_stats`](@ref) statistics (cosine: dot over the norm
product, `0` when either norm is `0`; jaccard: intersections over
`size_i + size_j - inter`, `0` when the union is empty — for `:both` the
statistics are the summed row/column parts), and keep the top-k per row
with [`_sim_heap_topk!`](@ref) (self excluded, strictly positive values
only, ties → smaller index). Returns the compact upper-unbounded
`SparseMatrixCSC{Float32, Int32}` holding `Float32(S[i, j])` per kept
directed edge `i → j` (at most `k` per row).
"""
function _similarity_knn_profiles(g::MRIOGraph{TV, TI, D}, method::Symbol, on::Symbol, k::Int) where {TV, TI, D}
    n = Graphs.nv(g)
    keff = min(k, n - 1)
    I = Int32[]
    J = Int32[]
    V = Float32[]
    if keff < 1
        return SparseArrays.sparse(I, J, V, n, n)
    end
    sizehint!(I, keff * n)
    sizehint!(J, keff * n)
    sizehint!(V, keff * n)
    rowsq, colsq, rownnz, colnnz = _profile_stats(g)
    stat = Vector{Float64}(undef, n)
    if method === :cosine
        if on === :out
            @inbounds for i in 1:n
                stat[i] = sqrt(rowsq[i])
            end
        elseif on === :in
            @inbounds for i in 1:n
                stat[i] = sqrt(colsq[i])
            end
        else
            @inbounds for i in 1:n
                stat[i] = sqrt(rowsq[i] + colsq[i])
            end
        end
    else
        if on === :out
            @inbounds for i in 1:n
                stat[i] = Float64(rownnz[i])
            end
        elseif on === :in
            @inbounds for i in 1:n
                stat[i] = Float64(colnnz[i])
            end
        else
            @inbounds for i in 1:n
                stat[i] = Float64(rownnz[i] + colnnz[i])
            end
        end
    end
    Sfull = Matrix{Float64}(undef, _SIMILARITY_BLOCK, n)
    Xfull = Matrix{Float64}(undef, _SIMILARITY_BLOCK, _SIMILARITY_PANEL)
    Yfull = Matrix{Float64}(undef, n, _SIMILARITY_PANEL)
    rowbuf = Vector{Float64}(undef, n)
    hval = Vector{Float64}(undef, keff)
    hidx = Vector{Int}(undef, keff)
    r1 = 1
    while r1 <= n
        r2 = min(r1 + _SIMILARITY_BLOCK - 1, n)
        rows = r1:r2
        nb = r2 - r1 + 1
        Sv = view(Sfull, 1:nb, :)
        fill!(Sv, 0.0)
        if on === :out || on === :both
            _accumulate_profile_dots!(Sv, Xfull, Yfull, g, rows, false, method === :jaccard)
        end
        if on === :in || on === :both
            _accumulate_profile_dots!(Sv, Xfull, Yfull, g, rows, true, method === :jaccard)
        end
        @inbounds for a in 1:nb
            i = rows[a]
            if method === :cosine
                ni = stat[i]
                if ni == 0.0
                    fill!(rowbuf, 0.0)
                else
                    for j in 1:n
                        d = Sv[a, j]
                        nj = stat[j]
                        rowbuf[j] = (d == 0.0 || nj == 0.0) ? 0.0 : d / (ni * nj)
                    end
                end
            else
                si = stat[i]
                for j in 1:n
                    inter = Sv[a, j]
                    u = si + stat[j] - inter
                    rowbuf[j] = (inter == 0.0 || u == 0.0) ? 0.0 : inter / u
                end
            end
            hlen = _sim_heap_topk!(hval, hidx, rowbuf, n, keff, i)
            for t in 1:hlen
                push!(I, Int32(i))
                push!(J, Int32(hidx[t]))
                push!(V, Float32(hval[t]))
            end
        end
        r1 = r2 + 1
    end
    return SparseArrays.sparse(I, J, V, n, n)
end

"""
    _similarity_knn_random_walk(g::MRIOGraph, k::Int, srcs::Vector{Int}, damping, tol, max_iter) -> SparseMatrixCSC{Float32,Int32}

PPR-style top-k relatedness (`method === :random_walk`, `k ≥ 1` already
validated): for each source `s` in `srcs` (deduplicated), run personalized
PageRank with teleport `δ_s` on the effective-weight transitions of `g` as
built (directed rows; for undirected `g` the symmetric `A` rows) with
dangling mass redistributed to the personalization vector, via
[`_weighted_pagerank`](@ref) — one solve per source. The score vector `p_s`
(visit probabilities, summing to 1) is the similarity of every node to `s`;
keep its top-k excluding `s` itself ([`_sim_heap_topk!`](@ref):
strictly positive values only, ties → smaller index, `k` clamped to
`n - 1`). Only source rows carry edges. Returns the compact
`SparseMatrixCSC{Float32, Int32}` holding `Float32(p_s[j])` per kept
directed edge `s → j`.
"""
function _similarity_knn_random_walk(
        g::MRIOGraph{TV, TI, D},
        k::Int,
        srcs::Vector{Int},
        damping::Real,
        tol::Real,
        max_iter::Integer,
    ) where {TV, TI, D}
    n = Graphs.nv(g)
    keff = min(k, n - 1)
    I = Int32[]
    J = Int32[]
    V = Float32[]
    if keff < 1
        return SparseArrays.sparse(I, J, V, n, n)
    end
    sizehint!(I, keff * length(srcs))
    sizehint!(J, keff * length(srcs))
    sizehint!(V, keff * length(srcs))
    tele = zeros(Float64, n)
    hval = Vector{Float64}(undef, keff)
    hidx = Vector{Int}(undef, keff)
    for s in srcs
        fill!(tele, 0.0)
        tele[s] = 1.0
        p = _weighted_pagerank(g, damping, tele, tol, max_iter; fname = "node_similarity (:random_walk)")
        hlen = _sim_heap_topk!(hval, hidx, p, n, keff, s)
        @inbounds for t in 1:hlen
            push!(I, Int32(s))
            push!(J, Int32(hidx[t]))
            push!(V, Float32(hval[t]))
        end
    end
    return SparseArrays.sparse(I, J, V, n, n)
end

"""
    _symmetrize_knn(S::SparseMatrixCSC{Float32}, n::Int, symmetrize::Symbol) -> SparseMatrixCSC{Float32,Int32}

Symmetrize a directed kNN weight matrix `S` (n×n) into the undirected pair
set: the union of kept directed edges as unordered pairs `{i, j}`
(`i ≠ j`; stored self-pairs, which top-k never emits, are dropped). The
pair weight is the maximum (`:max`) or the arithmetic mean (`:mean`) over
the present directed values among `S[i, j]` and `S[j, i]` — a one-sided
pair keeps its value under both rules. Returns the one-sided
upper-triangular `SparseMatrixCSC{Float32, Int32}` with `weights[i, j] = s`
for `i < j` and `0` elsewhere, so the pair weight reads back as
`W[i, j] + W[j, i] = s` (the established undirected fixture convention).
Runs in O(nnz log nnz) time over the kept edges only.
"""
function _symmetrize_knn(S::SparseArrays.SparseMatrixCSC{Float32}, n::Int, symmetrize::Symbol)
    rows, cols, vals = SparseArrays.findnz(S)
    m = length(vals)
    lo = Vector{Int}(undef, m)
    hi = Vector{Int}(undef, m)
    vv = Vector{Float32}(undef, m)
    cnt = 0
    @inbounds for t in 1:m
        i = Int(rows[t])
        j = Int(cols[t])
        if i == j
            continue
        end
        cnt += 1
        if i < j
            lo[cnt] = i
            hi[cnt] = j
        else
            lo[cnt] = j
            hi[cnt] = i
        end
        vv[cnt] = vals[t]
    end
    resize!(lo, cnt)
    resize!(hi, cnt)
    resize!(vv, cnt)
    order = sortperm(collect(zip(lo, hi)))
    I = Int32[]
    J = Int32[]
    V = Float32[]
    sizehint!(I, cnt)
    sizehint!(J, cnt)
    sizehint!(V, cnt)
    t = 1
    while t <= cnt
        u = t
        best = vv[order[t]]
        acc = Float64(vv[order[t]])
        while u + 1 <= cnt && lo[order[u + 1]] == lo[order[t]] && hi[order[u + 1]] == hi[order[t]]
            u += 1
            v = vv[order[u]]
            best = max(best, v)
            acc += Float64(v)
        end
        w = symmetrize === :max ? best : Float32(acc / (u - t + 1))
        push!(I, Int32(lo[order[t]]))
        push!(J, Int32(hi[order[t]]))
        push!(V, w)
        t = u + 1
    end
    return SparseArrays.sparse(I, J, V, n, n)
end

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
  transitions of `g` as built (directed rows; for undirected `g` the
  symmetric `A` rows) with teleport to `δ_s` and dangling-node mass
  redistributed to the personalization vector (standard
  Andersen–Chung–Lang PPR; for uniform personalization this reduces exactly
  to the `pagerank_scores` iteration). The score vector `p_s` (visit
  probabilities, summing to 1) is the similarity of every node to `s`.
  Documented cost: one solve per source.

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
"""
function node_similarity(
        g::MRIOGraph{TV, TI, D};
        method::Symbol = :cosine,
        on::Symbol = :out,
        k::Integer = 10,
        sources = nothing,
        damping::Real = 0.85,
        tol::Real = 1.0e-6,
        max_iter::Integer = 100,
    ) where {TV, TI, D}
    n = Graphs.nv(g)
    kint, srcs = _validate_similarity_args(method, on, k, sources, damping, tol, max_iter, n)
    _check_nonnegative_weights(g, "node_similarity")
    S = if method === :random_walk
        _similarity_knn_random_walk(g, kint, srcs, damping, tol, max_iter)
    else
        _similarity_knn_profiles(g, method, on, kint)
    end
    return mrio_graph(S, g.nodes; direction = :directed)
end

"""
    node_similarity(mrio::MRIO; source=nothing, weights=nothing, direction=:directed, threshold=0.0, min_share=0.0, self_loops=false, method=:cosine, on=:out, k=10, sources=nothing, damping=0.85, tol=1.0e-6, max_iter=100) -> MRIOGraph

Build `mrio_graph(mrio; source, weights, direction, threshold, min_share,
self_loops)` and forward to `node_similarity(g; method, on, k, sources,
damping, tol, max_iter)`. The wrapped transactions/technical matrix is used
by reference; the Leontief factorization is never touched. Returns a
directed compact kNN `MRIOGraph` over the selected matrix's supplier-side
`row_indices` (shared reference). `direction` defaults to `:directed`
(unlike `communities`): profiles are built from the directed
supply/purchase flows. See the `MRIOGraph` method for the methods,
profiles, top-k rule, and keyword contract.
"""
function node_similarity(
        mrio::Juliora.MRIO;
        source::Union{Nothing, Symbol} = nothing,
        weights::Union{Nothing, Symbol} = nothing,
        direction::Symbol = :directed,
        threshold::Real = 0.0,
        min_share::Real = 0.0,
        self_loops::Bool = false,
        method::Symbol = :cosine,
        on::Symbol = :out,
        k::Integer = 10,
        sources = nothing,
        damping::Real = 0.85,
        tol::Real = 1.0e-6,
        max_iter::Integer = 100,
    )
    g = mrio_graph(
        mrio;
        source = source,
        weights = weights,
        direction = direction,
        threshold = threshold,
        min_share = min_share,
        self_loops = self_loops,
    )
    return node_similarity(
        g;
        method = method,
        on = on,
        k = k,
        sources = sources,
        damping = damping,
        tol = tol,
        max_iter = max_iter,
    )
end

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
"""
function similarity_graph(
        g::MRIOGraph{TV, TI, D};
        method::Symbol = :cosine,
        on::Symbol = :out,
        k::Integer = 10,
        sources = nothing,
        damping::Real = 0.85,
        tol::Real = 1.0e-6,
        max_iter::Integer = 100,
        symmetrize::Symbol = :max,
    ) where {TV, TI, D}
    symmetrize in _SIMILARITY_SYMMETRIZE ||
        throw(ArgumentError("symmetrize must be one of $(_SIMILARITY_SYMMETRIZE), got $symmetrize"))
    _check_nonnegative_weights(g, "similarity_graph")
    knn = node_similarity(
        g;
        method = method,
        on = on,
        k = k,
        sources = sources,
        damping = damping,
        tol = tol,
        max_iter = max_iter,
    )
    n = Graphs.nv(g)
    S = _symmetrize_knn(knn.weights, n, symmetrize)
    return mrio_graph(S, g.nodes; direction = :undirected)
end

"""
    similarity_graph(mrio::MRIO; source=nothing, weights=nothing, direction=:directed, threshold=0.0, min_share=0.0, self_loops=false, method=:cosine, on=:out, k=10, sources=nothing, damping=0.85, tol=1.0e-6, max_iter=100, symmetrize=:max) -> MRIOGraph

Build `mrio_graph(mrio; source, weights, direction, threshold, min_share,
self_loops)` and forward to `similarity_graph(g; method, on, k, sources,
damping, tol, max_iter, symmetrize)`. The wrapped transactions/technical
matrix is used by reference; the Leontief factorization is never touched.
Returns an undirected compact kNN `MRIOGraph` over the selected matrix's
supplier-side `row_indices` (shared reference) with one-sided
upper-triangular storage (read pair weights via
`Graphs.weights(g)[i, j]`). See the `MRIOGraph` method and
[`node_similarity`](@ref) for the methods, profiles, top-k rule, and
keyword contract.
"""
function similarity_graph(
        mrio::Juliora.MRIO;
        source::Union{Nothing, Symbol} = nothing,
        weights::Union{Nothing, Symbol} = nothing,
        direction::Symbol = :directed,
        threshold::Real = 0.0,
        min_share::Real = 0.0,
        self_loops::Bool = false,
        method::Symbol = :cosine,
        on::Symbol = :out,
        k::Integer = 10,
        sources = nothing,
        damping::Real = 0.85,
        tol::Real = 1.0e-6,
        max_iter::Integer = 100,
        symmetrize::Symbol = :max,
    )
    g = mrio_graph(
        mrio;
        source = source,
        weights = weights,
        direction = direction,
        threshold = threshold,
        min_share = min_share,
        self_loops = self_loops,
    )
    return similarity_graph(
        g;
        method = method,
        on = on,
        k = k,
        sources = sources,
        damping = damping,
        tol = tol,
        max_iter = max_iter,
        symmetrize = symmetrize,
    )
end
