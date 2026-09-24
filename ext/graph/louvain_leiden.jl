# Multilevel Louvain over dense pair weights.
#
# Everything is defined on a pair-weight matrix `P` (square, diagonal counted
# once): node weights `k_i^out = Σ_j P[i, j]`, `k_i^in = Σ_j P[j, i]`,
# `m = Σ_ij P[i, j]`, and for a partition π `Q(π) = (Σ_c e_c − γ Σ_c
# K_c^out K_c^in / m) / m` with `e_c = Σ_{i,j ∈ c} P[i, j]` (ordered double
# sum; a diagonal `P[i, i]` inside `c` contributes once). Undirected graphs are
# the symmetric special case (`K^out = K^in`), so one implementation serves
# both directednesses. This is exactly the `_partition_modularity` definition.
#
# Level-0 pair weights are read on the fly from the wrapped matrix (never
# materialized): directed graphs use `filtered_weight`, undirected graphs use
# `sym_weight`. Aggregation `P₁[c, d] = Σ_{i ∈ c} Σ_{j ∈ d} P[i, j]` is the
# full ordered double sum; aggregate levels therefore read `P₁[c, d]` directly
# (diagonal kept) and NEVER through `sym_weight`, which would double-count.
#
# Memory rules: level 0 never allocates anything O(n²) — strengths,
# accumulators, and membership maps are O(n), and the pair weights stream from
# the wrapped matrix. Only the aggregates `P₁` (n₁×n₁ `Float64`, after compact
# labels) are dense; a pathological level-0 partition with `n₁ ≈ n` would make
# the aggregate O(n²), but real MRIO structure shrinks immediately (each level
# starts at the lifted previous partition and only strictly improves, so `Q`
# is monotone across levels).
#
# NOTE (final fixpoint post-processing, beyond the bare multilevel loop):
# aggregation locks the members of a supernode together, so the composed
# partition can admit a strictly improving flat single-node move; conversely
# the flat pass alone can disconnect a community by moving a bridge node.
# The driver therefore alternates the flat local-moving pass (ΔQ > 0 moves
# only, so `Q` never decreases) with the connectivity split
# (`_leiden_split_disconnected`, which never decreases `Q`) to their joint
# fixpoint (safety cap `_LEIDEN_FIXPOINT_MAX_ITERS`). Each active cycle
# strictly increases `Q`, so termination is finite and the returned partition
# is flat single-node optimal with every community connected — what the
# "Local optimality" and connectivity tests pin.

"""
    _LOUVAIN_MAX_SWEEPS

Hard safety cap on the local-moving sweeps per level (float-safety net;
normal convergence needs far fewer sweeps).
"""
const _LOUVAIN_MAX_SWEEPS = 500

"""
    _LouvainDirView{M}

Level-0 pair-weight view of a directed graph: `P[i, j] =
filtered_weight(W, i, j, cutoff, self_loops)`, read on the fly from the
wrapped matrix `W` (kept as-is; accumulated in `Float64`).
"""
struct _LouvainDirView{M}
    W::M
    cutoff::Float64
    self_loops::Bool
end

"""
    _LouvainSymView{M}

Level-0 pair-weight view of an undirected graph: `P[i, j] = sym_weight(W, i,
j, cutoff, self_loops)` (symmetric matrix semantics, diagonal counted once),
read on the fly from the wrapped matrix `W`.
"""
struct _LouvainSymView{M}
    W::M
    cutoff::Float64
    self_loops::Bool
end

"""
    _LouvainAggView{M}

Pair-weight view of a dense aggregate `P₁`: entries ARE the pair weights and
are read directly (`P[i, j]`, diagonal kept). `symmetric` records whether the
aggregate is symmetric (undirected lineage, so `a_in == a_out` is computed
once); it changes only the accumulation fast path, never the values.
"""
struct _LouvainAggView{M}
    P::M
    symmetric::Bool
end

"""
    _louvain_view(g::MRIOGraph) -> view

Level-0 pair-weight view of `g` (`_LouvainDirView` for directed graphs,
`_LouvainSymView` for undirected ones), sharing the wrapped matrix by
reference. Dispatch on the directedness value parameter is type-stable.
"""
_louvain_view(g::MRIOGraph{TV, TI, true}) where {TV, TI} =
    _LouvainDirView(g.weights, weight_cutoff(g.filter), g.filter[3])
_louvain_view(g::MRIOGraph{TV, TI, false}) where {TV, TI} =
    _LouvainSymView(g.weights, weight_cutoff(g.filter), g.filter[3])

"""
    _louvain_n(V) -> Int

Number of nodes of the pair-weight view `V`.
"""
_louvain_n(V::_LouvainDirView) = size(V.W, 1)
_louvain_n(V::_LouvainSymView) = size(V.W, 1)
_louvain_n(V::_LouvainAggView) = size(V.P, 1)

"""
    _louvain_symmetric(V) -> Bool

Whether the pair weights of `V` are symmetric (`a_in(i, c) == a_out(i, c)`
may then be computed once).
"""
_louvain_symmetric(::_LouvainDirView) = false
_louvain_symmetric(::_LouvainSymView) = true
_louvain_symmetric(V::_LouvainAggView) = V.symmetric

"""
    _louvain_pair(V, i, j) -> Float64

Pair weight `P[i, j]` of the view `V` as `Float64` (level 0: on-the-fly
filter/symmetrization of the wrapped matrix, diagonal counted once;
aggregates: the stored entry directly).
"""
@inline function _louvain_pair(V::_LouvainDirView, i::Int, j::Int)
    return Float64(filtered_weight(V.W, i, j, V.cutoff, V.self_loops))
end
@inline function _louvain_pair(V::_LouvainSymView, i::Int, j::Int)
    return Float64(sym_weight(V.W, i, j, V.cutoff, V.self_loops))
end
@inline function _louvain_pair(V::_LouvainAggView, i::Int, j::Int)
    @boundscheck checkbounds(V.P, i, j)
    @inbounds return V.P[i, j]
end

"""
    _louvain_strengths!(V, kout, kin) -> (kout, kin)

Fill `kout[i] = Σ_j P[i, j]` and `kin[j] = Σ_i P[i, j]` (diagonal counted
once) over the view `V` in one O(n²) streaming pass. `kout`/`kin` must have
length `_louvain_n(V)`; O(n) result, no O(n²) scratch.
"""
function _louvain_strengths!(V, kout::Vector{Float64}, kin::Vector{Float64})
    n = _louvain_n(V)
    fill!(kout, 0.0)
    fill!(kin, 0.0)
    for j in 1:n
        col = 0.0
        for i in 1:n
            w = _louvain_pair(V, i, j)
            kout[i] += w
            col += w
        end
        kin[j] = col
    end
    return kout, kin
end

"""
    _louvain_aggregate(V, cmap, n1) -> Matrix{Float64}

Dense aggregate `P₁[c, d] = Σ_{i ∈ c} Σ_{j ∈ d} P[i, j]` (full ordered double
sum) for the compact label map `cmap` (values `1:n1`). The undirected level-0
path accumulates via `symmetric_tile_accumulate!` (pair `i < j` adds `w` to
both `P₁[ci, cj]` and `P₁[cj, ci]` — `2w` when `ci == cj`, matching `P₁[c, c]
= e_c` — and the diagonal is added once); all other paths add each ordered
pair weight once. Labels are compacted before allocating, so `P₁` is `n1×n1`.
"""
function _louvain_aggregate(V, cmap::Vector{Int}, n1::Int)
    p1 = zeros(Float64, n1, n1)
    _louvain_aggregate_into!(p1, V, cmap)
    return p1
end

function _louvain_aggregate_into!(p1::Matrix{Float64}, V::_LouvainSymView, cmap::Vector{Int})
    # The combined cutoff is already folded into `V.cutoff`; repack it as a
    # filter tuple with `min_share = 0` so `weight_cutoff` reproduces it.
    f = (V.cutoff, 0.0, V.self_loops, 0.0)
    symmetric_tile_accumulate!(
        (i, j, w) -> begin
            ci = cmap[i]
            cj = cmap[j]
            fw = Float64(w)
            if i == j
                p1[ci, ci] += fw
            else
                p1[ci, cj] += fw
                p1[cj, ci] += fw
            end
        end,
        V.W,
        f,
    )
    return p1
end

function _louvain_aggregate_into!(p1::Matrix{Float64}, V::_LouvainDirView, cmap::Vector{Int})
    W = V.W
    n = size(W, 1)
    cutoff = V.cutoff
    self_loops = V.self_loops
    for j in 1:n
        cj = cmap[j]
        for i in 1:n
            x = filtered_weight(W, i, j, cutoff, self_loops)
            if x != zero(x)
                p1[cmap[i], cj] += Float64(x)
            end
        end
    end
    return p1
end

function _louvain_aggregate_into!(p1::Matrix{Float64}, V::_LouvainAggView, cmap::Vector{Int})
    P = V.P
    n = size(P, 1)
    for j in 1:n
        cj = cmap[j]
        for i in 1:n
            x = P[i, j]
            if x != 0.0
                p1[cmap[i], cj] += x
            end
        end
    end
    return p1
end

"""
    _louvain_gain(k_i_out, k_i_in, self_i, a_out_c, a_in_c, a_out_d, a_in_d, Kc_out, Kc_in, Kd_out, Kd_in, m, γ) -> Float64

Modularity gain of moving node `i` from community `c` to community `d`
(`d ≠ c`; a fresh singleton has `a_out_d = a_in_d = Kd_out = Kd_in = 0`):
with `Δe = (a_out_d + a_in_d + self_i) − (a_out_c + a_in_c − self_i)` (each
`a(i, c)` includes `P[i, i]` when `i ∈ c`) and the post-move community
strengths `K'`, `ΔQ = (Δe − γ (Kc_out' Kc_in' + Kd_out' Kd_in' − Kc_out Kc_in
− Kd_out Kd_in) / m) / m`. Pure function of the pinned `Q` definition.
"""
@inline function _louvain_gain(
        k_i_out::Float64,
        k_i_in::Float64,
        self_i::Float64,
        a_out_c::Float64,
        a_in_c::Float64,
        a_out_d::Float64,
        a_in_d::Float64,
        Kc_out::Float64,
        Kc_in::Float64,
        Kd_out::Float64,
        Kd_in::Float64,
        m::Float64,
        γ::Float64,
    )
    delta_e = (a_out_d + a_in_d + self_i) - (a_out_c + a_in_c - self_i)
    Kc_out_new = Kc_out - k_i_out
    Kc_in_new = Kc_in - k_i_in
    Kd_out_new = Kd_out + k_i_out
    Kd_in_new = Kd_in + k_i_in
    delta_k = (Kc_out_new * Kc_in_new + Kd_out_new * Kd_in_new - Kc_out * Kc_in - Kd_out * Kd_in)
    return (delta_e - γ * delta_k / m) / m
end

"""
    _louvain_compact(membership) -> (cmap, k)

Compact the arbitrary positive labels of `membership` to `1:k` in order of
first node appearance, returning the per-node map `cmap` and the community
count `k`.
"""
function _louvain_compact(membership::Vector{Int})
    remap = Dict{Int, Int}()
    cmap = Vector{Int}(undef, length(membership))
    k = 0
    for (i, c) in enumerate(membership)
        r = get(remap, c, 0)
        if r == 0
            k += 1
            remap[c] = k
            cmap[i] = k
        else
            cmap[i] = r
        end
    end
    return cmap, k
end

"""
    _louvain_local_move!(V, membership, kout, kin, m, γ, rng) -> membership

Local-moving phase on the pair-weight view `V` (length-`n` `membership`
input, mutated in place; may be singletons or any partition with positive
labels): repeat sweeps over all nodes in `rng`-shuffled
order until a sweep performs zero moves (hard cap `_LOUVAIN_MAX_SWEEPS`
sweeps). Per node, accumulate `a_out(i, c)`/`a_in(i, c)` per neighboring
community into reusable O(n) buffers (a symmetric view accumulates once and
uses it for both), consider every community with `a_out + a_in > 0` (in
ascending label order) plus a fresh singleton (evaluated last, so it only
wins on a strict improvement over every existing id), and move iff the best
`_louvain_gain` is strictly positive (`ΔQ > 0`; ties keep the smallest
community id). Strict positivity keeps isolated/zero-strength nodes in
singletons and makes termination finite. No per-node heap allocation.
"""
function _louvain_local_move!(
        V,
        membership::Vector{Int},
        kout::Vector{Float64},
        kin::Vector{Float64},
        m::Float64,
        γ::Float64,
        rng::Random.AbstractRNG,
    )
    n = length(membership)
    start_max = maximum(membership)
    cap = max(start_max, n) + 1
    kout_c = zeros(Float64, cap)
    kin_c = zeros(Float64, cap)
    cnt = zeros(Int, cap)
    for i in 1:n
        c = membership[i]
        kout_c[c] += kout[i]
        kin_c[c] += kin[i]
        cnt[c] += 1
    end
    nlabels = start_max
    free_ids = Vector{Int}()
    for c in 1:start_max
        if cnt[c] == 0
            push!(free_ids, c)
        end
    end
    acc_out = zeros(Float64, cap)
    acc_in = zeros(Float64, cap)
    seen = zeros(Int, cap)
    tag = 0
    touched = Vector{Int}(undef, cap)
    order = collect(1:n)
    sym = _louvain_symmetric(V)
    for _sweep in 1:_LOUVAIN_MAX_SWEEPS
        Random.shuffle!(rng, order)
        nmoves = 0
        for i in order
            ci = membership[i]
            tag += 1
            nt = 0
            if sym
                for j in 1:n
                    w = _louvain_pair(V, i, j)
                    if w != 0.0
                        c = membership[j]
                        if seen[c] != tag
                            seen[c] = tag
                            nt += 1
                            touched[nt] = c
                        end
                        acc_out[c] += w
                    end
                end
            else
                for j in 1:n
                    wo = _louvain_pair(V, i, j)
                    if wo != 0.0
                        c = membership[j]
                        if seen[c] != tag
                            seen[c] = tag
                            nt += 1
                            touched[nt] = c
                        end
                        acc_out[c] += wo
                    end
                    wi = _louvain_pair(V, j, i)
                    if wi != 0.0
                        c = membership[j]
                        if seen[c] != tag
                            seen[c] = tag
                            nt += 1
                            touched[nt] = c
                        end
                        acc_in[c] += wi
                    end
                end
            end
            self_i = _louvain_pair(V, i, i)
            kio = kout[i]
            kii = kin[i]
            Kc_o = kout_c[ci]
            Kc_i = kin_c[ci]
            a_c_o = acc_out[ci]
            a_c_i = sym ? a_c_o : acc_in[ci]
            best_d = 0
            best_gain = 0.0
            # Ascending label order: ties among improving candidates keep the
            # smallest community id without sorting.
            for d in 1:nlabels
                if d != ci && seen[d] == tag
                    a_d_o = acc_out[d]
                    a_d_i = sym ? a_d_o : acc_in[d]
                    gv = _louvain_gain(
                        kio,
                        kii,
                        self_i,
                        a_c_o,
                        a_c_i,
                        a_d_o,
                        a_d_i,
                        Kc_o,
                        Kc_i,
                        kout_c[d],
                        kin_c[d],
                        m,
                        γ,
                    )
                    if gv > best_gain
                        best_gain = gv
                        best_d = d
                    end
                end
            end
            # The fresh singleton (an empty label, or a brand-new one) sorts
            # after every existing id: it is evaluated last and only wins on
            # a strictly larger gain.
            fresh_is_best = false
            gv_fresh = _louvain_gain(
                kio,
                kii,
                self_i,
                a_c_o,
                a_c_i,
                0.0,
                0.0,
                Kc_o,
                Kc_i,
                0.0,
                0.0,
                m,
                γ,
            )
            if gv_fresh > best_gain
                best_gain = gv_fresh
                fresh_is_best = true
            end
            for t in 1:nt
                c = touched[t]
                acc_out[c] = 0.0
                if !sym
                    acc_in[c] = 0.0
                end
            end
            if fresh_is_best || best_d != 0
                if fresh_is_best
                    if isempty(free_ids)
                        nlabels += 1
                        if nlabels > cap
                            # Unreachable safety net (labels only grow by
                            # consuming/abandoning empty slots): grow every buffer.
                            newcap = 2 * cap + 1
                            resize!(kout_c, newcap)
                            resize!(kin_c, newcap)
                            resize!(cnt, newcap)
                            resize!(acc_out, newcap)
                            resize!(acc_in, newcap)
                            resize!(seen, newcap)
                            resize!(touched, newcap)
                            cap = newcap
                        end
                        best_d = nlabels
                    else
                        best_d = pop!(free_ids)
                    end
                end
                kout_c[ci] -= kio
                kin_c[ci] -= kii
                cnt[ci] -= 1
                if cnt[ci] == 0
                    push!(free_ids, ci)
                end
                kout_c[best_d] += kio
                kin_c[best_d] += kii
                cnt[best_d] += 1
                membership[i] = best_d
                nmoves += 1
            end
        end
        nmoves == 0 && break
    end
    return membership
end

"""
    _louvain_partition(g::MRIOGraph, γ::Float64, rng::Random.AbstractRNG) -> Vector{Int32}

Multilevel Louvain partition of `g` at resolution `γ`: local-move level 0
from singletons; if communities merged, compact the labels, push the map onto
a history stack, build the dense aggregate, and repeat from singletons on the
aggregate (each level starts at the lifted previous partition and only
strictly improves, so `Q` is monotone across levels). Node counts strictly
decrease per continued level, so the driver terminates. The composed
per-original-node membership then gets one final local-moving pass on the
level-0 view (aggregation locks supernode members together, so without this
pass the composed partition could admit a strictly improving flat
single-node move; the pass only applies `ΔQ > 0` moves, so `Q` never
decreases and the returned partition is flat single-node optimal).
Returns `Int32[1]` for a
single node and `1:n` singletons when the total pair weight `m` is zero.
"""
function _louvain_partition(g::MRIOGraph, γ::Float64, rng::Random.AbstractRNG)
    v0 = _louvain_view(g)
    n0 = _louvain_n(v0)
    n0 == 1 && return Int32[1]
    kout = zeros(Float64, n0)
    kin = zeros(Float64, n0)
    _louvain_strengths!(v0, kout, kin)
    m = sum(kout)
    m == 0.0 && return Int32.(1:n0)
    gamma = Float64(γ)
    membership = collect(1:n0)
    _louvain_local_move!(v0, membership, kout, kin, m, gamma, rng)
    cmap, n1 = _louvain_compact(membership)
    n1 == n0 && return Vector{Int32}(cmap)
    history = Vector{Vector{Int}}()
    push!(history, cmap)
    p1 = _louvain_aggregate(v0, cmap, n1)
    sym = _louvain_symmetric(v0)
    top = membership
    while true
        v = _LouvainAggView(p1, sym)
        n = size(p1, 1)
        ko = zeros(Float64, n)
        ki = zeros(Float64, n)
        _louvain_strengths!(v, ko, ki)
        mm = sum(ko)
        mem = collect(1:n)
        if mm != 0.0 # Unreachable float-safety net: aggregates of m > 0 keep mm > 0.
            _louvain_local_move!(v, mem, ko, ki, mm, gamma, rng)
        end
        cmap_next, n2 = _louvain_compact(mem)
        if n2 == n
            top = mem
            break
        end
        push!(history, cmap_next)
        p1 = _louvain_aggregate(v, cmap_next, n2)
    end
    cur = copy(top)
    for h in reverse(history)
        prev = Vector{Int}(undef, length(h))
        for i in eachindex(h)
            prev[i] = cur[h[i]]
        end
        cur = prev
    end
    _louvain_local_move!(v0, cur, kout, kin, m, gamma, rng)
    return Vector{Int32}(cur)
end

"""
    _LEIDEN_MAX_LEVELS

Hard safety cap on the Leiden multilevel loop (float-safety net; normal
convergence needs a handful of levels).
"""
const _LEIDEN_MAX_LEVELS = 500

"""
    _LEIDEN_FIXPOINT_MAX_ITERS

Safety cap on the final flat-move/split fixpoint loop of
[`_leiden_partition`](@ref) (float-safety net; normal convergence needs a
handful of cycles — each active cycle strictly increases `Q`).
"""
const _LEIDEN_FIXPOINT_MAX_ITERS = 50

# Leiden over the same pair-weight machinery (Traag,
# Waltman & van Eck 2019, "From Louvain to Leiden", arXiv:1810.08473).
#
# Assumes finite NON-NEGATIVE effective pair weights (the MRIO domain:
# filtered flows). Negatives would break the refinement scaling and the
# safety-net rationale below.
#
# Cross weight between disjoint node sets: `E(A, B) := Σ_{i∈A,j∈B} P[i,j] +
# Σ_{i∈B,j∈A} P[i,j]` (for symmetric `P` twice the pair-once weight).
# Well-connectedness: `wc(A, B) ⟺ E(A,B) ≥ γ (K_A^out K_B^in + K_A^in
# K_B^out)/m` (for symmetric `P`: `E(A,B) ≥ 2γ K_A K_B/m`, i.e. the paper's
# `E(A,B) ≥ γ‖A‖‖B‖` with volumes). "Adjacency" means `E(A,B) > 0`.
#
# Deviations from the paper: (1) the `MoveNodesFast` queue is replaced by
# rng-shuffled sweeps — on near-complete MRIO graphs every move changes
# nearly every node's neighbourhood, so the queue degenerates to full
# sweeps; (2) the returned partition gets the shared final flat local-move
# pass (fair comparison with `_louvain_partition`) followed by the
# connectivity safety net. The paper's outer iteration is NOT implemented —
# `communities(...; nruns)` runs independent iterations and keeps the best.

"""
    _leiden_well_connected(E, KAo, KAi, KBo, KBi, m, γ) -> Bool

Well-connectedness predicate `wc(A, B) ⟺ E(A,B) ≥ γ (K_A^out K_B^in +
K_A^in K_B^out)/m` for disjoint node sets with cross weight `E` and
community strengths `K`. Pure function; `m > 0` is required.
"""
@inline function _leiden_well_connected(
        E::Float64,
        KAo::Float64,
        KAi::Float64,
        KBo::Float64,
        KBi::Float64,
        m::Float64,
        γ::Float64,
    )
    return E >= γ * (KAo * KBi + KAi * KBo) / m
end

"""
    _leiden_same(a, b) -> Bool

Whether the length-`n` label vectors `a` and `b` encode the same partition
(co-membership agreement; labels may differ). Canonicalizes both through
`_louvain_compact` (first-appearance order) and compares elementwise.
"""
function _leiden_same(a::Vector{Int}, b::Vector{Int})
    length(a) == length(b) || return false
    ca, _ = _louvain_compact(a)
    cb, _ = _louvain_compact(b)
    return ca == cb
end

"""
    _leiden_lift(membership, cmap, n1) -> Vector{Int}

Lift the carried partition `membership` (on the current level's nodes)
through the refined map `cmap` (values `1:n1`) to the initial partition on
the aggregate: supernode `c` inherits the `membership` label of its refined
members (all share one label since the refined partition refines the
carried one; the first member wins). Labels stay arbitrary positive `Int`s.
"""
function _leiden_lift(membership::Vector{Int}, cmap::Vector{Int}, n1::Int)
    lifted = Vector{Int}(undef, n1)
    seen = falses(n1)
    for i in eachindex(cmap)
        c = cmap[i]
        if !seen[c]
            seen[c] = true
            lifted[c] = membership[i]
        end
    end
    return lifted
end

"""
    _leiden_refine(V, membership, kout, kin, m, γ, θ, rng) -> Vector{Int}

Refinement phase (`RefinePartition`/`MergeNodesSubset`): `refined` starts as
singletons (`refined[i] == i`). For each community `S` of `membership`
independently: `R = {v ∈ S : wc({v}, S∖{v})}` is visited in `rng`-random
order; when `v` is still a singleton, targets `𝒯 = {C of the refined
partition with C ⊆ S, C ≠ {v}, wc(C, S∖C), E({v}, C) > 0, ΔQ(v ↦ C) ≥ 0}`
(with `ΔQ` from `_louvain_gain` moving `v` out of its singleton into `C`)
are considered in ascending label order, and `C′ ∈ 𝒯` is picked with
probability `∝ exp(ΔQ/θ)` (numerically stable: the max `ΔQ` is subtracted
before exponentiating; a non-positive/non-finite `θ` falls back to the
smallest-id argmax). `v` merges into `C′`. Nodes not in `R` and leftover
singletons stay singleton refined communities. If float round-off ever leaves
the cumulative Boltzmann weight below the draw after the loop, `pick` keeps
the first (smallest-label) candidate — an implicit smallest-label fallback.

Guarantees: the output refines `membership`, and every refined community is
connected (it grows from a singleton by attaching nodes with `E({v},C) >
0`). `m > 0` is required. Per-community `O(|S|²)` pair reads, `O(n)` extra
bookkeeping plus per-community member lists.
"""
function _leiden_refine(
        V,
        membership::Vector{Int},
        kout::Vector{Float64},
        kin::Vector{Float64},
        m::Float64,
        γ::Float64,
        θ::Float64,
        rng::Random.AbstractRNG,
    )
    n = length(membership)
    refined = collect(1:n)
    if m == 0.0
        return refined
    end
    order = Dict{Int, Vector{Int}}()
    keys_order = Int[]
    for i in 1:n
        c = membership[i]
        if haskey(order, c)
            push!(order[c], i)
        else
            order[c] = [i]
            push!(keys_order, c)
        end
    end
    e_out_s = zeros(Float64, n)
    e_in_s = zeros(Float64, n)
    for key in keys_order
        S = order[key]
        ns = length(S)
        ns <= 1 && continue
        for u in S
            e_out_s[u] = 0.0
            e_in_s[u] = 0.0
        end
        kS_out = 0.0
        kS_in = 0.0
        for u in S
            kS_out += kout[u]
            kS_in += kin[u]
            for w in S
                e_out_s[u] += _louvain_pair(V, u, w)
                e_in_s[u] += _louvain_pair(V, w, u)
            end
        end
        R = Int[]
        for v in S
            self_v = _louvain_pair(V, v, v)
            Ev = (e_out_s[v] - self_v) + (e_in_s[v] - self_v)
            if _leiden_well_connected(
                    Ev,
                    kout[v],
                    kin[v],
                    kS_out - kout[v],
                    kS_in - kin[v],
                    m,
                    γ,
                )
                push!(R, v)
            end
        end
        isempty(R) && continue
        members = Dict{Int, Vector{Int}}()
        kc_out = Dict{Int, Float64}()
        kc_in = Dict{Int, Float64}()
        eC = Dict{Int, Float64}()
        sumE = Dict{Int, Float64}()
        for u in S
            members[u] = [u]
            kc_out[u] = kout[u]
            kc_in[u] = kin[u]
            eC[u] = _louvain_pair(V, u, u)
            sumE[u] = e_out_s[u] + e_in_s[u]
        end
        Random.shuffle!(rng, R)
        cand_labels = Int[]
        cand_gains = Float64[]
        for v in R
            # Still a singleton in the refined partition?
            if length(members[refined[v]]) != 1
                continue
            end
            self_v = _louvain_pair(V, v, v)
            kio = kout[v]
            kii = kin[v]
            empty!(cand_labels)
            empty!(cand_gains)
            for C in sort!(collect(keys(members)))
                C == v && continue
                Cmem = members[C]
                a_o = 0.0
                a_i = 0.0
                for w in Cmem
                    a_o += _louvain_pair(V, v, w)
                    a_i += _louvain_pair(V, w, v)
                end
                (a_o + a_i) > 0.0 || continue
                EC = sumE[C] - 2.0 * eC[C]
                _leiden_well_connected(
                    EC,
                    kc_out[C],
                    kc_in[C],
                    kS_out - kc_out[C],
                    kS_in - kc_in[C],
                    m,
                    γ,
                ) || continue
                gv = _louvain_gain(
                    kio,
                    kii,
                    self_v,
                    self_v,
                    self_v,
                    a_o,
                    a_i,
                    kio,
                    kii,
                    kc_out[C],
                    kc_in[C],
                    m,
                    γ,
                )
                gv >= 0.0 || continue
                push!(cand_labels, C)
                push!(cand_gains, gv)
            end
            isempty(cand_labels) && continue
            pick = cand_labels[1]
            if length(cand_labels) > 1
                if !(θ > 0.0) || !isfinite(θ)
                    best = cand_gains[1]
                    for t in 2:length(cand_labels)
                        if cand_gains[t] > best
                            best = cand_gains[t]
                            pick = cand_labels[t]
                        end
                    end
                else
                    gmax = maximum(cand_gains)
                    acc = 0.0
                    for t in eachindex(cand_gains)
                        acc += exp((cand_gains[t] - gmax) / θ)
                    end
                    draw = Random.rand(rng) * acc
                    run = 0.0
                    for t in eachindex(cand_labels)
                        run += exp((cand_gains[t] - gmax) / θ)
                        if run >= draw
                            pick = cand_labels[t]
                            break
                        end
                    end
                end
            end
            # Merge the singleton {v} into C = pick.
            a_o = 0.0
            a_i = 0.0
            for w in members[pick]
                a_o += _louvain_pair(V, v, w)
                a_i += _louvain_pair(V, w, v)
            end
            push!(members[pick], v)
            delete!(members, v)
            refined[v] = pick
            kc_out[pick] = kc_out[pick] + kio
            kc_in[pick] = kc_in[pick] + kii
            eC[pick] = eC[pick] + a_o + a_i + self_v
            sumE[pick] = sumE[pick] + e_out_s[v] + e_in_s[v]
            delete!(kc_out, v)
            delete!(kc_in, v)
            delete!(eC, v)
            delete!(sumE, v)
        end
    end
    return refined
end

"""
    _leiden_split_disconnected(V, membership) -> (Vector{Int}, Bool)

Connectivity safety net: on the level-0 view `V`, replace every community
of `membership` by its connected components (adjacency = effective pair
weight `> 0`, weak connectivity for directed views). Isolated nodes stay
singletons. This pass is sound because splitting a disconnected community
never decreases `Q`; it strictly improves when both split parts have
positive strength (the cross terms vanish and `Σ_c K_c^out·K_c^in ≤
K_F^out·K_F^in`) — the Q-improving formalization of "communities must be
connected". Returns the split membership and whether anything split.
`O(n²)` pair reads, `O(n)` memory.
"""
function _leiden_split_disconnected(V, membership::Vector{Int})
    n = length(membership)
    result = Vector{Int}(undef, n)
    visited = falses(n)
    order = Dict{Int, Vector{Int}}()
    keys_order = Int[]
    for i in 1:n
        c = membership[i]
        if haskey(order, c)
            push!(order[c], i)
        else
            order[c] = [i]
            push!(keys_order, c)
        end
    end
    sym = _louvain_symmetric(V)
    newlabel = 0
    stack = Int[]
    for key in keys_order
        C = order[key]
        for seed in C
            visited[seed] && continue
            newlabel += 1
            empty!(stack)
            push!(stack, seed)
            visited[seed] = true
            result[seed] = newlabel
            while !isempty(stack)
                u = pop!(stack)
                for v in C
                    if !visited[v] && v != u
                        adjacent = if sym
                            _louvain_pair(V, u, v) > 0.0
                        else
                            _louvain_pair(V, u, v) > 0.0 || _louvain_pair(V, v, u) > 0.0
                        end
                        if adjacent
                            visited[v] = true
                            result[v] = newlabel
                            push!(stack, v)
                        end
                    end
                end
            end
        end
    end
    did_split = !_leiden_same(result, membership)
    return result, did_split
end

"""
    _leiden_fixpoint_postprocess!(V, membership, kout, kin, m, γ, rng) -> membership

Shared Leiden output post-processing on the level-0 view `V`: alternate the
flat local-moving pass (`_louvain_local_move!`, `ΔQ > 0` moves only) with the
connectivity split (`_leiden_split_disconnected`) to their joint fixpoint
(safety cap `_LEIDEN_FIXPOINT_MAX_ITERS` iterations). The split of a
disconnected community never decreases `Q` (strict iff both parts have
positive strength) and no single move can re-merge two weak components (zero
cross-weight ⇒ fresh-singleton-dominated), while the flat pass alone can
disconnect a community by moving a bridge node; alternating to the fixpoint
therefore yields a partition that is BOTH flat single-node optimal AND has
every community connected. Each active cycle strictly increases `Q`, so
termination is finite and the cap is a float-safety net. Mutates
`membership` in place and returns it.
"""
function _leiden_fixpoint_postprocess!(V, membership::Vector{Int}, kout::Vector{Float64}, kin::Vector{Float64}, m::Float64, γ::Float64, rng::Random.AbstractRNG)
    for _ in 1:_LEIDEN_FIXPOINT_MAX_ITERS
        before = copy(membership)
        _louvain_local_move!(V, membership, kout, kin, m, γ, rng)
        moved = membership != before
        split_result, did_split = _leiden_split_disconnected(V, membership)
        if did_split
            copy!(membership, split_result)
        end
        moved == false && did_split == false && break
    end
    return membership
end

"""
    _leiden_partition(g::MRIOGraph, γ::Float64, rng::Random.AbstractRNG; θ=0.01) -> Vector{Int32}

Leiden partition of `g` at resolution `γ` (Traag, Waltman & van Eck 2019,
adapted to the pair-weight machinery; assumes finite non-negative
effective weights).

One Leiden iteration: local-moving (`ΔQ > 0` moves from the carried
partition) on the current level's view, then refinement of that partition
(sub-partition merges with `ΔQ ≥ 0` sampled `∝ exp(ΔQ/θ)`, `θ = 0.01` per
the paper's experiments). The aggregate is built on the REFINED partition
while the carried (non-refined) partition is lifted as the next level's
initial partition (quality-preserving: `Q` on the aggregate at the lifted
partition equals `Q` below). Levels repeat until the fixpoint (refinement
recovers the carried partition and nothing moved), the paper's singleton
`done` state, or no shrinkage (level cap `_LEIDEN_MAX_LEVELS`).

The final carried partition is flattened to level 0 and run through the
shared fixpoint post-processing (`_leiden_fixpoint_postprocess!`): the flat
local-moving pass alternated with the connectivity safety net
(`_leiden_split_disconnected`, which never decreases `Q`; strictly improves
when both split parts have positive strength) to their joint fixpoint, so
the result is flat single-node optimal and every community connected.
Returns arbitrary positive labels (`communities`
relabels); `Int32[1]` for a single node and `1:n` singletons when the
total pair weight `m` is zero.
"""
function _leiden_partition(g::MRIOGraph, γ::Float64, rng::Random.AbstractRNG; θ::Real = 0.01)
    v0 = _louvain_view(g)
    n0 = _louvain_n(v0)
    n0 == 1 && return Int32[1]
    kout0 = zeros(Float64, n0)
    kin0 = zeros(Float64, n0)
    _louvain_strengths!(v0, kout0, kin0)
    m0 = sum(kout0)
    m0 == 0.0 && return Int32.(1:n0)
    gamma = Float64(γ)
    theta = Float64(θ)
    sym0 = _louvain_symmetric(v0)
    history = Vector{Vector{Int}}()
    membership = collect(1:n0)
    before = copy(membership)
    _louvain_local_move!(v0, membership, kout0, kin0, m0, gamma, rng)
    moved0 = membership != before
    _, nc0 = _louvain_compact(membership)
    if nc0 == n0
        flat0 = copy(membership)
        _leiden_fixpoint_postprocess!(v0, flat0, kout0, kin0, m0, gamma, rng)
        return Vector{Int32}(flat0)
    end
    refined0 = _leiden_refine(v0, membership, kout0, kin0, m0, gamma, theta, rng)
    if _leiden_same(refined0, membership) && !moved0
        flat0 = copy(membership)
        _leiden_fixpoint_postprocess!(v0, flat0, kout0, kin0, m0, gamma, rng)
        return Vector{Int32}(flat0)
    end
    cmap0, n1 = _louvain_compact(refined0)
    if n1 == n0
        flat0 = copy(membership)
        _leiden_fixpoint_postprocess!(v0, flat0, kout0, kin0, m0, gamma, rng)
        return Vector{Int32}(flat0)
    end
    push!(history, cmap0)
    pcur = _louvain_aggregate(v0, cmap0, n1)
    carried = _leiden_lift(membership, cmap0, n1)
    for _level in 1:_LEIDEN_MAX_LEVELS
        ncur = size(pcur, 1)
        vcur = _LouvainAggView(pcur, sym0)
        kcur_out = zeros(Float64, ncur)
        kcur_in = zeros(Float64, ncur)
        _louvain_strengths!(vcur, kcur_out, kcur_in)
        mcur = sum(kcur_out)
        mcur == 0.0 && break # Unreachable float-safety net: strengths sum to m > 0 by construction.
        bef = copy(carried)
        _louvain_local_move!(vcur, carried, kcur_out, kcur_in, mcur, gamma, rng)
        moved = carried != bef
        _, nc = _louvain_compact(carried)
        nc == ncur && break
        refcur = _leiden_refine(vcur, carried, kcur_out, kcur_in, mcur, gamma, theta, rng)
        _leiden_same(refcur, carried) && !moved && break
        cmapcur, n2 = _louvain_compact(refcur)
        n2 == ncur && break
        push!(history, cmapcur)
        pcur = _louvain_aggregate(vcur, cmapcur, n2)
        carried = _leiden_lift(carried, cmapcur, n2)
    end
    flat = copy(carried)
    for h in reverse(history)
        prev = Vector{Int}(undef, length(h))
        for i in eachindex(h)
            prev[i] = flat[h[i]]
        end
        flat = prev
    end
    _leiden_fixpoint_postprocess!(v0, flat, kout0, kin0, m0, gamma, rng)
    return Vector{Int32}(flat)
end
