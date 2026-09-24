# (Flow-weighted) PageRank over `MRIOGraph`.
#
# Memory rules: the weighted iteration works on the effective weights `w(i, j)`
# read on the fly from the wrapped dense matrix — no pruned or symmetrized
# matrix is ever materialized. The iteration allocates exactly three length-n
# `Float64` vectors (`r`, `work`, `inv_s`) and nothing per iteration. Graph
# code never touches `LeontiefFactorization.data`, which would materialize a
# dense n×n inverse.

"""
    pagerank_scores(g::MRIOGraph; damping=0.85, weighted=true, tol=1.0e-6, max_iter=100) -> SeriesEntry
    pagerank_scores(mrio::MRIO; source=nothing, weights=nothing, direction=:directed, threshold=0.0, min_share=0.0, self_loops=false, damping=0.85, weighted=true, tol=1.0e-6, max_iter=100) -> SeriesEntry

Compute (flow-weighted) PageRank scores over the MRIO graph.

# Iteration (weighted path, `weighted=true`)

With `n = nv(g)`, effective weights `w(i, j)` of `g` (directed:
`filtered_weight`; undirected: `sym_weight`, i.e. undirected graphs run
PageRank on their symmetrized weights), out-strengths `s_i = Σ_j w(i, j)`
(`symmetric_strengths!` for undirected graphs), `inv_s[i] = 1 / s_i` (or `0.0`
when `s_i == 0`), and starting vector `r = fill(1 / n, n)`:

```julia
for iter in 1:max_iter
    dangling = Σ_{i: inv_s[i] == 0} r[i]
    work[j] = Σ_i w(i, j) * r[i] * inv_s[i]          # filtered_tvec! / symmetric_tvec!
    β = (1 - damping + damping * dangling) / n
    err = Σ_j |damping * work[j] + β - r[j]|
    r[j] = damping * work[j] + β
    err < n * tol && return SeriesEntry(r, g.nodes)
end
```

This is the weighted generalization of `Graphs.pagerank`'s iteration
(identical fixed point for unit weights, including the dangling convention:
dangling nodes — zero out-strength — redistribute their mass uniformly via
`β`). The convergence criterion `err < n * tol` matches Graphs'
`err < N * ϵ`, so `tol` means the same on both code paths. The scores always
sum to 1 (`Σ scores == 1`) and are non-negative.

# Unweighted path (`weighted=false`)

Delegates to `Graphs.pagerank`: `simple, _ = to_simple_graph(g)` extracts the
unweighted topology, then `scores = Graphs.pagerank(simple, damping, max_iter,
tol)` (i.e. `Graphs.pagerank(g, α, n, ϵ)` with `α = damping`, `n = max_iter`,
`ϵ = tol`). Weights are ignored entirely. Note that undirected self-loops are
dropped on this path (`SimpleGraph` limitation, consistent with
`to_simple_graph`). Non-convergence raises Graphs' `ErrorException`.

# Weighted-path requirements

Effective weights must be finite and non-negative (the `ArgumentError` covers
negative values, NaN and ±Inf): the retained pairs (values pruned by the
filter cannot violate this) are streamed once before iterating and the first
retained invalid weight throws an `ArgumentError` naming the offending pair
`(i, j)`. The unweighted path skips this check. Non-convergence of the
weighted iteration throws an `ErrorException`.

# Keyword arguments

- `damping`: teleportation damping factor, `0 < damping < 1` (`ArgumentError`
  otherwise).
- `weighted`: `true` (default) runs the flow-weighted iteration above;
  `false` delegates to `Graphs.pagerank` on the extracted simple graph.
- `tol`: convergence tolerance, `tol > 0` (`ArgumentError` otherwise); maps to
  Graphs' `ϵ`.
- `max_iter`: maximum iterations, `max_iter >= 1` (`ArgumentError` otherwise);
  maps to Graphs' `n`.
- MRIO method only: `source`, `weights`, `direction`, `threshold`, `min_share`,
  `self_loops` select the wrapped matrix and graph filter exactly as in
  `mrio_graph` (which the MRIO method builds and forwards to; the Leontief
  factorization is never touched).

The graph must be non-empty (`nv(g) >= 1`, else `ArgumentError`).

# Returns

A `SeriesEntry` holding the `Vector{Float64}` scores over `g.nodes` (the
shared node `DataFrame` reference, never copied).

# Memory

Weighted iteration allocates exactly three length-`n` `Float64` vectors
(`r`, `work`, `inv_s`) and nothing per iteration; the filter is applied
on the fly inside the kernels, never as a materialized pruned matrix.
"""
function pagerank_scores(
        g::MRIOGraph{TV, TI, D};
        damping::Real = 0.85,
        weighted::Bool = true,
        tol::Real = 1.0e-6,
        max_iter::Integer = 100,
    ) where {TV, TI, D}
    n = Graphs.nv(g)
    _validate_pagerank_args(damping, tol, max_iter, n)
    if !weighted
        simple, _ = to_simple_graph(g)
        scores = Graphs.pagerank(simple, Float64(damping), Int(max_iter), Float64(tol))
        return Juliora.SeriesEntry(Vector{Float64}(scores), g.nodes)
    end
    _check_nonnegative_weights(g)
    alpha = Float64(damping)
    tolerance = Float64(tol)
    niter = Int(max_iter)
    r = fill(1.0 / n, n)
    work = Vector{Float64}(undef, n)
    inv_s = Vector{Float64}(undef, n)
    if D
        out_strengths!(inv_s, g.weights, g.filter)
    else
        symmetric_strengths!(inv_s, g.weights, g.filter)
    end
    @inbounds for i in 1:n
        s = inv_s[i]
        inv_s[i] = s == 0.0 ? 0.0 : 1.0 / s
    end
    W = g.weights
    f = g.filter
    err = 0.0
    @inbounds for _ in 1:niter
        dangling = 0.0
        for i in 1:n
            if inv_s[i] == 0.0
                dangling += r[i]
            end
        end
        if D
            filtered_tvec!(work, W, f, r, inv_s)
        else
            symmetric_tvec!(work, W, f, r, inv_s)
        end
        beta = (1.0 - alpha + alpha * dangling) / n
        err = 0.0
        for j in 1:n
            newval = alpha * work[j] + beta
            err += abs(newval - r[j])
            r[j] = newval
        end
        if err < n * tolerance
            return Juliora.SeriesEntry(r, g.nodes)
        end
    end
    return error(
        "pagerank_scores did not converge after max_iter=$niter iterations " *
            "(last L1 change $err); increase max_iter or tol",
    )
end

"""
    pagerank_scores(mrio::MRIO; source=nothing, weights=nothing, direction=:directed, threshold=0.0, min_share=0.0, self_loops=false, damping=0.85, weighted=true, tol=1.0e-6, max_iter=100) -> SeriesEntry

Build `mrio_graph(mrio; source, weights, direction, threshold, min_share,
self_loops)` and forward to `pagerank_scores(g; damping, weighted, tol,
max_iter)`. The wrapped transactions/technical matrix is used by reference;
the Leontief factorization is never touched. Returns a `SeriesEntry` over the
selected matrix's supplier-side `row_indices` (shared reference). See the
`MRIOGraph` method for the iteration, semantics and keyword documentation.
"""
function pagerank_scores(
        mrio::Juliora.MRIO;
        source::Union{Nothing, Symbol} = nothing,
        weights::Union{Nothing, Symbol} = nothing,
        direction::Symbol = :directed,
        threshold::Real = 0.0,
        min_share::Real = 0.0,
        self_loops::Bool = false,
        damping::Real = 0.85,
        weighted::Bool = true,
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
    return pagerank_scores(g; damping = damping, weighted = weighted, tol = tol, max_iter = max_iter)
end

"""
    _validate_pagerank_args(damping, tol, max_iter, n)

Validate the PageRank iteration parameters shared by both code paths:
`0 < damping < 1`, `tol > 0`, `max_iter >= 1` and `n >= 1`, throwing an
`ArgumentError` otherwise.
"""
function _validate_pagerank_args(damping::Real, tol::Real, max_iter::Integer, n::Integer)
    0 < Float64(damping) < 1 ||
        throw(ArgumentError("damping must satisfy 0 < damping < 1, got $damping"))
    Float64(tol) > 0 || throw(ArgumentError("tol must be > 0, got $tol"))
    Int(max_iter) >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    Int(n) >= 1 || throw(ArgumentError("pagerank_scores requires a non-empty graph (nv(g) = $n)"))
    return nothing
end

"""
    _check_nonnegative_weights(g::MRIOGraph, fname::AbstractString = "pagerank_scores (weighted)")

Stream the retained effective-weight pairs (the same traversal as
`count_edges`: ordered `(i, j)` per the self-loop rule for directed graphs,
diagonal plus `i < j` for undirected graphs) and throw an `ArgumentError`
naming the first pair `(i, j)` whose retained effective weight is not finite
and non-negative (negative, NaN or ±Inf).
Values pruned by the filter cannot violate this check. `fname` names the
calling entry point in the thrown message.
"""
function _check_nonnegative_weights(
        g::MRIOGraph{TV, TI, D},
        fname::AbstractString = "pagerank_scores (weighted)",
    ) where {TV, TI, D}
    W = g.weights
    n = size(W, 1)
    cutoff = weight_cutoff(g.filter)
    self_loops = g.filter[3]
    if D
        @inbounds for j in 1:n
            for i in 1:n
                if i != j || self_loops
                    x = W[i, j]
                    if x != zero(x) && !(abs(x) < cutoff) && !(isfinite(x) && x >= zero(x))
                        throw(
                            ArgumentError(
                                "$(fname) requires finite non-negative effective weights, got w($i, $j) = $x",
                            ),
                        )
                    end
                end
            end
        end
    else
        if self_loops
            @inbounds for i in 1:n
                x = W[i, i]
                if x != zero(x) && !(abs(x) < cutoff) && !(isfinite(x) && x >= zero(x))
                    throw(
                        ArgumentError(
                            "$(fname) requires finite non-negative effective weights, got w($i, $i) = $x",
                        ),
                    )
                end
            end
        end
        @inbounds for j in 1:n
            for i in 1:(j - 1)
                x = W[i, j] + W[j, i]
                if x != zero(x) && !(abs(x) < cutoff) && !(isfinite(x) && x >= zero(x))
                    throw(
                        ArgumentError(
                            "$(fname) requires finite non-negative effective weights, got w($i, $j) = $x",
                        ),
                    )
                end
            end
        end
    end
    return nothing
end
