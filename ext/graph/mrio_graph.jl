# Zero-copy dense graph view over an MRIO matrix (`MRIOGraph`).
#
# Memory rules: the wrapped dense matrix is the source of truth. Construction
# allocates no n×n structure — no `W + W'`, no `copy`, no `sparse(W)`, no
# `Float32` conversion. Symmetrized weights are read on the fly
# (`W[i, j] + W[j, i]`) and `threshold`/`min_share` filtering happens inside
# the kernels in `dense_kernels.jl`. `to_simple_graph` is the only opt-in
# extraction path.

"""
    MRIOGraph{TV,TI,D} <: Graphs.AbstractGraph{TI}

Zero-copy graph view over a dense MRIO matrix.

An edge `i → j` means "monetary flow supplied by node `i` to buyer `j`"
(`Z[i, j]` of the wrapped transactions matrix).

# Type parameters
- `TV<:Real`: element type of the wrapped matrix (kept as-is; never converted).
- `TI<:Integer`: vertex id type (`Int32` in practice; memory-first).
- `D::Bool`: directedness value parameter (`true` for `:directed`). Graphs.jl's
  contract is `is_directed(::Type)`, so directedness is encoded in the type and
  SimpleTraits-based generics dispatch on it.

# Fields
- `weights::Matrix{TV}`: zero-copy reference to the dense MRIO matrix. It
  `===` the input matrix whenever the caller passes a `Matrix`; only a
  non-`Matrix` `AbstractMatrix` input is copied once via `Matrix(W)`.
- `nodes::DataFrame`: shared reference to the node metadata (the selected
  matrix's supplier-side `row_indices`); never copied.
- `filter::Tuple{Float64, Float64, Bool, Float64}`: `(threshold, min_share,
  self_loops, scale)` with `threshold::Float64` the absolute cutoff,
  `min_share::Float64` the minimum share of `scale = Σ|W|`,
  `self_loops::Bool`, and `scale::Float64` computed once in the constructor
  (only when `min_share > 0`, else `0.0`). The effective cutoff is
  `τ = max(threshold, min_share * scale)`; values in `(-τ, τ)` vanish, so
  exactly-zero values are never edges. For `τ = 0` exact zeros are excluded
  by the explicit `x != zero(x)` kernel guards, not by the interval rule.

Directed graphs use `w(i, j) = filt(W[i, j])` (diagonal dropped unless
`self_loops`); undirected graphs use `w(i, j) = filt(W[i, j] + W[j, i])` with
the diagonal counted once, never doubled.

`Graphs.degree(g, v)` inherits Graphs.jl's generic `length(outneighbors(g, v))`
definition, so self-loops count once there (unlike `SimpleGraph`, which counts
them twice).
"""
struct MRIOGraph{TV <: Real, TI <: Integer, D} <: Graphs.AbstractGraph{TI}
    weights::Matrix{TV}
    nodes::DataFrame
    filter::Tuple{Float64, Float64, Bool, Float64}
end

"""
    FilteredWeights{TV,TI,D} <: AbstractMatrix{Float64}

Zero-copy `AbstractMatrix` view returned by `Graphs.weights(g)`: `getindex`
evaluates the filtered/symmetrized effective weight `w(i, j)` on the fly, so
generic Graphs.jl functions (e.g. `Graphs.modularity(g, c;
distmx=Graphs.weights(g))`) see the filtered weights without any copy.
"""
struct FilteredWeights{TV, TI, D} <: AbstractMatrix{Float64}
    W::Matrix{TV}
    cutoff::Float64
    self_loops::Bool
end

Base.size(F::FilteredWeights) = size(F.W)
Base.IndexStyle(::Type{<:FilteredWeights}) = IndexCartesian()
Base.eltype(::Type{<:FilteredWeights}) = Float64

@inline function Base.getindex(F::FilteredWeights{TV, TI, true}, i::Int, j::Int) where {TV, TI}
    @boundscheck checkbounds(F, i, j)
    @inbounds x = filtered_weight(F.W, i, j, F.cutoff, F.self_loops)
    return Float64(x)
end

@inline function Base.getindex(F::FilteredWeights{TV, TI, false}, i::Int, j::Int) where {TV, TI}
    @boundscheck checkbounds(F, i, j)
    @inbounds x = sym_weight(F.W, i, j, F.cutoff, F.self_loops)
    return Float64(x)
end

Base.getindex(F::FilteredWeights, i::Integer, j::Integer) = F[Int(i), Int(j)]

LinearAlgebra.issymmetric(::FilteredWeights{TV, TI, false}) where {TV, TI} = true

"""
    MRIOEdgeIter{TV,TI,D}

Lazy O(1)-memory edge iterator returned by `Graphs.edges(g)`: directed graphs
yield every ordered pair `(i, j)` with nonzero effective weight; undirected
graphs yield `SimpleEdge(min(i, j), max(i, j))` for each unordered pair with
nonzero effective weight (including self-loops `(i, i)` when `self_loops` is
true). A full pass is O(n²) time.
"""
struct MRIOEdgeIter{TV, TI, D}
    g::MRIOGraph{TV, TI, D}
end

Base.eltype(::Type{<:MRIOEdgeIter{TV, TI, D}}) where {TV, TI, D} = Graphs.SimpleEdge{TI}
Base.IteratorEltype(::Type{<:MRIOEdgeIter}) = Base.HasEltype()
Base.IteratorSize(::Type{<:MRIOEdgeIter}) = Base.SizeUnknown()

function Base.iterate(it::MRIOEdgeIter{TV, TI, true}, state::Tuple{Int, Int} = (1, 1)) where {TV, TI}
    W = it.g.weights
    n = size(W, 1)
    cutoff = weight_cutoff(it.g.filter)
    self_loops = it.g.filter[3]
    i, j = state
    @inbounds while i <= n
        while j <= n
            if i != j || self_loops
                x = W[i, j]
                if x != zero(x) && !(abs(x) < cutoff)
                    edge = Graphs.SimpleEdge{TI}(TI(i), TI(j))
                    return edge, j < n ? (i, j + 1) : (i + 1, 1)
                end
            end
            j += 1
        end
        i += 1
        j = 1
    end
    return nothing
end

function Base.iterate(it::MRIOEdgeIter{TV, TI, false}, state::Tuple{Int, Int} = (1, 1)) where {TV, TI}
    W = it.g.weights
    n = size(W, 1)
    cutoff = weight_cutoff(it.g.filter)
    self_loops = it.g.filter[3]
    i, j = state
    @inbounds while i <= n
        while j <= n
            if j >= i && (i != j || self_loops)
                x = i == j ? W[i, i] : W[i, j] + W[j, i]
                if x != zero(x) && !(abs(x) < cutoff)
                    edge = Graphs.SimpleEdge{TI}(TI(i), TI(j))
                    return edge, j < n ? (i, j + 1) : (i + 1, i + 1)
                end
            end
            j += 1
        end
        i += 1
        j = i
    end
    return nothing
end

# --- Graphs.jl AbstractGraph interface ---

Graphs.is_directed(::Type{<:MRIOGraph{TV, TI, D}}) where {TV, TI, D} = D

Base.eltype(::Type{<:MRIOGraph{TV, TI, D}}) where {TV, TI, D} = TI

Graphs.edgetype(::Type{<:MRIOGraph{TV, TI, D}}) where {TV, TI, D} = Graphs.SimpleEdge{TI}
# Instance-level method required: Graphs.jl defines `edgetype(::AbstractGraph)`
# as not-implemented with no delegation to the `Type`-level method, so without
# this `Graphs.edgetype(g)` on an instance throws.
Graphs.edgetype(g::MRIOGraph) = Graphs.edgetype(typeof(g))

Graphs.nv(g::MRIOGraph) = size(g.weights, 1)
Graphs.vertices(g::MRIOGraph{TV, TI}) where {TV, TI} = Base.OneTo(TI(size(g.weights, 1)))

"""
    Graphs.ne(g::MRIOGraph)

Number of post-filter edges: ordered nonzero pairs (directed) or unordered
pairs `i ≤ j` (undirected). Computed by one O(n²) streaming count pass per
call.
"""
Graphs.ne(g::MRIOGraph{TV, TI, D}) where {TV, TI, D} = count_edges(g.weights, g.filter, D)

function Graphs.has_edge(g::MRIOGraph{TV, TI, D}, s::Integer, d::Integer) where {TV, TI, D}
    n = size(g.weights, 1)
    (1 <= s <= n && 1 <= d <= n) || return false
    cutoff = weight_cutoff(g.filter)
    self_loops = g.filter[3]
    i, j = Int(s), Int(d)
    w = D ? filtered_weight(g.weights, i, j, cutoff, self_loops) : sym_weight(g.weights, i, j, cutoff, self_loops)
    return w != zero(w)
end

function Graphs.outneighbors(g::MRIOGraph{TV, TI, D}, v::Integer) where {TV, TI, D}
    n = size(g.weights, 1)
    1 <= v <= n || throw(BoundsError(g, v))
    cutoff = weight_cutoff(g.filter)
    self_loops = g.filter[3]
    W = g.weights
    iv = Int(v)
    nbrs = TI[]
    if D
        @inbounds for j in 1:n
            if filtered_weight(W, iv, j, cutoff, self_loops) != zero(eltype(W))
                push!(nbrs, TI(j))
            end
        end
    else
        @inbounds for j in 1:n
            if sym_weight(W, iv, j, cutoff, self_loops) != zero(eltype(W))
                push!(nbrs, TI(j))
            end
        end
    end
    return nbrs
end

function Graphs.inneighbors(g::MRIOGraph{TV, TI, D}, v::Integer) where {TV, TI, D}
    n = size(g.weights, 1)
    1 <= v <= n || throw(BoundsError(g, v))
    cutoff = weight_cutoff(g.filter)
    self_loops = g.filter[3]
    W = g.weights
    jv = Int(v)
    nbrs = TI[]
    if D
        @inbounds for i in 1:n
            if filtered_weight(W, i, jv, cutoff, self_loops) != zero(eltype(W))
                push!(nbrs, TI(i))
            end
        end
    else
        @inbounds for i in 1:n
            if sym_weight(W, jv, i, cutoff, self_loops) != zero(eltype(W))
                push!(nbrs, TI(i))
            end
        end
    end
    return nbrs
end

Graphs.edges(g::MRIOGraph{TV, TI, D}) where {TV, TI, D} = MRIOEdgeIter{TV, TI, D}(g)
Graphs.weights(g::MRIOGraph{TV, TI, D}) where {TV, TI, D} =
    FilteredWeights{TV, TI, D}(g.weights, weight_cutoff(g.filter), g.filter[3])

# --- construction ---

"""
    _graph_scale(W, min_share) -> Float64

`Σ|W|` in one zero-allocation O(n²) pass when `min_share > 0`, else `0.0`.
"""
function _graph_scale(W::AbstractMatrix, min_share::Float64)
    if min_share <= 0
        return 0.0
    end
    total = 0.0
    @inbounds for j in axes(W, 2)
        for i in axes(W, 1)
            total += abs(W[i, j])
        end
    end
    return total
end

function _validate_graph_options(direction, threshold, min_share)
    direction in (:directed, :undirected) ||
        throw(ArgumentError("direction must be :directed or :undirected, got $direction"))
    th = Float64(threshold)
    th >= 0 || throw(ArgumentError("threshold must be ≥ 0, got $threshold"))
    ms = Float64(min_share)
    0 <= ms <= 1 || throw(ArgumentError("min_share must satisfy 0 ≤ min_share ≤ 1, got $min_share"))
    return th, ms
end

"""
    _wrap_mrio_matrix(W, nodes; direction, threshold, min_share, self_loops) -> MRIOGraph

Shared construction path: validates squareness (`DimensionMismatch` unless
`size(W, 1) == size(W, 2) == nrow(nodes)`), stores `W` as-is when it is a
`Matrix` (so `g.weights === W`), copies once via `Matrix(W)` only for
non-`Matrix` `AbstractMatrix` input, and computes the `min_share` scale in one
O(n²) pass. Never converts the element type (a `Float32` matrix stays
`Float32`) and never densifies/sparsifies anything.
"""
function _wrap_mrio_matrix(
        W::AbstractMatrix,
        nodes::DataFrame;
        direction::Symbol = :directed,
        threshold::Real = 0.0,
        min_share::Real = 0.0,
        self_loops::Bool = false,
    )
    th, ms = _validate_graph_options(direction, threshold, min_share)
    sl = Bool(self_loops)
    nrows, ncols = size(W, 1), size(W, 2)
    nrows == ncols ||
        throw(DimensionMismatch("weight matrix must be square, got $(size(W))"))
    nrow(nodes) == nrows || throw(
        DimensionMismatch("node table has $(nrow(nodes)) rows but the weight matrix is $nrows × $ncols"),
    )
    dense = W isa Matrix ? W : Matrix(W)
    scale = _graph_scale(dense, ms)
    filt = (th, ms, sl, scale)
    directed = direction === :directed
    return MRIOGraph{eltype(dense), Int32, directed}(dense, nodes, filt)
end

function _entry_for_source(mrio::Juliora.MRIO, source::Symbol)
    if source === :Z || source === :T
        return mrio.T
    elseif source === :A
        return mrio.A
    else
        throw(ArgumentError("source must be one of :Z, :T, :A, got $source"))
    end
end

function _entry_for_weights(mrio::Juliora.MRIO, weights::Symbol)
    if weights === :flows
        return mrio.T
    elseif weights === :technical
        return mrio.A
    else
        throw(ArgumentError("weights must be :flows or :technical, got $weights"))
    end
end

"""
    mrio_graph(mrio::MRIO; source=nothing, weights=nothing, direction=:directed, threshold=0.0, min_share=0.0, self_loops=false) -> MRIOGraph

Build a zero-copy graph view over an MRIO database matrix.

An edge `i → j` means "monetary flow supplied by node `i` to buyer `j`"
(`Z[i, j]` of the wrapped transactions matrix). Self-flows are dropped by
default (`self_loops=false`).

# Matrix selection
`weights = :flows` selects the transactions matrix `mrio.T` (a.k.a. `Z`);
`weights = :technical` selects `mrio.A`. `source = :Z`/`:T` selects `mrio.T`;
`source = :A` selects `mrio.A`. Both `nothing` (default) selects `mrio.T`. If
both are given they must select the same matrix, else an `ArgumentError` is
thrown.

# Filtering
`threshold ≥ 0` is an absolute cutoff on `|w|`; `0 ≤ min_share ≤ 1` is a
minimum share of `Σ|W|` (effective cutoff `τ = max(threshold, min_share *
Σ|W|)`). Both are applied on the fly inside the graph kernels, never as a
materialized pruned matrix.

# Memory
The wrapped matrix and the node table (the selected matrix's supplier-side
`row_indices`) are stored by reference — construction allocates no copy.
Graph-only workflows can use `mrio_graph(W, nodes)` to skip constructing a
full `MRIO` (and its Leontief factorization) entirely. Graph code never
touches `LeontiefFactorization.data`, which would materialize a dense n×n
inverse.
"""
function mrio_graph(
        mrio::Juliora.MRIO;
        source::Union{Nothing, Symbol} = nothing,
        weights::Union{Nothing, Symbol} = nothing,
        direction::Symbol = :directed,
        threshold::Real = 0.0,
        min_share::Real = 0.0,
        self_loops::Bool = false,
    )
    from_source = source === nothing ? nothing : _entry_for_source(mrio, source)
    from_weights = weights === nothing ? nothing : _entry_for_weights(mrio, weights)
    entry = if from_source !== nothing && from_weights !== nothing
        from_source === from_weights || throw(
            ArgumentError("source $source and weights $weights select different matrices"),
        )
        from_source
    elseif from_source !== nothing
        from_source
    elseif from_weights !== nothing
        from_weights
    else
        mrio.T
    end
    return _wrap_mrio_matrix(
        entry.data,
        entry.row_indices;
        direction = direction,
        threshold = threshold,
        min_share = min_share,
        self_loops = self_loops,
    )
end

"""
    mrio_graph(W::AbstractMatrix, nodes::DataFrame; direction=:directed, threshold=0.0, min_share=0.0, self_loops=false) -> MRIOGraph

Build a zero-copy graph view over a raw square matrix `W` with `nrow(nodes) ==
size(W, 1)` (`DimensionMismatch` otherwise). Edge convention, filtering and
memory semantics are identical to the `MRIO` method: `W[i, j]` is the flow
supplied by `i` to buyer `j`; when `W isa Matrix` it is stored as-is
(`g.weights === W`), otherwise it is copied once via `Matrix(W)`; the element
type is never converted.
"""
function mrio_graph(
        W::AbstractMatrix,
        nodes::DataFrame;
        direction::Symbol = :directed,
        threshold::Real = 0.0,
        min_share::Real = 0.0,
        self_loops::Bool = false,
    )
    return _wrap_mrio_matrix(
        W,
        nodes;
        direction = direction,
        threshold = threshold,
        min_share = min_share,
        self_loops = self_loops,
    )
end

"""
    graph_summary(g::MRIOGraph) -> DataFrame

Summarize `g` in a one-row `DataFrame` with columns `nodes`, `edges`
(post-filter `ne(g)`), `directed`, `threshold`, `min_share`, `self_loops`,
`total_weight`, `retained_weight`, `retained_share` and `memory_bytes`.

Candidate pairs are all `(i, j)` when self-loops are kept, else `i != j`
(directed), or unordered `i ≤ j` analogously (undirected). `total_weight`
sums `|raw pair weight|` over the candidates (directed: `W[i, j]`;
undirected off-diagonal: `W[i, j] + W[j, i]`; diagonal: `W[i, i]`),
`retained_weight` sums `|w(i, j)|` over the surviving pairs, and
`retained_share = retained / total` (`1.0` when `total == 0`).
`memory_bytes = sizeof(g.weights)` is the memory the graph references
(shared with the MRIO; construction itself allocates no copy).

Runs a single O(n²) streaming pass over the dense matrix per call.
"""
function graph_summary(g::MRIOGraph{TV, TI, D}) where {TV, TI, D}
    total, retained, nedges = weight_sums(g.weights, g.filter, D)
    return DataFrame(
        nodes = size(g.weights, 1),
        edges = nedges,
        directed = D,
        threshold = g.filter[1],
        min_share = g.filter[2],
        self_loops = g.filter[3],
        total_weight = total,
        retained_weight = retained,
        retained_share = total == 0 ? 1.0 : retained / total,
        memory_bytes = sizeof(g.weights),
    )
end

"""
    _topk_keep_mask(srcs, dsts, vals, n, k, directed) -> BitVector

Per-node top-k edge mask over the candidate triples: a directed edge is kept
iff it is among the `k` heaviest (by `|w|`) out-edges of its source node; an
undirected edge is kept iff it is among the `k` heaviest incident edges of
either endpoint. Ties at the cutoff are kept.
"""
function _topk_keep_mask(
        srcs::AbstractVector{<:Integer},
        dsts::AbstractVector{<:Integer},
        vals::AbstractVector{Float32},
        n::Int,
        k::Int,
        directed::Bool,
    )
    m = length(vals)
    keep = trues(m)
    incident = [Int[] for _ in 1:n]
    if directed
        for e in 1:m
            push!(incident[Int(srcs[e])], e)
        end
        for v in 1:n
            _apply_topk_cutoff!(keep, vals, incident[v], k)
        end
    else
        for e in 1:m
            push!(incident[Int(srcs[e])], e)
            if Int(dsts[e]) != Int(srcs[e])
                push!(incident[Int(dsts[e])], e)
            end
        end
        cutoffs = fill(Float32(-Inf), n)
        for v in 1:n
            idx = incident[v]
            if length(idx) > k
                scratch = Vector{Float32}(undef, length(idx))
                for t in eachindex(idx)
                    scratch[t] = abs(vals[idx[t]])
                end
                cutoffs[v] = partialsort!(scratch, k; rev = true)
            end
        end
        for e in 1:m
            if abs(vals[e]) < cutoffs[Int(srcs[e])] && abs(vals[e]) < cutoffs[Int(dsts[e])]
                keep[e] = false
            end
        end
    end
    return keep
end

function _apply_topk_cutoff!(keep::AbstractVector{Bool}, vals::AbstractVector{Float32}, idx::Vector{Int}, k::Int)
    if length(idx) > k
        scratch = Vector{Float32}(undef, length(idx))
        for t in eachindex(idx)
            scratch[t] = abs(vals[idx[t]])
        end
        cutoff = partialsort!(scratch, k; rev = true)
        for e in idx
            if abs(vals[e]) < cutoff
                keep[e] = false
            end
        end
    end
    return keep
end

"""
    to_simple_graph(g::MRIOGraph; threshold=0.0, topk=nothing) -> (simple_graph, distmx)

Opt-in extraction of `g` into a Graphs.jl simple graph plus its weight
matrix. A single streaming pass over the dense matrix emits the pruned
compact edges `(src::Int32, dst::Int32, w::Float32)` (per-node `topk`
pruning, when requested, filters the emitted candidates afterwards).

The effective weight is `w(i, j)` per the `MRIOGraph` filter, with the
additional absolute cutoff `threshold` applied on top (`|w| < threshold` is
also dropped). `topk::Union{Nothing,Integer}` keeps an edge only if it is
among the `topk` heaviest (by `|w|`) out-edges of its source node (directed)
or incident edges of either endpoint (undirected).

Returns `(simple_graph, distmx)` where `simple_graph` is a
`Graphs.SimpleDiGraph{Int32}` (directed) or `Graphs.SimpleGraph{Int32}`
(undirected) over `n` vertices, and `distmx` is the `SparseMatrixCSC{Float32}`
built from the extracted triples — the `distmx` to pass to Graphs.jl
weight-aware generics. Note that `SimpleGraph` has no self-loops, so
undirected self-loops are dropped from both the graph and `distmx` (and are
therefore not emitted at all); for undirected graphs `distmx` holds both
`(i, j)` and `(j, i)` entries.
"""
function to_simple_graph(
        g::MRIOGraph{TV, TI, D};
        threshold::Real = 0.0,
        topk::Union{Nothing, Integer} = nothing,
    ) where {TV, TI, D}
    extra = Float64(threshold)
    extra >= 0 || throw(ArgumentError("threshold must be ≥ 0, got $threshold"))
    if topk !== nothing
        topk >= 1 || throw(ArgumentError("topk must be ≥ 1, got $topk"))
    end
    W = g.weights
    n = size(W, 1)
    cutoff = weight_cutoff(g.filter)
    self_loops = g.filter[3]
    srcs = TI[]
    dsts = TI[]
    vals = Float32[]
    if D
        @inbounds for j in 1:n
            for i in 1:n
                if i != j || self_loops
                    x = W[i, j]
                    if x != zero(x) && !(abs(x) < cutoff) && !(abs(Float64(x)) < extra)
                        push!(srcs, TI(i))
                        push!(dsts, TI(j))
                        push!(vals, Float32(x))
                    end
                end
            end
        end
    else
        # Undirected self-loops are dropped: SimpleGraph cannot represent them.
        @inbounds for j in 1:n
            for i in 1:(j - 1)
                x = W[i, j] + W[j, i]
                if x != zero(x) && !(abs(x) < cutoff) && !(abs(Float64(x)) < extra)
                    push!(srcs, TI(i))
                    push!(dsts, TI(j))
                    push!(vals, Float32(x))
                end
            end
        end
    end
    if topk !== nothing
        keep = _topk_keep_mask(srcs, dsts, vals, n, Int(topk), D)
        srcs = srcs[keep]
        dsts = dsts[keep]
        vals = vals[keep]
    end
    if D
        simple = Graphs.SimpleDiGraph{TI}(n)
        for e in eachindex(srcs)
            Graphs.add_edge!(simple, srcs[e], dsts[e])
        end
        distmx = SparseArrays.sparse(srcs, dsts, vals, n, n)
        return simple, distmx
    else
        simple = Graphs.SimpleGraph{TI}(n)
        for e in eachindex(srcs)
            Graphs.add_edge!(simple, srcs[e], dsts[e])
        end
        both_srcs = vcat(srcs, dsts)
        both_dsts = vcat(dsts, srcs)
        both_vals = vcat(vals, vals)
        distmx = SparseArrays.sparse(both_srcs, both_dsts, both_vals, n, n)
        return simple, distmx
    end
end
