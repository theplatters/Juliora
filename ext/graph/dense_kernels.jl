# Dense on-the-fly filtering and streaming kernels for `MRIOGraph`.
#
# The dense MRIO matrix is the source of truth. These kernels never allocate an
# O(n²) scratch structure: filtering is applied elementwise inside the kernels
# and only O(n) accumulators (provided by the caller) plus tile-local scratch
# are used.

"""
    weight_cutoff(f) -> Float64

Effective absolute cutoff `τ = max(threshold, min_share * scale)` for the
filter tuple `f = (threshold, min_share, self_loops, scale)`.
"""
@inline function weight_cutoff(f)
    return max(Float64(f[1]), Float64(f[2]) * Float64(f[4]))
end

"""
    filtered_weight(W, i, j, cutoff, self_loops)

Directed effective weight `w(i, j)`: `W[i, j]` with `|x| < cutoff` mapped to
zero, and the diagonal dropped unless `self_loops` is true.
"""
@inline function filtered_weight(W, i, j, cutoff, self_loops)
    if i == j && !self_loops
        return zero(eltype(W))
    end
    x = W[i, j]
    return abs(x) < cutoff ? zero(x) : x
end

"""
    sym_weight(W, i, j, cutoff, self_loops)

Undirected effective weight `w(i, j)`: off-diagonal pairs read the symmetrized
value `W[i, j] + W[j, i]` on the fly (the diagonal is counted once, never
doubled) with `|x| < cutoff` mapped to zero; the diagonal is dropped unless
`self_loops` is true.
"""
@inline function sym_weight(W, i, j, cutoff, self_loops)
    if i == j
        if !self_loops
            return zero(eltype(W))
        end
        x = W[i, i]
        return abs(x) < cutoff ? zero(x) : x
    end
    x = W[i, j] + W[j, i]
    return abs(x) < cutoff ? zero(x) : x
end

"""
    out_strengths!(s, W, f) -> s

Fill `s[i]` with `Σ_j w(i, j)` over the directed effective weights. One O(n²)
streaming pass in column-major order (outer loop over columns `j`, inner loop
over rows `i`, scatter-accumulating into `s[i]`), O(n) result. `s` must have
length `size(W, 1)`.
"""
function out_strengths!(s, W, f)
    n = size(W, 1)
    size(W, 2) == n || throw(DimensionMismatch("weight matrix must be square, got $(size(W))"))
    length(s) == n || throw(DimensionMismatch("accumulator length $(length(s)) must match matrix size $n"))
    cutoff = weight_cutoff(f)
    self_loops = f[3]
    fill!(s, zero(eltype(s)))
    @inbounds for j in 1:n
        for i in 1:n
            s[i] += filtered_weight(W, i, j, cutoff, self_loops)
        end
    end
    return s
end

"""
    in_strengths!(s, W, f) -> s

Fill `s[j]` with `Σ_i w(i, j)` over the directed effective weights. One O(n²)
streaming pass (column-major, cache-friendly), O(n) result. `s` must have
length `size(W, 1)`.
"""
function in_strengths!(s, W, f)
    n = size(W, 1)
    size(W, 2) == n || throw(DimensionMismatch("weight matrix must be square, got $(size(W))"))
    length(s) == n || throw(DimensionMismatch("accumulator length $(length(s)) must match matrix size $n"))
    cutoff = weight_cutoff(f)
    self_loops = f[3]
    fill!(s, zero(eltype(s)))
    @inbounds for j in 1:n
        acc = zero(eltype(s))
        for i in 1:n
            acc += filtered_weight(W, i, j, cutoff, self_loops)
        end
        s[j] = acc
    end
    return s
end

"""
    symmetric_strengths!(s, W, f) -> s

Fill `s[i]` with `Σ_j w(i, j)` over the undirected (symmetrized) effective
weights, implemented on top of [`symmetric_tile_accumulate!`](@ref). `s` must
have length `size(W, 1)`.
"""
function symmetric_strengths!(s, W, f)
    n = size(W, 1)
    size(W, 2) == n || throw(DimensionMismatch("weight matrix must be square, got $(size(W))"))
    length(s) == n || throw(DimensionMismatch("accumulator length $(length(s)) must match matrix size $n"))
    fill!(s, zero(eltype(s)))
    symmetric_tile_accumulate!(
        (i, j, w) -> begin
            s[i] += w
            if i != j
                s[j] += w
            end
        end,
        W,
        f,
    )
    return s
end

"""
    symmetric_tile_accumulate!(op, W, f; tile_size=512)

Stream the upper-triangular tiles of the square matrix `W` (default tile edge
`tile_size = 512`), reading both `(i, j)` and its mirror `(j, i)` per tile so
each element of `W` is read exactly once in cache-friendly order. Calls
`op(i, j, w)` once per surviving unordered pair `i < j` and once per surviving
diagonal entry `op(i, i, w)`, where `w` is the undirected effective weight per
[`sym_weight`](@ref) and "surviving" means `w != 0`. Never allocates O(n²)
scratch.
"""
function symmetric_tile_accumulate!(op, W, f; tile_size::Int = 512)
    n = size(W, 1)
    size(W, 2) == n || throw(DimensionMismatch("weight matrix must be square, got $(size(W))"))
    tile_size >= 1 || throw(ArgumentError("tile_size must be ≥ 1, got $tile_size"))
    cutoff = weight_cutoff(f)
    self_loops = f[3]
    if self_loops
        @inbounds for i in 1:n
            x = W[i, i]
            if x != zero(x) && !(abs(x) < cutoff)
                op(i, i, x)
            end
        end
    end
    @inbounds for jb in 1:tile_size:n
        jhi = min(jb + tile_size - 1, n)
        ib = 1
        while ib < jb
            ihi = min(ib + tile_size - 1, n)
            for j in jb:jhi
                for i in ib:ihi
                    x = W[i, j] + W[j, i]
                    if x != zero(x) && !(abs(x) < cutoff)
                        op(i, j, x)
                    end
                end
            end
            ib += tile_size
        end
        for i in jb:jhi
            for j in (i + 1):jhi
                x = W[i, j] + W[j, i]
                if x != zero(x) && !(abs(x) < cutoff)
                    op(i, j, x)
                end
            end
        end
    end
    return nothing
end

"""
    count_edges(W, f, directed::Bool) -> Int

Streaming edge count matching `ne(g)` semantics: ordered nonzero pairs for
directed graphs, unordered pairs `i ≤ j` (diagonal included only when the
filter keeps self-loops) for undirected graphs. O(n²) time, O(1) memory.
"""
function count_edges(W, f, directed::Bool)
    n = size(W, 1)
    size(W, 2) == n || throw(DimensionMismatch("weight matrix must be square, got $(size(W))"))
    cutoff = weight_cutoff(f)
    self_loops = f[3]
    count = 0
    if directed
        @inbounds for j in 1:n
            for i in 1:n
                if i != j || self_loops
                    x = W[i, j]
                    if x != zero(x) && !(abs(x) < cutoff)
                        count += 1
                    end
                end
            end
        end
    else
        if self_loops
            @inbounds for i in 1:n
                x = W[i, i]
                if x != zero(x) && !(abs(x) < cutoff)
                    count += 1
                end
            end
        end
        @inbounds for j in 1:n
            for i in 1:(j - 1)
                x = W[i, j] + W[j, i]
                if x != zero(x) && !(abs(x) < cutoff)
                    count += 1
                end
            end
        end
    end
    return count
end

"""
    weight_sums(W, f, directed::Bool) -> (total, retained, nedges)

Streaming `(total_weight, retained_weight, nedges)` triple as defined for
`graph_summary`: `total_weight` sums `|raw pair weight|` over the candidate
pairs (all `(i, j)` when self-loops are kept, else `i != j`; unordered `i ≤ j`
analogously for undirected graphs, with off-diagonal raw pair weight
`W[i, j] + W[j, i]` and diagonal raw weight `W[i, i]`),
`retained_weight` sums `|w(i, j)|` over the surviving (nonzero effective
weight) pairs, and `nedges` counts the surviving pairs. O(n²) time, O(1)
memory.
"""
function weight_sums(W, f, directed::Bool)
    n = size(W, 1)
    size(W, 2) == n || throw(DimensionMismatch("weight matrix must be square, got $(size(W))"))
    cutoff = weight_cutoff(f)
    self_loops = f[3]
    total = 0.0
    retained = 0.0
    nedges = 0
    if directed
        @inbounds for j in 1:n
            for i in 1:n
                if i != j || self_loops
                    x = W[i, j]
                    total += abs(x)
                    if x != zero(x) && !(abs(x) < cutoff)
                        retained += abs(x)
                        nedges += 1
                    end
                end
            end
        end
    else
        if self_loops
            @inbounds for i in 1:n
                x = W[i, i]
                total += abs(x)
                if x != zero(x) && !(abs(x) < cutoff)
                    retained += abs(x)
                    nedges += 1
                end
            end
        end
        @inbounds for j in 1:n
            for i in 1:(j - 1)
                x = W[i, j] + W[j, i]
                total += abs(x)
                if x != zero(x) && !(abs(x) < cutoff)
                    retained += abs(x)
                    nedges += 1
                end
            end
        end
    end
    return total, retained, nedges
end

"""
    filtered_tvec!(y, W, f, x, inv_s) -> y

Fill `y[j]` with `Σ_i w(i, j) * x[i] * inv_s[i]` over the directed effective
weights `w(i, j)` per [`filtered_weight`](@ref). One O(n²) streaming pass in
column-major order (outer loop over columns `j` with a local accumulator, as
in `in_strengths!`, inner loop over rows `i`), O(n) result. `y`, `x` and
`inv_s` must have length `size(W, 1)`.

NOTE: a BLAS `mul!(y, transpose(W), x)` cannot apply the on-the-fly filter
(rule R3) and the self-loop rule without materializing a pruned matrix; this
hand-written column-major loop performs the same contiguous access pattern
and allocates nothing.
"""
function filtered_tvec!(y, W, f, x, inv_s)
    n = size(W, 1)
    size(W, 2) == n || throw(DimensionMismatch("weight matrix must be square, got $(size(W))"))
    length(y) == n || throw(DimensionMismatch("output length $(length(y)) must match matrix size $n"))
    length(x) == n || throw(DimensionMismatch("input length $(length(x)) must match matrix size $n"))
    length(inv_s) == n || throw(DimensionMismatch("inverse-strength length $(length(inv_s)) must match matrix size $n"))
    cutoff = weight_cutoff(f)
    self_loops = f[3]
    fill!(y, zero(eltype(y)))
    @inbounds for j in 1:n
        acc = zero(eltype(y))
        for i in 1:n
            w = filtered_weight(W, i, j, cutoff, self_loops)
            if w != zero(w)
                acc += w * x[i] * inv_s[i]
            end
        end
        y[j] = acc
    end
    return y
end

"""
    symmetric_tvec!(y, W, f, x, inv_s) -> y

Fill `y[j]` with `Σ_i w(i, j) * x[i] * inv_s[i]` over the undirected
(symmetrized) effective weights `w(i, j)` per [`sym_weight`](@ref),
implemented on top of [`symmetric_tile_accumulate!`](@ref): each surviving
unordered pair `i < j` contributes `w * x[j] * inv_s[j]` to `y[i]` and
`w * x[i] * inv_s[i]` to `y[j]`, and each surviving diagonal entry
contributes `w * x[i] * inv_s[i]` to `y[i]`. `y`, `x` and `inv_s` must have
length `size(W, 1)`.
"""
function symmetric_tvec!(y, W, f, x, inv_s)
    n = size(W, 1)
    size(W, 2) == n || throw(DimensionMismatch("weight matrix must be square, got $(size(W))"))
    length(y) == n || throw(DimensionMismatch("output length $(length(y)) must match matrix size $n"))
    length(x) == n || throw(DimensionMismatch("input length $(length(x)) must match matrix size $n"))
    length(inv_s) == n || throw(DimensionMismatch("inverse-strength length $(length(inv_s)) must match matrix size $n"))
    fill!(y, zero(eltype(y)))
    symmetric_tile_accumulate!(
        (i, j, w) -> begin
            @inbounds begin
                if i == j
                    y[i] += w * x[i] * inv_s[i]
                else
                    y[i] += w * x[j] * inv_s[j]
                    y[j] += w * x[i] * inv_s[i]
                end
            end
        end,
        W,
        f,
    )
    return y
end
