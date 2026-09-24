# Normalized-Laplacian spectral clustering + small k-means.
#
# `:spectral` always operates on the symmetrized projection of the effective
# weights (`A[i, j] = sym_weight(W, i, j, cutoff, self_loops)`), for directed
# and undirected graphs alike; for undirected graphs this is just the graph's
# effective weights. A directed graph is therefore partitioned via its
# symmetrized projection, while `result.modularity` is still scored on `g` as
# built by `communities`.
#
# Pipeline (`_spectral_partition`): size gate (`n ≤ 5_000`); stream the
# strengths once via `symmetric_strengths!`; exclude isolated nodes (each
# becomes its own singleton community); build the dense symmetric normalized
# Laplacian `L_sym = I − D^(−1/2) A D^(−1/2)` (`Float64`, on the non-isolated
# nodes only); eigen-decompose with `eigen(Symmetric(L_sym))`; pick `k` by
# eigengap (auto) or clamp the request; embed with the Ng–Jordan–Weiss
# row-normalized eigenvectors; cluster the embedding with the in-repo
# k-means++ / Lloyd k-means; emit singletons for isolated and zero-row nodes.
#
# Memory rules: the wrapped matrix is streamed on the fly (strengths pass,
# then the Laplacian build) and never materialized as a pruned copy; the only
# O(n²) workspace is the small dense Laplacian (allowed since `n ≤ 5_000`)
# plus its eigen decomposition. Accumulation and decomposition run in
# `Float64` without converting the wrapped matrix.

"""
    _SPECTRAL_MAX_N

Maximum node count for `:spectral`: the method needs an O(n²) workspace plus
an O(n³) eigen decomposition, so it is gated to `n ≤ 5_000`. Larger graphs
must be reduced first (e.g. `aggregate(mrio, [:CountryCode])`) or clustered
with `algorithm = :louvain` / `:leiden`.
"""
const _SPECTRAL_MAX_N = 5_000

"""
    _SPECTRAL_MAX_ITER

Hard cap on the Lloyd iterations of [`_spectral_kmeans`](@ref) (float-safety
net; normal convergence needs far fewer iterations).
"""
const _SPECTRAL_MAX_ITER = 300

"""
    _SPECTRAL_ZERO_ROW_TOL

Row-norm threshold of the Ng–Jordan–Weiss embedding below which a row counts
as (near-)zero: such rows stay zero and their nodes become singleton
communities instead of entering k-means.
"""
const _SPECTRAL_ZERO_ROW_TOL = 1.0e-12

"""
    _spectral_check_size(n::Int) -> Int

Enforce the spectral size gate: `n > $_SPECTRAL_MAX_N` throws an
`ArgumentError` explaining that spectral clustering needs an O(n²) workspace
plus an O(n³) eigen decomposition, is gated to `n ≤ 5000`, and directing
users to `aggregate(mrio, [:CountryCode])` first or to
`algorithm = :louvain` / `:leiden`. Returns `n` otherwise.
"""
function _spectral_check_size(n::Int)
    n > _SPECTRAL_MAX_N && throw(
        ArgumentError(
            "spectral clustering needs an O(n²) workspace plus an O(n³) eigen decomposition " *
                "and is gated to n ≤ 5000 (got n = $n); " *
                "aggregate(mrio, [:CountryCode]) first or use algorithm = :louvain or algorithm = :leiden",
        ),
    )
    return n
end

"""
    _spectral_auto_k(lambdas, nprime::Int) -> Int

Eigengap cluster count on the `nprime` ascending Laplacian eigenvalues
`lambdas`: with `λ_1 ≤ … ≤ λ_{ngaps+1}` the `ngaps + 1` smallest eigenvalues
for `ngaps = min(nprime − 1, 25)` consecutive gaps `g_i = λ_{i+1} − λ_i`
(`i = 1..ngaps`), return `k = argmax_i g_i` (ties → smallest `i`), clamped
to `1 ≤ k ≤ nprime`. `nprime ≤ 1` gives `k = 1` without any gap work.
"""
function _spectral_auto_k(lambdas::AbstractVector{<:Real}, nprime::Int)
    nprime <= 1 && return 1
    ngaps = min(nprime - 1, 25)
    best_i = 1
    best_gap = lambdas[2] - lambdas[1]
    for i in 2:ngaps
        gap = lambdas[i + 1] - lambdas[i]
        if gap > best_gap
            best_gap = gap
            best_i = i
        end
    end
    return clamp(best_i, 1, nprime)
end

"""
    _spectral_sqdist(A, i, B, j, d) -> Float64

Squared Euclidean distance between row `i` of `A` and row `j` of `B` over the
`d` columns, in `Float64`. Pure helper for the k-means routines.
"""
@inline function _spectral_sqdist(A::AbstractMatrix, i::Int, B::AbstractMatrix, j::Int, d::Int)
    acc = 0.0
    @inbounds for t in 1:d
        diff = Float64(A[i, t]) - Float64(B[j, t])
        acc += diff * diff
    end
    return acc
end

"""
    _spectral_plusplus_seeds(X::Matrix{Float64}, k::Int, rng::Random.AbstractRNG) -> Vector{Int}

k-means++ seeding on the rows of `X`: the first center is uniform over
`1:m`, each subsequent center is drawn ∝ squared distance to the nearest
already-chosen center (all randomness via `rng`). Returns the `k` chosen row
indices (`1 ≤ k ≤ m` required, else `ArgumentError`). Indices are always
distinct when `m ≥ k`: points at zero distance have zero sampling weight,
and when every remaining point coincides with the chosen set the fallback
draws uniformly among the unchosen indices (the center value then necessarily
duplicates an existing one).
"""
function _spectral_plusplus_seeds(X::Matrix{Float64}, k::Int, rng::Random.AbstractRNG)
    m = size(X, 1)
    d = size(X, 2)
    1 <= k <= m || throw(
        ArgumentError("_spectral_plusplus_seeds requires 1 ≤ k ≤ $m (rows of X), got k = $k"),
    )
    chosen = Vector{Int}(undef, k)
    chosen[1] = rand(rng, 1:m)
    k == 1 && return chosen
    in_chosen = falses(m)
    in_chosen[chosen[1]] = true
    best2 = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        best2[i] = _spectral_sqdist(X, i, X, chosen[1], d)
    end
    for c in 2:k
        total = 0.0
        @inbounds for i in 1:m
            if !in_chosen[i]
                total += best2[i]
            end
        end
        pick = 0
        if total > 0.0 && isfinite(total)
            r = rand(rng) * total
            acc = 0.0
            @inbounds for i in 1:m
                if !in_chosen[i]
                    acc += best2[i]
                    if acc >= r
                        pick = i
                        break
                    end
                end
            end
            if pick == 0
                # Float round-off safety net (accumulated `acc < r`): take
                # the last unchosen index.
                @inbounds for i in m:-1:1
                    if !in_chosen[i]
                        pick = i
                        break
                    end
                end
            end
        else
            # Every remaining point coincides with the chosen set: draw
            # uniformly among the unchosen indices.
            nleft = 0
            @inbounds for i in 1:m
                if !in_chosen[i]
                    nleft += 1
                end
            end
            draw = rand(rng, 1:nleft)
            @inbounds for i in 1:m
                if !in_chosen[i]
                    draw -= 1
                    if draw == 0
                        pick = i
                        break
                    end
                end
            end
        end
        chosen[c] = pick
        in_chosen[pick] = true
        @inbounds for i in 1:m
            dd = _spectral_sqdist(X, i, X, pick, d)
            if dd < best2[i]
                best2[i] = dd
            end
        end
    end
    return chosen
end

"""
    _spectral_kmeans(X::Matrix{Float64}, k::Int, rng::Random.AbstractRNG; max_iter=$(_SPECTRAL_MAX_ITER)) -> (assign, centers, niter)

Lloyd k-means on the rows of `X` with squared Euclidean distances and
k-means++ seeding ([`_spectral_plusplus_seeds`](@ref), all randomness via
`rng`): repeat nearest-center assignment (distance ties → lowest cluster id)
and mean recomputation until assignments stop changing (hard cap `max_iter`
iterations, `≥ 1` required). Empty clusters are allowed: their label goes
unused and their center is kept. Returns the per-row assignments
(`1:k`), the `k × size(X, 2)` center matrix, and the iteration count.
Requires `1 ≤ k ≤ size(X, 1)`, else `ArgumentError`.
"""
function _spectral_kmeans(X::Matrix{Float64}, k::Int, rng::Random.AbstractRNG; max_iter::Int = _SPECTRAL_MAX_ITER)
    m = size(X, 1)
    d = size(X, 2)
    1 <= k <= m || throw(
        ArgumentError("_spectral_kmeans requires 1 ≤ k ≤ $m (rows of X), got k = $k"),
    )
    max_iter >= 1 || throw(ArgumentError("_spectral_kmeans requires max_iter ≥ 1, got $max_iter"))
    seeds = _spectral_plusplus_seeds(X, k, rng)
    centers = Matrix{Float64}(undef, k, d)
    @inbounds for c in 1:k
        for t in 1:d
            centers[c, t] = X[seeds[c], t]
        end
    end
    old_centers = similar(centers)
    counts = Vector{Int}(undef, k)
    assign = zeros(Int, m)
    niter = 0
    for iter in 1:max_iter
        changed = false
        @inbounds for i in 1:m
            best = 1
            best_d = _spectral_sqdist(X, i, centers, 1, d)
            for c in 2:k
                dd = _spectral_sqdist(X, i, centers, c, d)
                if dd < best_d
                    best_d = dd
                    best = c
                end
            end
            if assign[i] != best
                assign[i] = best
                changed = true
            end
        end
        niter = iter
        copy!(old_centers, centers)
        fill!(centers, 0.0)
        fill!(counts, 0)
        @inbounds for i in 1:m
            c = assign[i]
            counts[c] += 1
            for t in 1:d
                centers[c, t] += X[i, t]
            end
        end
        @inbounds for c in 1:k
            if counts[c] > 0
                inv = 1.0 / counts[c]
                for t in 1:d
                    centers[c, t] *= inv
                end
            else
                # Empty cluster: the label goes unused; keep the old center.
                for t in 1:d
                    centers[c, t] = old_centers[c, t]
                end
            end
        end
        changed || break
    end
    return assign, centers, niter
end

"""
    _spectral_partition(g::MRIOGraph, ncommunities::Union{Nothing,Int}, rng::Random.AbstractRNG) -> Vector{Int32}

Spectral partition of `g`, always on the symmetrized
projection of the effective weights: `A[i, j] = sym_weight(W, i, j, cutoff,
self_loops)` for both directed and undirected graphs (for undirected graphs
this is just `g`'s effective weights). A directed `g` is therefore
partitioned via its symmetrized projection; `result.modularity` is still
scored on `g` as built by [`communities`](@ref).

Pipeline: (1) size gate — `n > 5000` throws an `ArgumentError` (O(n²)
workspace plus O(n³) eigen decomposition) directing users to
`aggregate(mrio, [:CountryCode])` first or to `algorithm = :louvain` /
`:leiden`; (2) stream the strengths `d_i = Σ_j A[i, j]` (diagonal counted
once) on the fly; isolated nodes (`d_i == 0`) are excluded from the eigen
problem and each becomes its own singleton community; (3) build the symmetric
normalized Laplacian `L_sym = I − D^(−1/2) A D^(−1/2)` (`Float64` dense, on
the non-isolated nodes) and decompose it with
`LinearAlgebra.eigen(Symmetric(L_sym))` (ascending eigenvalues); (4) cluster
count — `ncommunities === nothing` selects `k` by eigengap
([`_spectral_auto_k`](@ref)), else `k = min(ncommunities, n′)`; (5)
Ng–Jordan–Weiss embedding: the `k` eigenvectors for the `k` smallest
eigenvalues (including the leading one), each row normalized to unit length
(rows with norm `< 1.0e-12` stay zero and their nodes become singletons);
(6) [`_spectral_kmeans`](@ref) on the normalized rows (k-means++ seeding via
`rng`, Lloyd iterations, ties → lowest cluster id, empty clusters allowed so
their label goes unused).

Edge cases: `n == 1` returns `Int32[1]` without any eigen work; an all-zero
matrix returns singletons `1:n`; `k == 1` returns one community (k-means
skipped); `k ≥ m` (requested at least as many communities as embedded rows)
gives each embedded node its own cluster. Returns arbitrary positive `Int32`
labels (`communities` relabels via `_relabel_first_appearance`).

Cost: two O(n²) streaming passes over the wrapped matrix plus one dense
eigen decomposition (O(n³) time, O(n²) memory); the wrapped matrix is never
converted (accumulation/decomposition in `Float64`).
"""
function _spectral_partition(g::MRIOGraph, ncommunities::Union{Nothing, Int}, rng::Random.AbstractRNG)
    W = g.weights
    n = size(W, 1)
    _spectral_check_size(n)
    n == 1 && return Int32[1]
    f = g.filter
    cutoff = weight_cutoff(f)
    self_loops = f[3]
    strengths = zeros(Float64, n)
    symmetric_strengths!(strengths, W, f)
    noniso = Int[i for i in 1:n if strengths[i] != 0.0]
    nprime = length(noniso)
    nprime == 0 && return Int32.(1:n)
    inv_sqrt = Vector{Float64}(undef, nprime)
    @inbounds for (pos, v) in enumerate(noniso)
        inv_sqrt[pos] = 1.0 / sqrt(strengths[v])
    end
    laplacian = Matrix{Float64}(undef, nprime, nprime)
    @inbounds for jj in 1:nprime
        j = noniso[jj]
        sj = inv_sqrt[jj]
        for ii in 1:nprime
            i = noniso[ii]
            a = Float64(sym_weight(W, i, j, cutoff, self_loops))
            laplacian[ii, jj] = (ii == jj ? 1.0 : 0.0) - a * inv_sqrt[ii] * sj
        end
    end
    decomposition = eigen(Symmetric(laplacian))
    lambdas = decomposition.values
    vectors = decomposition.vectors
    k = ncommunities === nothing ? _spectral_auto_k(lambdas, nprime) : clamp(Int(ncommunities), 1, nprime)
    # Ng–Jordan–Weiss embedding: first k eigenvectors, rows to unit length.
    embedding = Matrix{Float64}(undef, nprime, k)
    is_singleton = falses(nprime)
    @inbounds for ii in 1:nprime
        norm2 = 0.0
        for j in 1:k
            norm2 += vectors[ii, j]^2
        end
        norm = sqrt(norm2)
        if norm < _SPECTRAL_ZERO_ROW_TOL
            is_singleton[ii] = true
            for j in 1:k
                embedding[ii, j] = 0.0
            end
        else
            for j in 1:k
                embedding[ii, j] = vectors[ii, j] / norm
            end
        end
    end
    raw = Vector{Int32}(undef, n)
    # Isolated nodes (excluded from the eigen problem) each get a fresh
    # singleton label after the k-means range.
    fresh = k + 1
    @inbounds for i in 1:n
        if strengths[i] == 0.0
            raw[i] = Int32(fresh)
            fresh += 1
        end
    end
    if k == 1
        # One community for all embedded nodes; k-means is skipped.
        @inbounds for (pos, v) in enumerate(noniso)
            if is_singleton[pos]
                raw[v] = Int32(fresh)
                fresh += 1
            else
                raw[v] = Int32(1)
            end
        end
    else
        keep = [pos for pos in 1:nprime if !is_singleton[pos]]
        m = length(keep)
        if m == 0
            @inbounds for (pos, v) in enumerate(noniso)
                raw[v] = Int32(fresh)
                fresh += 1
            end
        elseif k >= m
            # At least as many communities requested as embedded rows: each
            # embedded node is its own cluster (k-means skipped).
            label = 1
            @inbounds for pos in keep
                raw[noniso[pos]] = Int32(label)
                label += 1
            end
            @inbounds for (pos, v) in enumerate(noniso)
                if is_singleton[pos]
                    raw[v] = Int32(fresh)
                    fresh += 1
                end
            end
        else
            points = Matrix{Float64}(undef, m, k)
            @inbounds for (r, pos) in enumerate(keep)
                for j in 1:k
                    points[r, j] = embedding[pos, j]
                end
            end
            assign, _, _ = _spectral_kmeans(points, k, rng)
            @inbounds for (r, pos) in enumerate(keep)
                raw[noniso[pos]] = Int32(assign[r])
            end
            @inbounds for (pos, v) in enumerate(noniso)
                if is_singleton[pos]
                    raw[v] = Int32(fresh)
                    fresh += 1
                end
            end
        end
    end
    return raw
end
