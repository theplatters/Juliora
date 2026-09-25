using Test
using Juliora
using Graphs
using DataFrames
using LinearAlgebra
using SparseArrays
using Random
using Statistics

# Naive ground-truth references for the MRIOGraph semantics contract.
# These use plain loops over the raw matrix only and never touch the
# implementation's kernels.
function gref_tau(W::AbstractMatrix, threshold::Real, min_share::Real)
    return max(Float64(threshold), Float64(min_share) * sum(abs, W))
end

function gref_directed_w(W::AbstractMatrix, i::Int, j::Int, tau::Float64, self_loops::Bool)
    if i == j && !self_loops
        return 0.0
    end
    x = Float64(W[i, j])
    return abs(x) < tau ? 0.0 : x
end

function gref_undirected_w(W::AbstractMatrix, i::Int, j::Int, tau::Float64, self_loops::Bool)
    if i == j
        if !self_loops
            return 0.0
        end
        x = Float64(W[i, i])
        return abs(x) < tau ? 0.0 : x
    end
    x = Float64(W[i, j]) + Float64(W[j, i])
    return abs(x) < tau ? 0.0 : x
end

function gref_edge_set(W::AbstractMatrix, directed::Bool, tau::Float64, self_loops::Bool)
    n = size(W, 1)
    wfun = directed ? gref_directed_w : gref_undirected_w
    edges = Set{Tuple{Int, Int}}()
    if directed
        for i in 1:n, j in 1:n
            wfun(W, i, j, tau, self_loops) != 0.0 && push!(edges, (i, j))
        end
    else
        for i in 1:n, j in i:n
            wfun(W, i, j, tau, self_loops) != 0.0 && push!(edges, (i, j))
        end
    end
    return edges
end

function gref_out_neighbors(W::AbstractMatrix, v::Int, directed::Bool, tau::Float64, self_loops::Bool)
    n = size(W, 1)
    wfun = directed ? gref_directed_w : gref_undirected_w
    return Set(j for j in 1:n if wfun(W, v, j, tau, self_loops) != 0.0)
end

function gref_in_neighbors(W::AbstractMatrix, v::Int, directed::Bool, tau::Float64, self_loops::Bool)
    n = size(W, 1)
    if directed
        return Set(i for i in 1:n if gref_directed_w(W, i, v, tau, self_loops) != 0.0)
    else
        # Undirected graphs are symmetric: inneighbors == outneighbors.
        return Set(i for i in 1:n if gref_undirected_w(W, v, i, tau, self_loops) != 0.0)
    end
end

function gref_nodes(n::Int)
    return DataFrame(CountryCode = string.("C", 1:n), Sector = fill("s", n))
end

function gref_pagerank(W::AbstractMatrix, directed::Bool, tau::Float64, self_loops::Bool, alpha::Float64; tol = 1.0e-14, max_iter = 100_000)
    n = size(W, 1)
    wfun = directed ? gref_directed_w : gref_undirected_w
    s = zeros(n)
    for j in 1:n, i in 1:n
        s[i] += wfun(W, i, j, tau, self_loops)
    end
    r = fill(1.0 / n, n)
    for _ in 1:max_iter
        dangling = 0.0
        for i in 1:n
            s[i] == 0.0 && (dangling += r[i])
        end
        rnew = zeros(n)
        for j in 1:n, i in 1:n
            if s[i] != 0.0
                rnew[j] += wfun(W, i, j, tau, self_loops) * r[i] / s[i]
            end
        end
        err = 0.0
        for j in 1:n
            rnew[j] = alpha * rnew[j] + (1.0 - alpha + alpha * dangling) / n
            err += abs(rnew[j] - r[j])
        end
        r = rnew
        err < n * tol && return r
    end
    error("gref_pagerank did not converge")
end

gref_pagerank_alloc(g) = @allocated pagerank_scores(g)

gref_tvec_alloc(op, y, W, f, x, inv_s) = @allocated op(y, W, f, x, inv_s)

# Function-barrier drain: counts the edges of `g` via the public iterator
# protocol. Kept at top level so `@allocated` measures the iterator itself
# rather than testset-scope variable capture.
function gref_drain_edges(g)
    n = 0
    for e in Graphs.edges(g)
        n += 1
    end
    return n
end

# Small dense fixtures: asymmetric, scattered zeros, nonzero diagonals.
const GRAPH_W3 = [0.0 2.0 -0.5; 1.0 4.0 0.0; 0.0 3.0 1.5]
const GRAPH_W5 = [
    2.0 0.0 1.0 0.0 -1.0;
    0.5 3.0 0.0 2.0 0.0;
    0.0 -2.0 1.5 0.0 4.0;
    1.0 0.0 0.0 2.5 0.0;
    0.0 0.0 -0.5 1.5 3.5
]

const GRAPH_PR3 = [0.0 2.0 2.0; 0.0 0.0 1.0; 0.0 0.0 0.0]
const GRAPH_PR4 = [0.5 1.0 0.0 2.0; 0.0 0.0 0.0 0.0; 1.0 0.0 0.5 0.5; 0.0 3.0 0.0 0.0]
const GRAPH_PR5 = [0.0 4.0 0.0 0.0 1.0; 0.0 0.0 2.0 0.0 0.0; 0.0 0.0 0.0 3.0 0.0; 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]

# Shared planted fixture (milestone 3 communities): two strong 3-node blocks
# with all off-diagonal entries 10.0 plus a single weak bridge Z[1, 4] = 1.0.
const GRAPH_PLANT_Z = [
    0.0 10.0 10.0 1.0 0.0 0.0;
    10.0 0.0 10.0 0.0 0.0 0.0;
    10.0 10.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 10.0 10.0;
    0.0 0.0 0.0 10.0 0.0 10.0;
    0.0 0.0 0.0 10.0 10.0 0.0
]
const GRAPH_PLANT_MEMB = Int32[1, 1, 1, 2, 2, 2]

function gref_plant_mrio()
    idx = DataFrame(
        CountryCode = ["A", "A", "A", "B", "B", "B"],
        Sector = ["s1", "s2", "s3", "s1", "s2", "s3"],
    )
    return MRIO(
        Z = MatrixEntry(copy(GRAPH_PLANT_Z), idx, idx),
        Y = MatrixEntry(
            reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 6, 1),
            DataFrame(Category = ["Final"]),
            idx,
        ),
        VA = MatrixEntry(fill(1.0, 2, 6), idx, DataFrame(Category = ["Comp", "Tax"])),
    )
end

function gref_modularity(
        W::AbstractMatrix,
        directed::Bool,
        tau::Float64,
        self_loops::Bool,
        membership::AbstractVector{<:Integer},
        γ::Real,
    )
    n = size(W, 1)
    wfun = directed ? gref_directed_w : gref_undirected_w
    A = [wfun(W, i, j, tau, self_loops) for i in 1:n, j in 1:n]
    m = sum(A)
    m == 0.0 && return 0.0
    Q = 0.0
    for c in unique(membership)
        members = [i for i in 1:n if membership[i] == c]
        e = sum(A[i, j] for i in members, j in members)
        if directed
            kout = sum(A[i, j] for i in members, j in 1:n)
            kin = sum(A[i, j] for i in 1:n, j in members)
            Q += e - γ * kout * kin / m
        else
            k = sum(A[i, j] for i in members, j in 1:n)
            Q += e - γ * k^2 / m
        end
    end
    return Q / m
end

function gref_q_from_P(P::AbstractMatrix, membership::AbstractVector{<:Integer}, γ::Real)
    # Naive modularity straight from the pinned Q definition on an explicit
    # pair-weight matrix (diagonal counted once, no kernels involved).
    n = size(P, 1)
    m = sum(P)
    m == 0.0 && return 0.0
    Q = 0.0
    for c in unique(membership)
        members = [i for i in 1:n if membership[i] == c]
        e = sum(P[i, j] for i in members, j in members)
        kout = sum(P[i, j] for i in members, j in 1:n)
        kin = sum(P[i, j] for i in 1:n, j in members)
        Q += e - γ * kout * kin / m
    end
    return Q / m
end

# Same-partition comparison: equal iff every node pair agrees on co-membership
# (labels may be permuted).
function gref_same_partition(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || return false
    n = length(a)
    for i in 1:n, j in 1:n
        if (a[i] == a[j]) != (b[i] == b[j])
            return false
        end
    end
    return true
end

# Function-barrier allocation probe for one Louvain run. Kept at top level so
# `@allocated` measures the run itself rather than testset-scope capture.
function gref_louvain_alloc(g)
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    return @allocated ext._louvain_partition(g, 1.0, Random.MersenneTwister(1))
end

# Function-barrier allocation probe for one Leiden run (same contract as
# `gref_louvain_alloc`).
function gref_leiden_alloc(g)
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    return @allocated ext._leiden_partition(g, 1.0, Random.MersenneTwister(1))
end

# Function-barrier allocation probe for one `_partition_modularity` call.
# Kept at top level so `@allocated` measures the call itself rather than
# testset-scope variable capture.
function gref_mod_alloc(g, memb, γ)
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    return @allocated ext._partition_modularity(g, memb, γ)
end

# Ring-of-cliques toy (Leiden tests): 4 cliques of 5 nodes with strong
# intra-clique weights (5.0 off-diagonal) joined in a ring by single weak
# bridge edges (0.5). Non-negative, asymmetric only at the bridges.
function gref_ring_w()
    n = 20
    W = zeros(n, n)
    for b in 0:3, i in 1:5, j in 1:5
        i != j && (W[b * 5 + i, b * 5 + j] = 5.0)
    end
    W[5, 6] = 0.5
    W[10, 11] = 0.5
    W[15, 16] = 0.5
    W[20, 1] = 0.5
    return W
end

function gref_ring_memb()
    return Int32[v for b in 0:3 for v in fill(b + 1, 5)]
end

# Three disjoint bidirectional pairs (weight 3.0 each direction): the
# pair-wise partition is optimal with Q = 2/3 at γ = 1 (directed: m = 18,
# per community e = 6, K^out = K^in = 6, so Q = 3(6 - 36/18)/18 = 2/3;
# undirected: symmetrized pairs 6.0, m = 36, K = 12, Q = 3(12-144/36)/36).
function gref_pairs_w()
    W = zeros(6, 6)
    for (i, j) in ((1, 2), (3, 4), (5, 6))
        W[i, j] = 3.0
        W[j, i] = 3.0
    end
    return W
end

const GRAPH_PAIRS_MEMB = Int32[1, 1, 2, 2, 3, 3]

# Assert every community of `membership` induces a connected subgraph of the
# extracted topology of `g` (weak connectivity covers directed graphs).
function gref_assert_connected(membership::AbstractVector, g)
    simple, _ = to_simple_graph(g)
    for c in unique(membership)
        verts = Int32[i for i in eachindex(membership) if membership[i] == c]
        sub = Graphs.induced_subgraph(simple, verts)
        sg = sub isa Tuple ? sub[1] : sub
        @test length(Graphs.weakly_connected_components(sg)) == 1
    end
end

@testset "AbstractGraph conformance" begin
    # (threshold, min_share): no filter, absolute filter, scale-relative filter.
    filter_cfgs = [(0.0, 0.0), (1.5, 0.0), (0.0, 0.2)]
    for W in (GRAPH_W3, GRAPH_W5)
        n = size(W, 1)
        nodes = gref_nodes(n)
        for directed in (true, false)
            direction = directed ? :directed : :undirected
            for self_loops in (false, true)
                for (threshold, min_share) in filter_cfgs
                    tau = gref_tau(W, threshold, min_share)
                    wfun = directed ? gref_directed_w : gref_undirected_w
                    g = mrio_graph(
                        copy(W),
                        nodes;
                        direction = direction,
                        threshold = threshold,
                        min_share = min_share,
                        self_loops = self_loops,
                    )
                    ref_edges = gref_edge_set(W, directed, tau, self_loops)

                    @test Graphs.nv(g) == n
                    @test Graphs.ne(g) == length(ref_edges)
                    @test Graphs.is_directed(g) == directed
                    @test Graphs.is_directed(typeof(g)) == directed
                    @test Graphs.eltype(g) == Int32
                    @test Graphs.edgetype(g) == Graphs.SimpleEdge{Int32}
                    @test collect(Graphs.vertices(g)) == Int32.(1:n)

                    for i in 1:n, j in 1:n
                        expected = wfun(W, i, j, tau, self_loops) != 0.0
                        @test Graphs.has_edge(g, i, j) == expected
                        @test Graphs.weights(g)[i, j] == wfun(W, i, j, tau, self_loops)
                    end
                    for v in 1:n
                        @test Set(Int.(Graphs.outneighbors(g, v))) ==
                            gref_out_neighbors(W, v, directed, tau, self_loops)
                        @test Set(Int.(Graphs.inneighbors(g, v))) ==
                            gref_in_neighbors(W, v, directed, tau, self_loops)
                    end
                    got_edges = Set(
                        directed ? (Int(Graphs.src(e)), Int(Graphs.dst(e))) :
                            (min(Int(Graphs.src(e)), Int(Graphs.dst(e))), max(Int(Graphs.src(e)), Int(Graphs.dst(e)))) for e in Graphs.edges(
                                g,
                            )
                    )
                    @test got_edges == ref_edges
                end
            end
        end
    end

    # Ecosystem interop: a Graphs.jl generic dispatches on is_directed(::Type),
    # so this checks the directedness value-parameter encoding end to end.
    g_dir = mrio_graph(copy(GRAPH_W3), gref_nodes(3))
    g_und = mrio_graph(copy(GRAPH_W3), gref_nodes(3); direction = :undirected)
    @test isfinite(Graphs.modularity(g_dir, [1, 1, 2]))
    @test isfinite(Graphs.modularity(g_und, [1, 1, 2]))
end

@testset "Memory guards" begin
    # Zero-copy construction: identity with the caller's matrix.
    W = copy(GRAPH_W5)
    nodes = gref_nodes(5)
    g = mrio_graph(W, nodes)
    @test g.weights === W
    @test g.nodes === nodes

    # Element type is never converted.
    W32 = Float32.(GRAPH_W3)
    nodes3 = gref_nodes(3)
    g32 = mrio_graph(W32, nodes3)
    @test g32.weights isa Matrix{Float32}
    @test g32.weights === W32

    # MRIO-backed construction shares the selected matrix and node table.
    idx = DataFrame(CountryCode = ["A", "B", "C"], Sector = ["x", "x", "y"])
    zdata = [1.0 2.0 0.0; 0.5 3.0 1.0; 0.0 1.5 2.0]
    mrio = MRIO(
        Z = MatrixEntry(zdata, idx, idx),
        Y = MatrixEntry(reshape([1.0, 2.0, 3.0], 3, 1), DataFrame(CountryCode = ["A"], Sector = ["x"]), idx),
        VA = MatrixEntry(reshape([1.0, 2.0, 3.0], 1, 3), idx, DataFrame(VA = ["va"])),
    )
    gm = mrio_graph(mrio)
    @test gm.weights === mrio.Z.data
    @test gm.weights === mrio.T.data
    @test gm.nodes === mrio.Z.row_indices
    @test mrio_graph(mrio; source = :A).weights === mrio.A.data
    @test mrio_graph(mrio; weights = :technical).weights === mrio.A.data
    @test mrio_graph(mrio; source = :Z, weights = :flows).weights === mrio.T.data

    # Construction allocates no O(n^2) structure (generous bounds).
    Wbig = randn(MersenneTwister(42), 500, 500)
    nodesbig = gref_nodes(500)
    mrio_graph(Wbig, nodesbig) # warm-up (compilation)
    gbig = mrio_graph(Wbig, nodesbig)
    Graphs.ne(gbig) # warm-up (compilation)
    graph_summary(gbig) # warm-up (compilation)
    @test @allocated(mrio_graph(Wbig, nodesbig)) < 10_000
    @test @allocated(Graphs.ne(gbig)) < 10_000
    @test @allocated(graph_summary(gbig)) < 100_000

    # Edge iterator is O(1)-memory: iterating all ~90k edges of a dense
    # 300x300 matrix must not allocate per element (boxing each edge would
    # exceed 1 MB; the bound below is generous but far below that).
    Wdense = fill(2.0, 300, 300)
    nodesdense = gref_nodes(300)
    gdense = mrio_graph(copy(Wdense), nodesdense)
    @test gref_drain_edges(gdense) == 300 * 299
    gref_drain_edges(gdense) # warm-up (compilation)
    @test (@allocated gref_drain_edges(gdense)) < 100_000

    # Non-Matrix AbstractMatrix input is copied once via Matrix(W): the
    # stored weights equal the view but are a fresh Matrix, and the element
    # type is never converted.
    Wbase = copy(GRAPH_W5)
    nodes5 = gref_nodes(5)
    gview = mrio_graph(view(Wbase, :, :), nodes5)
    @test gview.weights isa Matrix{Float64}
    @test gview.weights == Wbase
    @test gview.weights !== Wbase
    W32base = Float32.(GRAPH_W3)
    g32view = mrio_graph(view(W32base, :, :), gref_nodes(3))
    @test g32view.weights isa Matrix{Float32}
    @test g32view.weights == W32base
end

@testset "On-the-fly filtering equals materialize-prune-then-run" begin
    # Toy matrix with borderline values: |w| == tau must survive (strict <).
    W = [0.0 2.0 -1.0 0.5; 1.0 3.0 0.0 -2.0; 0.0 0.5 2.0 1.0; -1.0 0.0 2.0 0.0]
    nodes = gref_nodes(4)
    filter_cfgs = [(0.0, 0.0, false), (1.0, 0.0, false), (1.0, 0.0, true), (0.5, 0.05, false), (0.5, 0.05, true)]
    for (threshold, min_share, self_loops) in filter_cfgs
        tau = gref_tau(W, threshold, min_share)
        g = mrio_graph(copy(W), nodes; threshold = threshold, min_share = min_share, self_loops = self_loops)

        # Directed reference: prune the raw matrix, then run unfiltered.
        Wp = copy(W)
        Wp[abs.(Wp) .< tau] .= 0.0
        if !self_loops
            for i in axes(Wp, 1)
                Wp[i, i] = 0.0
            end
        end
        g2 = mrio_graph(Wp, nodes; self_loops = self_loops)
        @test Graphs.ne(g) == Graphs.ne(g2)
        for i in 1:4, j in 1:4
            @test Graphs.weights(g)[i, j] == Graphs.weights(g2)[i, j]
        end
        for v in 1:4
            @test Set(Graphs.outneighbors(g, v)) == Set(Graphs.outneighbors(g2, v))
            @test Set(Graphs.inneighbors(g, v)) == Set(Graphs.inneighbors(g2, v))
        end

        # Undirected reference: symmetrize first (diagonal counted once),
        # prune on |S| < tau, run unfiltered.
        gu = mrio_graph(
            copy(W),
            nodes;
            direction = :undirected,
            threshold = threshold,
            min_share = min_share,
            self_loops = self_loops,
        )
        S = zeros(4, 4)
        for i in 1:4, j in 1:4
            S[i, j] = i == j ? (self_loops ? W[i, i] : 0.0) : W[i, j] + W[j, i]
        end
        Sp = copy(S)
        Sp[abs.(Sp) .< tau] .= 0.0
        # The pruned symmetrized matrix is the materialized equivalent of the
        # on-the-fly undirected view (it cannot round-trip through mrio_graph
        # itself, which would symmetrize a second time).
        for i in 1:4, j in 1:4
            @test Graphs.weights(gu)[i, j] == Sp[min(i, j), max(i, j)]
        end
        @test Graphs.ne(gu) == count(!iszero, [Sp[i, j] for i in 1:4 for j in i:4])
        for v in 1:4
            @test Set(Int.(Graphs.outneighbors(gu, v))) == gref_out_neighbors(W, v, false, tau, self_loops)
            @test Set(Int.(Graphs.inneighbors(gu, v))) == gref_in_neighbors(W, v, false, tau, self_loops)
        end
    end

    # Borderline values |w| == tau survive the strict-< filter.
    Wb = [0.0 1.0; -1.0 0.0]
    nodesb = gref_nodes(2)
    gb = mrio_graph(copy(Wb), nodesb; threshold = 1.0)
    @test Graphs.ne(gb) == 2
    @test Graphs.weights(gb)[1, 2] == 1.0
    @test Graphs.weights(gb)[2, 1] == -1.0
    gub = mrio_graph(copy(Wb), nodesb; direction = :undirected, threshold = 0.0)
    # 1.0 + -1.0 cancels to exactly zero: never an edge.
    @test !Graphs.has_edge(gub, 1, 2)
end

@testset "Tiled symmetric accumulation" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    @test ext !== nothing

    function gref_sym_matrix(W::AbstractMatrix, tau::Float64, self_loops::Bool)
        n = size(W, 1)
        S = zeros(n, n)
        for i in 1:n, j in 1:n
            S[i, j] = gref_undirected_w(W, i, j, tau, self_loops)
        end
        return S
    end

    rng = MersenneTwister(7)
    cases = [
        (randn(rng, 5, 5), 0.0, 0.0, false),
        (randn(rng, 5, 5), 0.8, 0.0, true),
        (randn(rng, 33, 33), 0.0, 0.02, false),
        (randn(rng, 33, 33), 1.0, 0.01, true),
    ]
    for (W, threshold, min_share, self_loops) in cases
        scale = min_share > 0 ? sum(abs, W) : 0.0
        f = (Float64(threshold), Float64(min_share), self_loops, Float64(scale))
        tau = gref_tau(W, threshold, min_share)
        S = gref_sym_matrix(W, tau, self_loops)
        s = zeros(size(W, 1))
        ext.symmetric_strengths!(s, W, f)
        @test s ≈ vec(sum(S, dims = 2))
    end

    # A 700x700 matrix with a tiny tile exercises multi-tile traversal.
    Wbig = randn(rng, 700, 700)
    threshold, min_share, self_loops = 0.5, 0.0, false
    scale = 0.0
    fbig = (Float64(threshold), Float64(min_share), self_loops, Float64(scale))
    taubig = gref_tau(Wbig, threshold, min_share)
    Sbig = gref_sym_matrix(Wbig, taubig, self_loops)
    sbig = zeros(700)
    ncalls = Ref(0)
    ext.symmetric_tile_accumulate!(
        (i, j, w) -> begin
            ncalls[] += 1
            sbig[i] += w
            if i != j
                sbig[j] += w
            end
        end,
        Wbig,
        fbig;
        tile_size = 8,
    )
    @test sbig ≈ vec(sum(Sbig, dims = 2))
    ref_ne = length(gref_edge_set(Wbig, false, taubig, self_loops))
    @test ncalls[] == ref_ne

    # Each unordered pair is visited exactly once; diagonals are reported
    # as op(i, i, w) when self_loops is true and never otherwise.
    Wsmall = [0.0 2.0 0.0; 1.0 4.0 3.0; 0.0 0.5 1.5]
    for self_loops in (false, true)
        fsmall = (0.0, 0.0, self_loops, 0.0)
        seen = Tuple{Int, Int}[]
        seen_w = Float64[]
        ext.symmetric_tile_accumulate!(
            (i, j, w) -> begin
                push!(seen, (i, j))
                push!(seen_w, Float64(w))
            end,
            Wsmall,
            fsmall;
            tile_size = 2,
        )
        tausmall = 0.0
        ref = gref_edge_set(Wsmall, false, tausmall, self_loops)
        @test Set(seen) == ref
        @test length(seen) == length(ref)
        for ((i, j), w) in zip(seen, seen_w)
            @test w == gref_undirected_w(Wsmall, i, j, tausmall, self_loops)
            @test i <= j
        end
        if self_loops
            @test (2, 2) in seen
            @test (3, 3) in seen
        else
            @test all(i != j for (i, j) in seen)
        end
    end
end

@testset "to_simple_graph" begin
    W = [0.0 2.0 -0.5 0.0; 1.0 4.0 0.0 3.0; 0.0 0.0 1.5 1.0; 0.0 2.0 0.0 0.0]
    nodes = gref_nodes(4)

    # Directed extraction matches the reference edge set and weights.
    g = mrio_graph(copy(W), nodes)
    sg, distmx = to_simple_graph(g)
    @test sg isa Graphs.SimpleDiGraph{Int32}
    @test distmx isa SparseArrays.SparseMatrixCSC{Float32}
    ref = gref_edge_set(W, true, 0.0, false)
    @test Set((Int(Graphs.src(e)), Int(Graphs.dst(e))) for e in Graphs.edges(sg)) == ref
    @test Graphs.ne(sg) == length(ref)
    for (i, j) in ref
        @test distmx[i, j] ≈ Float32(gref_directed_w(W, i, j, 0.0, false))
    end
    @test length(distmx.nzval) == length(ref)

    # Undirected extraction is symmetric; self-loops are dropped.
    Wsl = [5.0 2.0; 1.0 6.0]
    nodes2 = gref_nodes(2)
    gsl = mrio_graph(copy(Wsl), nodes2; direction = :undirected, self_loops = true)
    @test Graphs.has_edge(gsl, 1, 1)
    su, du = to_simple_graph(gsl)
    @test su isa Graphs.SimpleGraph{Int32}
    @test !Graphs.has_edge(su, 1, 1)
    @test !Graphs.has_edge(su, 2, 2)
    @test du[1, 1] == 0.0f0
    @test du[2, 2] == 0.0f0
    @test du[1, 2] == du[2, 1] == Float32(3.0)
    @test Graphs.ne(su) == 1

    gu = mrio_graph(copy(W), nodes; direction = :undirected)
    sgu, dgu = to_simple_graph(gu)
    @test sgu isa Graphs.SimpleGraph{Int32}
    refu = gref_edge_set(W, false, 0.0, false)
    @test Set(
        (min(Int(Graphs.src(e)), Int(Graphs.dst(e))), max(Int(Graphs.src(e)), Int(Graphs.dst(e)))) for e in Graphs.edges(
                sgu,
            )
    ) == refu
    for (i, j) in refu
        @test dgu[i, j] == dgu[j, i] == Float32(gref_undirected_w(W, i, j, 0.0, false))
    end

    # Extra threshold applies on top of the graph filter.
    gf = mrio_graph(copy(W), nodes; threshold = 0.5)
    sgf, dgf = to_simple_graph(gf; threshold = 2.0)
    ref_f = Set{Tuple{Int, Int}}()
    for i in 1:4, j in 1:4
        w = gref_directed_w(W, i, j, 0.5, false)
        if w != 0.0 && abs(w) >= 2.0
            push!(ref_f, (i, j))
        end
    end
    @test Set((Int(Graphs.src(e)), Int(Graphs.dst(e))) for e in Graphs.edges(sgf)) == ref_f
    for (i, j) in ref_f
        @test dgf[i, j] ≈ Float32(W[i, j])
    end

    # Directed topk keeps the k heaviest out-edges per source; ties kept.
    Wt = [0.0 5.0 4.0 4.0 1.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]
    nodest = gref_nodes(5)
    gt = mrio_graph(copy(Wt), nodest)
    sgt, dgt = to_simple_graph(gt; topk = 2)
    @test Set((Int(Graphs.src(e)), Int(Graphs.dst(e))) for e in Graphs.edges(sgt)) ==
        Set([(1, 2), (1, 3), (1, 4)])
    @test dgt[1, 5] == 0.0f0

    # Undirected topk keeps an edge if it is top-k for EITHER endpoint.
    Wu = zeros(4, 4)
    Wu[1, 2] = Wu[2, 1] = 1.0
    Wu[1, 3] = Wu[3, 1] = 1.0
    Wu[1, 4] = Wu[4, 1] = 10.0
    Wu[2, 3] = Wu[3, 2] = 9.0
    nodesu = gref_nodes(4)
    guu = mrio_graph(copy(Wu), nodesu; direction = :undirected)
    sguu, dguu = to_simple_graph(guu; topk = 1)
    gotu = Set(
        (min(Int(Graphs.src(e)), Int(Graphs.dst(e))), max(Int(Graphs.src(e)), Int(Graphs.dst(e)))) for e in Graphs.edges(
                sguu,
            )
    )
    # (1,4): top-1 for both 1 and 4; (2,3): top-1 for both 2 and 3;
    # (1,2),(1,3): below top-1 (10.0) for node 1 AND below top-1 for nodes 2,3 (9.0).
    @test gotu == Set([(1, 4), (2, 3)])
    # Either-endpoint rule: (1,2) survives topk=2 via node 2 (incident: 1.0, 9.0).
    sguu2, _ = to_simple_graph(guu; topk = 2)
    gotu2 = Set(
        (min(Int(Graphs.src(e)), Int(Graphs.dst(e))), max(Int(Graphs.src(e)), Int(Graphs.dst(e)))) for e in Graphs.edges(
                sguu2,
            )
    )
    @test (1, 2) in gotu2
    @test (1, 3) in gotu2

    # Directed extraction with self_loops=true: self-loops appear in the
    # SimpleDiGraph and on the distmx diagonal; without self-loops neither.
    Wdsl = [1.0 2.0; 3.0 4.0]
    nodesdsl = gref_nodes(2)
    gdsl = mrio_graph(copy(Wdsl), nodesdsl; self_loops = true)
    sgsl, dgsl = to_simple_graph(gdsl)
    @test Graphs.has_edge(sgsl, 1, 1)
    @test Graphs.has_edge(sgsl, 2, 2)
    @test dgsl[1, 1] == 1.0f0
    @test dgsl[2, 2] == 4.0f0
    gdsl0 = mrio_graph(copy(Wdsl), nodesdsl; self_loops = false)
    sgsl0, dgsl0 = to_simple_graph(gdsl0)
    @test !Graphs.has_edge(sgsl0, 1, 1)
    @test !Graphs.has_edge(sgsl0, 2, 2)
    @test dgsl0[1, 1] == 0.0f0
    @test dgsl0[2, 2] == 0.0f0

    # topk combined with the extra threshold: candidates are
    # threshold-filtered first, then per-node top-k ranks the survivors.
    # Row 1 out-edges: (1,2)=9, (1,3)=7, (1,4)=6, (1,5)=1; threshold 6.5
    # prunes (1,4) and (1,5) before ranking, so topk=1 keeps only (1,2).
    Wtk = zeros(5, 5)
    Wtk[1, 2] = 9.0
    Wtk[1, 3] = 7.0
    Wtk[1, 4] = 6.0
    Wtk[1, 5] = 1.0
    gtk = mrio_graph(copy(Wtk), gref_nodes(5))
    sgtk, dgtk = to_simple_graph(gtk; threshold = 6.5, topk = 1)
    @test Set((Int(Graphs.src(e)), Int(Graphs.dst(e))) for e in Graphs.edges(sgtk)) == Set([(1, 2)])
    @test dgtk[1, 2] == 9.0f0
    @test dgtk[1, 3] == 0.0f0
    @test dgtk[1, 4] == 0.0f0
    @test dgtk[1, 5] == 0.0f0
end

@testset "graph_summary" begin
    # Hand-computed toy: W = [4 1; 2 3], directed, no self-loops, tau = 1.5.
    # Candidates (1,2): |1| = 1, (2,1): |2| = 2 -> total 3, retained 2.
    W = [4.0 1.0; 2.0 3.0]
    nodes = gref_nodes(2)
    g = mrio_graph(copy(W), nodes; threshold = 1.5)
    df = graph_summary(g)
    @test only(df.nodes) == 2
    @test only(df.edges) == 1
    @test only(df.directed) == true
    @test only(df.threshold) == 1.5
    @test only(df.min_share) == 0.0
    @test only(df.self_loops) == false
    @test only(df.total_weight) == 3.0
    @test only(df.retained_weight) == 2.0
    @test only(df.retained_share) ≈ 2.0 / 3.0
    @test only(df.retained_share) < 1.0
    @test only(df.memory_bytes) == sizeof(W)

    # Unfiltered: everything retained, share 1.0.
    gu = mrio_graph(copy(W), nodes)
    dfu = graph_summary(gu)
    @test only(dfu.edges) == 2
    @test only(dfu.total_weight) == 3.0
    @test only(dfu.retained_weight) == 3.0
    @test only(dfu.retained_share) == 1.0
    @test only(dfu.memory_bytes) == sizeof(W)

    # Undirected hand-computed: raw pair (1,2) = 1 + 2 = 3, total 3.
    gund = mrio_graph(copy(W), nodes; direction = :undirected, threshold = 1.5)
    dfund = graph_summary(gund)
    @test only(dfund.edges) == 1
    @test only(dfund.directed) == false
    @test only(dfund.total_weight) == 3.0
    @test only(dfund.retained_weight) == 3.0
    @test only(dfund.retained_share) == 1.0

    # Undirected with a cutoff above the pair weight: retained 0, share 0.
    gund2 = mrio_graph(copy(W), nodes; direction = :undirected, threshold = 3.5)
    dfund2 = graph_summary(gund2)
    @test only(dfund2.edges) == 0
    @test only(dfund2.retained_weight) == 0.0
    @test only(dfund2.retained_share) == 0.0

    # Self-loops join the candidate set when enabled.
    gsl = mrio_graph(copy(W), nodes; self_loops = true)
    dfsl = graph_summary(gsl)
    @test only(dfsl.edges) == 4
    @test only(dfsl.total_weight) == 4.0 + 1.0 + 2.0 + 3.0
    @test only(dfsl.retained_weight) == 4.0 + 1.0 + 2.0 + 3.0

    # All-zero matrix: total 0 and retained_share 1.0 by definition.
    W0 = zeros(2, 2)
    nodes0 = gref_nodes(2)
    g0 = mrio_graph(copy(W0), nodes0)
    df0 = graph_summary(g0)
    @test only(df0.edges) == 0
    @test only(df0.total_weight) == 0.0
    @test only(df0.retained_weight) == 0.0
    @test only(df0.retained_share) == 1.0

    # Undirected with self-loops: hand-computed total counts the diagonal
    # ONCE (|W[i,i]|) and off-diagonals as |W[i,j] + W[j,i]|.
    # W = [4 1; 2 3] -> |4| + |3| + |1 + 2| = 10 (not 2*(4+3)+3 = 17).
    gundsl = mrio_graph(copy(W), nodes; direction = :undirected, self_loops = true)
    dfundsl = graph_summary(gundsl)
    @test only(dfundsl.edges) == 3
    @test only(dfundsl.total_weight) == 10.0
    @test only(dfundsl.retained_weight) == 10.0
    @test only(dfundsl.retained_share) == 1.0
end

@testset "Validation and stub fallback" begin
    nodes3 = gref_nodes(3)
    idx = DataFrame(CountryCode = ["A", "B"], Sector = ["x", "x"])
    mrio = MRIO(
        Z = MatrixEntry([0.0 2.0; 1.0 0.0], idx, idx),
        Y = MatrixEntry(reshape([1.0, 1.0], 2, 1), DataFrame(CountryCode = ["A"], Sector = ["x"]), idx),
        VA = MatrixEntry(reshape([1.0, 1.0], 1, 2), idx, DataFrame(VA = ["va"])),
    )

    @test_throws DimensionMismatch mrio_graph(randn(3, 4), gref_nodes(3))
    @test_throws DimensionMismatch mrio_graph(randn(3, 3), gref_nodes(4))
    @test_throws DimensionMismatch mrio_graph(randn(2, 3), nodes3)

    @test_throws ArgumentError mrio_graph(mrio; source = :X)
    @test_throws ArgumentError mrio_graph(mrio; weights = :flows2)
    @test_throws ArgumentError mrio_graph(mrio; source = :A, weights = :flows)
    @test_throws ArgumentError mrio_graph(mrio; source = :Z, weights = :technical)
    @test_throws ArgumentError mrio_graph(randn(3, 3), nodes3; direction = :sideways)
    @test_throws ArgumentError mrio_graph(randn(3, 3), nodes3; threshold = -1.0)
    @test_throws ArgumentError mrio_graph(randn(3, 3), nodes3; min_share = 1.5)
    @test_throws ArgumentError mrio_graph(randn(3, 3), nodes3; min_share = -0.1)

    g = mrio_graph(copy(GRAPH_W3), nodes3)
    @test_throws ArgumentError to_simple_graph(g; threshold = -1.0)
    @test_throws ArgumentError to_simple_graph(g; topk = 0)
    @test_throws ArgumentError to_simple_graph(g; topk = -3)

    # Extension loaded but no method matches: the vararg stub fires.
    err = try
        mrio_graph(1)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("using Graphs", sprint(showerror, err))

    # Every stub entry point reports the missing-extension error on bogus input.
    for f in (
            communities,
            community_table,
            community_summary,
            pagerank_scores,
            node_similarity,
            similarity_graph,
            compare_networks,
            compare_partitions,
            graph_summary,
            to_simple_graph,
            mrio_graph,
        )
        ferr = try
            f(42)
            nothing
        catch e
            e
        end
        @test ferr isa ErrorException
        @test occursin("using Graphs", sprint(showerror, ferr))
    end
end

@testset "PageRank" begin
    # A. Hand-computed values (3-node graph with dangling node).
    g3 = mrio_graph(copy(GRAPH_PR3), gref_nodes(3))
    ref_pr3 = [8 / 33, 10 / 33, 5 / 11]
    @test pagerank_scores(g3; damping = 0.5, tol = 1e-14, max_iter = 100_000).data ≈ ref_pr3 rtol = 1e-10
    @test pagerank_scores(g3; weighted = false, damping = 0.5, tol = 1e-14, max_iter = 100_000).data ≈ ref_pr3 rtol =
        1e-10

    # B. Weighted conformance vs gref_pagerank + C. invariants.
    Wr = rand(MersenneTwister(5), 4, 4)
    Wr[2, :] .= 0.0
    pr_cases = Any[GRAPH_PR3, GRAPH_PR4, GRAPH_PR5, zeros(3, 3), Wr]
    filter_cfgs_pr = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.2)]
    for W in pr_cases
        n = size(W, 1)
        nodes = gref_nodes(n)
        for directed in (true, false)
            direction = directed ? :directed : :undirected
            for self_loops in (false, true)
                for (threshold, min_share) in filter_cfgs_pr
                    tau = gref_tau(W, threshold, min_share)
                    for alpha in (0.5, 0.85)
                        g = mrio_graph(
                            copy(W),
                            nodes;
                            direction = direction,
                            threshold = threshold,
                            min_share = min_share,
                            self_loops = self_loops,
                        )
                        res = pagerank_scores(g; damping = alpha, tol = 1e-14, max_iter = 100_000)
                        @test res.data ≈ gref_pagerank(W, directed, tau, self_loops, alpha) atol = 1e-10
                        @test sum(res.data) ≈ 1.0 rtol = 1e-12
                        @test all(res.data .>= 0)
                        @test length(res.data) == size(W, 1)
                        @test eltype(res.data) == Float64
                        @test res.col_indices === nodes
                        if W == zeros(3, 3)
                            @test res.data ≈ [1 / 3, 1 / 3, 1 / 3] atol = 1e-12
                        end
                    end
                end
            end
        end
    end

    # D. Cross-check against Graphs.pagerank (dangling convention).
    W1 = Float64.(GRAPH_PR3 .!= 0)
    g1 = mrio_graph(copy(W1), gref_nodes(3))
    sg1, _ = to_simple_graph(g1)
    @test pagerank_scores(g1; damping = 0.85, tol = 1e-14, max_iter = 100_000).data ≈
        Graphs.pagerank(sg1, 0.85, 100_000, 1e-14) atol = 1e-10

    # B/C. Float32 input: the element type is never converted, scores stay Float64.
    g32 = mrio_graph(Float32.(GRAPH_PR3), gref_nodes(3))
    @test eltype(g32.weights) == Float32
    res32 = pagerank_scores(g32; damping = 0.5, tol = 1e-14, max_iter = 100_000)
    @test res32.data ≈ [8 / 33, 10 / 33, 5 / 11] rtol = 1e-8
    @test eltype(res32.data) == Float64

    # B. Single-node graphs all return [1.0].
    @test pagerank_scores(mrio_graph(zeros(1, 1), gref_nodes(1))).data ≈ [1.0] atol = 1e-12
    # Self-flow dropped without self_loops: still dangling, hence uniform.
    @test pagerank_scores(mrio_graph(reshape([2.0], 1, 1), gref_nodes(1))).data ≈ [1.0] atol = 1e-12
    # Self-edge: s = 2, r' = α·(2·1·(1/2)) + (1-α) = 1.
    @test pagerank_scores(mrio_graph(reshape([2.0], 1, 1), gref_nodes(1); self_loops = true)).data ≈ [1.0] atol =
        1e-12

    # D. Undirected cross-check vs Graphs.pagerank. Each unordered pair has
    # exactly one nonzero raw entry, so the symmetrized weights are exactly 1
    # per edge; node 5 is isolated (dangling).
    Wut = [0.0 1.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0]
    gu = mrio_graph(copy(Wut), gref_nodes(5); direction = :undirected)
    @test pagerank_scores(gu; damping = 0.85, tol = 1e-14, max_iter = 100_000).data ≈
        Graphs.pagerank(to_simple_graph(gu)[1], 0.85, 100_000, 1e-14) atol = 1e-10

    # E. Delegation identity (weighted=false).
    g4 = mrio_graph(copy(GRAPH_PR4), gref_nodes(4))
    sg4, _ = to_simple_graph(g4)
    @test pagerank_scores(g4; weighted = false, damping = 0.85, tol = 1e-12, max_iter = 5000).data ==
        Graphs.pagerank(sg4, 0.85, 5000, 1e-12)
    g5u = mrio_graph(copy(GRAPH_PR5), gref_nodes(5); direction = :undirected)
    sg5u, _ = to_simple_graph(g5u)
    @test pagerank_scores(g5u; weighted = false, damping = 0.85, tol = 1e-12, max_iter = 5000).data ==
        Graphs.pagerank(sg5u, 0.85, 5000, 1e-12)

    # F. MRIO method.
    idx = DataFrame(CountryCode = ["A", "B", "C"], Sector = ["x", "x", "y"])
    zdata = [0.0 2.0 2.0; 0.0 0.0 1.0; 0.0 0.0 0.0]
    mrio = MRIO(
        Z = MatrixEntry(zdata, idx, idx),
        Y = MatrixEntry(reshape([1.0, 2.0, 3.0], 3, 1), DataFrame(CountryCode = ["A"], Sector = ["x"]), idx),
        VA = MatrixEntry(reshape([1.0, 2.0, 3.0], 1, 3), idx, DataFrame(VA = ["va"])),
    )
    @test pagerank_scores(mrio; damping = 0.5, tol = 1e-14, max_iter = 100_000).data ≈
        pagerank_scores(mrio_graph(mrio); damping = 0.5, tol = 1e-14, max_iter = 100_000).data
    @test pagerank_scores(mrio).col_indices === mrio.T.row_indices
    @test pagerank_scores(mrio; threshold = 1.0, damping = 0.85, tol = 1e-14, max_iter = 100_000).data ≈
        gref_pagerank(zdata, true, 1.0, false, 0.85) atol = 1e-10
    @test pagerank_scores(mrio; weights = :technical).data ≈
        pagerank_scores(mrio_graph(mrio; weights = :technical)).data

    # G. Validation and error paths.
    @test_throws ArgumentError pagerank_scores(g3; damping = 0.0)
    @test_throws ArgumentError pagerank_scores(g3; damping = 1.0)
    @test_throws ArgumentError pagerank_scores(g3; damping = -0.1)
    @test_throws ArgumentError pagerank_scores(g3; damping = 1.5)
    @test_throws ArgumentError pagerank_scores(g3; tol = 0.0)
    @test_throws ArgumentError pagerank_scores(g3; tol = -1.0)
    @test_throws ArgumentError pagerank_scores(g3; max_iter = 0)
    @test_throws ArgumentError begin
        g0 = mrio_graph(zeros(0, 0), DataFrame(CountryCode = String[], Sector = String[]))
        pagerank_scores(g0)
    end
    # Validation runs before the weighted/unweighted branch.
    g0uw = mrio_graph(zeros(0, 0), DataFrame(CountryCode = String[], Sector = String[]))
    @test_throws ArgumentError pagerank_scores(g0uw; weighted = false)
    gneg = mrio_graph([-1.0 -1.0; 0.0 0.0], gref_nodes(2))
    @test_throws ArgumentError pagerank_scores(gneg)
    # G. The error message names the offending pair.
    gneg_err = try
        pagerank_scores(gneg)
        nothing
    catch e
        e
    end
    @test gneg_err isa ArgumentError
    @test occursin("w(1, 2)", sprint(showerror, gneg_err))
    @test occursin("finite non-negative", sprint(showerror, gneg_err))
    # G. NaN/Inf effective weights are rejected (not "did not converge").
    # (NaN sits at the retained off-diagonal (1, 2): a diagonal NaN with the
    # default self_loops=false is dropped by the self-loop rule everywhere.)
    @test_throws ArgumentError pagerank_scores(mrio_graph([0.0 NaN; 0.0 0.0], gref_nodes(2)))
    @test_throws ArgumentError pagerank_scores(mrio_graph([0.0 Inf; 0.0 0.0], gref_nodes(2)))
    # Undirected w(1, 2) = Inf + -Inf = NaN is retained and must throw.
    @test_throws ArgumentError pagerank_scores(
        mrio_graph([Inf -Inf; 0.0 0.0], gref_nodes(2); direction = :undirected),
    )
    # G. Undirected sym-sum and retained diagonal negatives throw.
    @test_throws ArgumentError pagerank_scores(
        mrio_graph([0.0 -5.0; 0.0 0.0], gref_nodes(2); direction = :undirected),
    )
    @test_throws ArgumentError pagerank_scores(
        mrio_graph([-1.0 0.0; 0.0 0.0], gref_nodes(2); self_loops = true),
    )
    @test length(pagerank_scores(gneg; weighted = false).data) == 2
    @test pagerank_scores(mrio_graph([-1.0 -1.0; 0.0 0.0], gref_nodes(2); threshold = 2.0)).data ≈ [0.5, 0.5]
    @test_throws ErrorException pagerank_scores(g3; max_iter = 1, tol = 1e-15)
    @test_throws ErrorException pagerank_scores(g3; weighted = false, max_iter = 1, tol = 1e-15)

    # H. Memory guard.
    Wbig = rand(MersenneTwister(11), 500, 500)
    gbig = mrio_graph(Wbig, gref_nodes(500))
    gref_pagerank_alloc(gbig) # warm-up (compilation)
    @test gref_pagerank_alloc(gbig) < 500_000

    # H. Kernel zero-allocation barriers: the iteration allocates nothing per
    # kernel call (the whole-call guard above cannot see this because the
    # SeriesEntry lookup Dict dominates it).
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    gk = mrio_graph(rand(MersenneTwister(21), 250, 250), gref_nodes(250))
    yk, xk, sk = rand(250), rand(250), rand(250)
    ext.filtered_tvec!(yk, gk.weights, gk.filter, xk, sk) # warm-up (compilation)
    ext.symmetric_tvec!(yk, gk.weights, gk.filter, xk, sk) # warm-up (compilation)
    @test gref_tvec_alloc(ext.filtered_tvec!, yk, gk.weights, gk.filter, xk, sk) < 256
    @test gref_tvec_alloc(ext.symmetric_tvec!, yk, gk.weights, gk.filter, xk, sk) < 256
end

@testset "Planted fixture" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    # The planted partition beats the all-in-one partition and every 5/1
    # split (enumerated), on both directednesses without self-loops.
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        g = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6); direction = direction)
        q_plant = ext._partition_modularity(g, GRAPH_PLANT_MEMB, 1.0)
        @test q_plant ≈ gref_modularity(GRAPH_PLANT_Z, directed, 0.0, false, GRAPH_PLANT_MEMB, 1.0)
        @test q_plant > ext._partition_modularity(g, ones(Int32, 6), 1.0)
        for s in 1:6
            split = ones(Int32, 6)
            split[s] = Int32(2)
            @test q_plant > ext._partition_modularity(g, split, 1.0)
        end
    end
    # The MRIO builder shares the Z matrix and its row metadata by reference.
    mrio = gref_plant_mrio()
    @test mrio.T.data == GRAPH_PLANT_Z
    gm = mrio_graph(mrio)
    @test gm.weights === mrio.T.data
    @test gm.nodes === mrio.T.row_indices
end

@testset "Modularity" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    mats = (GRAPH_W3, GRAPH_W5, GRAPH_PR4, GRAPH_PLANT_Z)
    for W in mats
        n = size(W, 1)
        for directed in (true, false)
            direction = directed ? :directed : :undirected
            for self_loops in (false, true)
                for (threshold, min_share) in ((0.0, 0.0), (1.5, 0.0))
                    tau = gref_tau(W, threshold, min_share)
                    g = mrio_graph(
                        copy(W),
                        gref_nodes(n);
                        direction = direction,
                        threshold = threshold,
                        min_share = min_share,
                        self_loops = self_loops,
                    )
                    membs = Vector{Int32}[
                        ones(Int32, n),
                        Int32.(1:n),
                        Int32.(rand(MersenneTwister(1000 + n), 1:3, n)),
                    ]
                    W === GRAPH_PLANT_Z && push!(membs, copy(GRAPH_PLANT_MEMB))
                    for γ in (0.5, 1.0, 2.0)
                        for memb in membs
                            q = ext._partition_modularity(g, memb, Float64(γ))
                            q_ref = gref_modularity(W, directed, tau, self_loops, memb, γ)
                            @test isapprox(q, q_ref; rtol = 1.0e-12, atol = 1.0e-12)
                            if directed || !self_loops
                                q_graphs = Graphs.modularity(
                                    g,
                                    memb;
                                    distmx = Graphs.weights(g),
                                    γ = Float64(γ),
                                )
                                @test isapprox(q, q_graphs; rtol = 1.0e-5)
                            end
                        end
                    end
                end
            end
        end
    end

    # Hand-computed cases: two disjoint bidirectional pairs partitioned per
    # pair give Q = 0.5 at γ = 1, directed and undirected.
    # Directed: m = 4, e_1 = e_2 = 2, K^out = K^in = 2 per community, so
    # Q = ((2 - 4/4) + (2 - 4/4)) / 4 = 0.5.
    # Undirected: symmetrized weights are 2 per pair, m = 8, S_1 = S_2 = 4,
    # K_1 = K_2 = 4, so Q = ((4 - 16/8) + (4 - 16/8)) / 8 = 0.5.
    WPAIR = [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
    MPAIR = Int32[1, 1, 2, 2]
    gpair_dir = mrio_graph(copy(WPAIR), gref_nodes(4))
    @test ext._partition_modularity(gpair_dir, MPAIR, 1.0) ≈ 0.5
    gpair_und = mrio_graph(copy(WPAIR), gref_nodes(4); direction = :undirected)
    @test ext._partition_modularity(gpair_und, MPAIR, 1.0) ≈ 0.5
    @test Graphs.modularity(gpair_dir, MPAIR; distmx = Graphs.weights(gpair_dir)) ≈ 0.5
    @test Graphs.modularity(gpair_und, MPAIR; distmx = Graphs.weights(gpair_und)) ≈ 0.5

    # Empty total weight gives 0.0, matching Graphs.modularity.
    gzero = mrio_graph(zeros(2, 2), gref_nodes(2))
    @test ext._partition_modularity(gzero, Int32[1, 1], 1.0) == 0.0
    @test Graphs.modularity(gzero, Int32[1, 1]; distmx = Graphs.weights(gzero)) == 0.0

    # Labels < 1 are rejected.
    @test_throws ArgumentError ext._partition_modularity(gzero, Int32[1, 0], 1.0)

    # Documented divergence case: undirected + self_loops = true + nonzero
    # diagonal. Our value follows the clean formula (diagonal counted once);
    # Graphs.jl double-counts diagonal weight in its total `m` (both v1.13.1
    # and v1.15.0), so equality with Graphs.modularity must NOT hold there.
    gdiv = mrio_graph(copy(GRAPH_W3), gref_nodes(3); direction = :undirected, self_loops = true)
    membdiv = Int32[1, 1, 2]
    qdiv = ext._partition_modularity(gdiv, membdiv, 1.0)
    @test qdiv ≈ gref_modularity(GRAPH_W3, false, 0.0, true, membdiv, 1.0)
    # Hand value from the clean formula (diagonal counted once). Sym pairs:
    # w(1,2) = 2 + 1 = 3, w(1,3) = -0.5 + 0 = -0.5, w(2,3) = 0 + 3 = 3;
    # diagonals 0.0, 4.0, 1.5. m = 5.5 + 2 * 5.5 = 16.5; e_1 = 0 + 4 + 2*3 =
    # 10, K_1 = 2.5 + 10 = 12.5; e_2 = 1.5, K_2 = 4.0; so
    # Q = (11.5 − 172.25 / 16.5) / 16.5. This is the contract if a future
    # Graphs.jl changes its self-loop convention.
    @test qdiv ≈ 0.06427915518824609
    # Deliberate probe of the Graphs.jl self-loop convention (both v1.13.1
    # and v1.15.0 diverge from the clean formula here by double-counting
    # diagonal weight in their total `m`); if a future Graphs.jl changes,
    # the hand-value assertion above is the contract.
    qdiv_graphs = Graphs.modularity(gdiv, membdiv; distmx = Graphs.weights(gdiv), γ = 1.0)
    @test !isapprox(qdiv, qdiv_graphs; rtol = 1.0e-5)

    # Allocation guard: `_partition_modularity` streams in O(n) memory (no
    # boxed per-element accumulator). Warm-up then measure on n = 1000
    # random graphs, directed and undirected.
    Wall = rand(MersenneTwister(31), 1000, 1000)
    mall = Int32.(rand(MersenneTwister(32), 1:4, 1000))
    gall_dir = mrio_graph(copy(Wall), gref_nodes(1000))
    gall_und = mrio_graph(copy(Wall), gref_nodes(1000); direction = :undirected)
    gref_mod_alloc(gall_dir, mall, 1.0) # warm-up (compilation)
    gref_mod_alloc(gall_und, mall, 1.0) # warm-up (compilation)
    @test gref_mod_alloc(gall_dir, mall, 1.0) < 500_000
    @test gref_mod_alloc(gall_und, mall, 1.0) < 500_000
end

@testset "Community plumbing" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)

    # Label relabeling to contiguous 1:k in order of first appearance.
    @test ext._relabel_first_appearance([7, 7, 3, 5, 3]) == Int32[1, 1, 2, 3, 2]
    @test ext._relabel_first_appearance(Int32[1, 1, 2, 3, 2]) == Int32[1, 1, 2, 3, 2]
    @test_throws ArgumentError ext._relabel_first_appearance(Int32[])
    @test_throws ArgumentError ext._relabel_first_appearance([1, 0, 2])
    @test_throws ArgumentError ext._relabel_first_appearance(Int32[-2])

    g = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6))
    res = communities(g; algorithm = :label_propagation, seed = 7)

    # Result fields share the graph state by reference, never by copy.
    @test res.membership isa Vector{Int32}
    @test res.algorithm == :label_propagation
    @test res.resolution == 1.0
    @test res.seed == 7
    @test res.nodes === g.nodes
    @test res.weights === g.weights
    @test res.filter == g.filter
    @test res.directed == true
    @test res.modularity ≈ ext._partition_modularity(g, res.membership, res.resolution)
    gund = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6); direction = :undirected)
    resund = communities(gund; algorithm = :label_propagation, seed = 7)
    @test resund.directed == false
    @test resund.modularity ≈ ext._partition_modularity(gund, resund.membership, resund.resolution)

    # community_table: node metadata plus the community column, fully copied.
    table = community_table(res)
    @test names(table) == [names(res.nodes); "community"]
    @test table.community == res.membership
    @test eltype(table.community) == Int32
    @test nrow(table) == nrow(res.nodes)
    @test table[!, 1] !== res.nodes[!, 1]
    @test table[!, :community] !== res.membership

    # community_summary on a hand-computable 4-node directed example.
    # W4 = [0 5 1 0; 0 0 0 0; 0 0 0 4; 2 0 0 0], membership [1, 1, 2, 2]:
    # internal(1) = A[1,2] = 5; external(1) = A[1,3] + A[4,1] = 1 + 2 = 3;
    # share(1) = 5/8 = 0.625. internal(2) = A[3,4] = 4;
    # external(2) = A[4,1] + A[1,3] = 2 + 1 = 3; share(2) = 4/7.
    # Countries [A, A, B, B]: community 1 is all A, community 2 is all B.
    # Sectors [s1, s2, s1, s2]: both communities tie s1/s2, first in node
    # order wins (s1) with share 1/2.
    W4 = [0.0 5.0 1.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 4.0; 2.0 0.0 0.0 0.0]
    nodes4 = DataFrame(CountryCode = ["A", "A", "B", "B"], Sector = ["s1", "s2", "s1", "s2"])
    res4 = CommunityResult(
        Int32[1, 1, 2, 2],
        0.0,
        :label_propagation,
        1.0,
        nothing,
        nodes4,
        copy(W4),
        (0.0, 0.0, false, 0.0),
        true,
    )
    s4 = community_summary(res4)
    @test s4.community == Int32[1, 2]
    @test eltype(s4.community) == Int32
    @test s4.size == [2, 2]
    @test s4.internal_flow ≈ [5.0, 4.0]
    @test s4.external_flow ≈ [3.0, 3.0]
    @test s4.internal_share ≈ [0.625, 4 / 7]
    @test s4.n_countries == [1, 1]
    @test s4.top_country == ["A", "B"]
    @test s4.top_country_share ≈ [1.0, 1.0]
    @test s4.n_sectors == [2, 2]
    @test s4.top_sector == ["s1", "s1"]
    @test s4.top_sector_share ≈ [0.5, 0.5]

    # community_summary on a hand-computable 4-node undirected example.
    # Wu4 strictly upper triangular: Wu4[1,2] = 1, Wu4[2,3] = 5,
    # Wu4[3,4] = 1, Wu4[1,4] = 2; membership [1, 1, 2, 2].
    # Symmetrized pairs: w(1,2) = 1, w(2,3) = 5, w(3,4) = 1, w(1,4) = 2.
    # Internal pairs {1,2} in c1, {3,4} in c2 (double sum):
    # internal_flow = [2*1, 2*1] = [2.0, 2.0].
    # Crossing pairs {2,3} and {1,4} each contribute A_ij + A_ji:
    # external_flow = [2*(5+2), 2*(5+2)] = [14.0, 14.0].
    # internal_share = [2/16, 2/16] = [0.125, 0.125].
    Wu4 = [0.0 1.0 0.0 2.0; 0.0 0.0 5.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 0.0 0.0]
    resu4 = CommunityResult(
        Int32[1, 1, 2, 2],
        0.0,
        :label_propagation,
        1.0,
        nothing,
        nodes4,
        copy(Wu4),
        (0.0, 0.0, false, 0.0),
        false,
    )
    su = community_summary(resu4)
    @test su.internal_flow ≈ [2.0, 2.0]
    @test su.external_flow ≈ [14.0, 14.0]
    @test su.internal_share ≈ [0.125, 0.125]

    # community_summary with a min_share-based cutoff on the same Wu4
    # fixture: Σ|W| = 1 + 5 + 1 + 2 = 9, so filter (0.0, 0.2, false, 9.0)
    # gives τ = 0.2 * 9 = 1.8. Sym pairs w(1,2) = 1 and w(3,4) = 1 are
    # pruned (|w| < 1.8); w(2,3) = 5 and w(1,4) = 2 survive. Both survivors
    # cross the [1,1,2,2] communities, so internal_flow = [0.0, 0.0] and
    # each crossing pair contributes A_ij + A_ji to both sides:
    # external_flow = [2*(5+2), 2*(5+2)] = [14.0, 14.0],
    # internal_share = [0.0, 0.0].
    resu4f = CommunityResult(
        Int32[1, 1, 2, 2],
        0.0,
        :label_propagation,
        1.0,
        nothing,
        nodes4,
        copy(Wu4),
        (0.0, 0.2, false, 9.0),
        false,
    )
    suf = community_summary(resu4f)
    @test suf.internal_flow ≈ [0.0, 0.0]
    @test suf.external_flow ≈ [14.0, 14.0]
    @test suf.internal_share ≈ [0.0, 0.0]

    # Graceful fallback: without country/sector metadata the composition
    # columns are absent and the core columns are identical.
    nodes_nb = DataFrame(ID = ["a", "b", "c", "d"])
    res_nb = CommunityResult(
        Int32[1, 1, 2, 2],
        0.0,
        :label_propagation,
        1.0,
        nothing,
        nodes_nb,
        copy(W4),
        (0.0, 0.0, false, 0.0),
        true,
    )
    s_nb = community_summary(res_nb)
    @test names(s_nb) == ["community", "size", "internal_flow", "external_flow", "internal_share"]
    @test Matrix(s_nb[!, [:internal_flow, :external_flow, :internal_share]]) ≈
        Matrix(s4[!, [:internal_flow, :external_flow, :internal_share]])
    @test s_nb.size == s4.size
end

@testset "Label propagation" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    mrio = gref_plant_mrio()
    for direction in (:directed, :undirected)
        g = mrio_graph(mrio; direction = direction)
        ra = communities(g; algorithm = :label_propagation, seed = 42)
        rb = communities(g; algorithm = :label_propagation, seed = 42)
        # Determinism with a fixed seed.
        @test ra.membership == rb.membership
        # Result invariants: contiguous 1:k labels in first-appearance order.
        @test length(ra.membership) == 6
        @test all(>=(1), ra.membership)
        @test sort(unique(ra.membership)) == collect(1:maximum(ra.membership))
        @test ra.membership == ext._relabel_first_appearance(ra.membership)
        # Modularity self-consistency.
        @test ra.modularity ≈ ext._partition_modularity(g, ra.membership, ra.resolution)
        @test ra.seed == 42
        # The MRIO method forwards identically and shares the row metadata.
        rm = communities(mrio; direction = direction, algorithm = :label_propagation, seed = 42)
        @test rm.membership == ra.membership
        @test rm.modularity ≈ ra.modularity
        @test rm.nodes === mrio.T.row_indices
        @test rm.weights === mrio.T.data
        # Best-of-nruns keeps a partition at least as good as a single run
        # (run 1 of nruns = 3 uses the same RNG stream as nruns = 1).
        single = communities(g; algorithm = :label_propagation, seed = 7)
        best3 = communities(g; algorithm = :label_propagation, seed = 7, nruns = 3)
        @test best3.seed == 7
        @test best3.modularity >= single.modularity - 1.0e-12
        @test length(best3.membership) == 6
        @test best3.membership == ext._relabel_first_appearance(best3.membership)
    end
    # Random fixtures (directed and undirected) satisfy the same invariants.
    Wr = rand(MersenneTwister(5), 8, 8)
    for direction in (:directed, :undirected)
        gr = mrio_graph(copy(Wr), gref_nodes(8); direction = direction)
        rr = communities(gr; algorithm = :label_propagation, seed = 3)
        @test length(rr.membership) == 8
        @test all(>=(1), rr.membership)
        @test rr.membership == ext._relabel_first_appearance(rr.membership)
        @test rr.modularity ≈ ext._partition_modularity(gr, rr.membership, rr.resolution)
    end
    # Edgeless graphs do not crash (every node keeps its own label).
    ge = mrio_graph(zeros(4, 4), gref_nodes(4))
    re = communities(ge; algorithm = :label_propagation, seed = 1)
    @test re.membership == ext._relabel_first_appearance(re.membership)
    @test re.modularity == 0.0
    # community_summary on the edgeless result: no flows anywhere, and the
    # zero-denominator rule gives internal_share == 0.0 (not NaN).
    se = community_summary(re)
    @test se.internal_flow ≈ zeros(4)
    @test se.external_flow ≈ zeros(4)
    @test se.internal_share ≈ zeros(4)
    # Unseeded runs complete and satisfy the invariants.
    g_unseeded = mrio_graph(mrio)
    ru = communities(g_unseeded; algorithm = :label_propagation)
    @test ru.seed === nothing
    @test length(ru.membership) == 6
end

@testset "Community validation" begin
    g3 = mrio_graph(copy(GRAPH_W3), gref_nodes(3))
    @test_throws ArgumentError communities(g3; algorithm = :bogus)
    err_alg = try
        communities(g3; algorithm = :bogus)
        nothing
    catch e
        e
    end
    @test err_alg isa ArgumentError
    @test occursin("louvain", sprint(showerror, err_alg))
    @test occursin("leiden", sprint(showerror, err_alg))
    @test_throws ArgumentError communities(g3; algorithm = :label_propagation, resolution = 0.0)
    @test_throws ArgumentError communities(g3; algorithm = :label_propagation, resolution = -1.0)
    @test_throws ArgumentError communities(g3; algorithm = :label_propagation, resolution = NaN)
    @test_throws ArgumentError communities(g3; algorithm = :label_propagation, resolution = Inf)
    @test_throws ArgumentError communities(g3; algorithm = :label_propagation, nruns = 0)
    @test_throws ArgumentError communities(g3; algorithm = :louvain, ncommunities = 3)
    err_nc = try
        communities(g3; algorithm = :louvain, ncommunities = 3)
        nothing
    catch e
        e
    end
    @test err_nc isa ArgumentError
    @test occursin("ncommunities", sprint(showerror, err_nc))
    @test occursin(":spectral", sprint(showerror, err_nc))
    @test_throws ArgumentError communities(
        g3;
        algorithm = :spectral,
        ncommunities = 0,
    )
    g0 = mrio_graph(zeros(0, 0), DataFrame(CountryCode = String[], Sector = String[]))
    @test_throws ArgumentError communities(g0; algorithm = :label_propagation)
    # seed/ncommunities must be Int-range Integers (Bool excluded): Bool is
    # an Integer subtype and huge Integers overflow Int(...), so both are
    # rejected with ArgumentError (not silent coercion / InexactError).
    @test_throws ArgumentError communities(g3; algorithm = :label_propagation, seed = true)
    @test_throws ArgumentError communities(g3; algorithm = :spectral, ncommunities = true)
    @test_throws ArgumentError communities(g3; algorithm = :label_propagation, seed = big(2)^80)
    @test_throws ArgumentError communities(g3; algorithm = :spectral, ncommunities = big(2)^80)
    @test_throws ArgumentError communities(g3; algorithm = :label_propagation, nruns = true)
    @test_throws ArgumentError communities(g3; algorithm = :label_propagation, nruns = big(2)^80)
    # Non-finite/negative effective weights are rejected before the run
    # loop (a NaN modularity would otherwise leave an empty membership
    # with modularity = -Inf). NaN sits at the retained off-diagonal
    # (1, 2): a diagonal NaN with the default self_loops = false is pruned
    # by the self-loop rule everywhere (mirroring the PageRank tests).
    @test_throws ArgumentError communities(
        mrio_graph([0.0 NaN; 0.0 0.0], gref_nodes(2));
        algorithm = :label_propagation,
    )
    @test_throws ArgumentError communities(
        mrio_graph([0.0 Inf; 0.0 0.0], gref_nodes(2));
        algorithm = :label_propagation,
    )
    @test_throws ArgumentError communities(
        mrio_graph([-1.0 0.0; 0.0 0.0], gref_nodes(2); self_loops = true);
        algorithm = :label_propagation,
    )
    # Undirected w(1, 2) = Inf + -Inf = NaN is retained and must throw.
    @test_throws ArgumentError communities(
        mrio_graph([Inf -Inf; 0.0 0.0], gref_nodes(2); direction = :undirected);
        algorithm = :label_propagation,
    )
    # The error message names the entry point and the offending pair.
    gnan_err = try
        communities(mrio_graph([0.0 NaN; 0.0 0.0], gref_nodes(2)); algorithm = :label_propagation)
        nothing
    catch e
        e
    end
    @test gnan_err isa ArgumentError
    @test occursin("communities", sprint(showerror, gnan_err))
    @test occursin("w(1, 2)", sprint(showerror, gnan_err))
    # Unknown input types still hit the stub fallback extension error.
    @test_throws ErrorException communities(42)
end

@testset "Louvain" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    @test ext !== nothing

    # Naive single-move quantities from an explicit pair-weight matrix, in
    # the argument order of `ext._louvain_gain`. A destination above every
    # used label is the fresh singleton (all `a`/`K` terms zero).
    function louv_move_stats(P::AbstractMatrix, memb::AbstractVector, i::Int, d::Int)
        n = size(P, 1)
        ci = memb[i]
        kout = vec(sum(P; dims = 2))
        kin = vec(sum(P; dims = 1))
        members_c = [j for j in 1:n if memb[j] == ci]
        a_out_c = sum(P[i, j] for j in members_c)
        a_in_c = sum(P[j, i] for j in members_c)
        Kc_out = sum(kout[j] for j in members_c)
        Kc_in = sum(kin[j] for j in members_c)
        if d > maximum(memb)
            a_out_d, a_in_d, Kd_out, Kd_in = 0.0, 0.0, 0.0, 0.0
        else
            members_d = [j for j in 1:n if memb[j] == d]
            a_out_d = sum(P[i, j] for j in members_d)
            a_in_d = sum(P[j, i] for j in members_d)
            Kd_out = sum(kout[j] for j in members_d)
            Kd_in = sum(kin[j] for j in members_d)
        end
        return kout[i], kin[i], Float64(P[i, i]), a_out_c, a_in_c, a_out_d, a_in_d, Kc_out, Kc_in, Kd_out, Kd_in
    end

    # Explicit pair-weight matrix of a level-0 view (test-side only).
    function louv_explicit_pairs(V, n::Int)
        return [ext._louvain_pair(V, i, j) for i in 1:n, j in 1:n]
    end

    rng = MersenneTwister(20260924)

    # 1. ΔQ identity: the implemented gain predicts the naive Q difference
    # for single-node moves (including fresh singletons), on level-0 views
    # of both directednesses and on implementation-built aggregates. The
    # pair-weight accessors are checked elementwise against the naive
    # references on the same matrices.
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        wfun = directed ? gref_directed_w : gref_undirected_w
        for self_loops in (false, true)
            for γ in (0.5, 1.0, 2.0)
                for trial in 1:6
                    n = rand(rng, 4:10)
                    W = rand(rng, n, n)
                    W[rand(rng, n, n) .< 0.3] .= 0.0
                    if trial % 2 == 0
                        for i in 1:n
                            W[i, i] = 0.0
                        end
                    end
                    tau = gref_tau(W, 0.0, 0.0)
                    g = mrio_graph(
                        copy(W),
                        gref_nodes(n);
                        direction = direction,
                        self_loops = self_loops,
                    )
                    V = ext._louvain_view(g)
                    for i in 1:n, j in 1:n
                        @test ext._louvain_pair(V, i, j) == wfun(W, i, j, tau, self_loops)
                    end
                    P = louv_explicit_pairs(V, n)
                    if sum(P) == 0.0
                        continue
                    end
                    memb = rand(rng, 1:3, n)
                    for _mv in 1:4
                        i = rand(rng, 1:n)
                        others = [d for d in unique(memb) if d != memb[i]]
                        d = rand(rng, vcat(others, maximum(memb) + 1))
                        pred = ext._louvain_gain(louv_move_stats(P, memb, i, d)..., sum(P), Float64(γ))
                        before = gref_q_from_P(P, memb, γ)
                        memb2 = copy(memb)
                        memb2[i] = d
                        @test isapprox(pred, gref_q_from_P(P, memb2, γ) - before; atol = 1.0e-10)
                    end
                    # Same identity one aggregation level up, on the
                    # implementation-built aggregate (double-sum convention).
                    idx, k = ext._compact_membership(memb, n)
                    p1 = ext._louvain_aggregate(V, idx, k)
                    ρ = [rand(rng, 1:k) for _ in 1:k]
                    for _mv in 1:3
                        i = rand(rng, 1:k)
                        others = [d for d in unique(ρ) if d != ρ[i]]
                        d = rand(rng, vcat(others, maximum(ρ) + 1))
                        pred = ext._louvain_gain(louv_move_stats(p1, ρ, i, d)..., sum(p1), Float64(γ))
                        before = gref_q_from_P(p1, ρ, γ)
                        ρ2 = copy(ρ)
                        ρ2[i] = d
                        @test isapprox(pred, gref_q_from_P(p1, ρ2, γ) - before; atol = 1.0e-10)
                    end
                end
            end
        end
    end

    # 2. Aggregation preservation: the implementation-built aggregate equals
    # the naive ordered double sum, conserves the community strengths, and
    # satisfies Q(P₁, ρ) == Q(P, lifted(ρ ∘ π)). A sym-on-aggregate
    # double-count would break the Q equality for undirected graphs.
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        n = 7
        W = rand(rng, n, n)
        W[rand(rng, n, n) .< 0.3] .= 0.0
        for self_loops in (false, true)
            g = mrio_graph(copy(W), gref_nodes(n); direction = direction, self_loops = self_loops)
            V = ext._louvain_view(g)
            P = louv_explicit_pairs(V, n)
            π = rand(rng, 1:3, n)
            idx, k = ext._compact_membership(π, n)
            p1 = ext._louvain_aggregate(V, idx, k)
            p1ref = zeros(k, k)
            for c in 1:k, d in 1:k
                p1ref[c, d] = sum(P[i, j] for i in 1:n if idx[i] == c for j in 1:n if idx[j] == d)
            end
            @test p1 ≈ p1ref atol = 1.0e-12
            kout = vec(sum(P; dims = 2))
            kin = vec(sum(P; dims = 1))
            for c in 1:k
                @test sum(p1[c, :]) ≈ sum(kout[idx .== c]) atol = 1.0e-12
                @test sum(p1[:, c]) ≈ sum(kin[idx .== c]) atol = 1.0e-12
            end
            if !directed
                @test p1 ≈ p1' atol = 1.0e-12
            end
            for γ in (0.5, 1.0, 2.0)
                ρ = rand(rng, 1:2, k)
                lifted = [ρ[idx[i]] for i in 1:n]
                @test gref_q_from_P(p1, ρ, γ) ≈ gref_q_from_P(P, lifted, γ) atol = 1.0e-12
            end
        end
    end

    # 3. Local optimality: the returned partition admits no strictly
    # improving single-node move (exhaustive check via the naive Q).
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        for trial in 1:4
            n = 8
            W = rand(rng, n, n)
            W[rand(rng, n, n) .< 0.3] .= 0.0
            g = mrio_graph(copy(W), gref_nodes(n); direction = direction, self_loops = trial % 2 == 0)
            memb = ext._louvain_partition(g, 1.0, MersenneTwister(100 + trial))
            @test length(memb) == n
            @test all(>=(1), memb)
            P = louv_explicit_pairs(ext._louvain_view(g), n)
            q0 = gref_q_from_P(P, memb, 1.0)
            for i in 1:n
                dests = [d for d in unique(memb) if d != memb[i]]
                push!(dests, maximum(memb) + 1)
                for d in dests
                    memb2 = copy(memb)
                    memb2[i] = d
                    @test gref_q_from_P(P, memb2, 1.0) <= q0 + 1.0e-9
                end
            end
        end
    end

    # 4. Planted recovery (plan test 1): `communities(...; algorithm =
    # :louvain, seed = 11)` recovers GRAPH_PLANT_MEMB exactly (labels may be
    # permuted), with modularity agreeing with Graphs.modularity — via the
    # MRIO method (default :undirected), the :directed variant, and the
    # directed graph-method path.
    mrio = gref_plant_mrio()
    res = communities(mrio; algorithm = :louvain, seed = 11)
    @test gref_same_partition(res.membership, GRAPH_PLANT_MEMB)
    @test res.modularity ≈
        ext._partition_modularity(mrio_graph(mrio; direction = :undirected), res.membership, res.resolution)
    gund = mrio_graph(mrio; direction = :undirected)
    @test res.modularity ≈ Graphs.modularity(gund, res.membership; distmx = Graphs.weights(gund), γ = res.resolution) rtol =
        1.0e-5
    resd = communities(mrio; direction = :directed, algorithm = :louvain, seed = 11)
    @test gref_same_partition(resd.membership, GRAPH_PLANT_MEMB)
    gdir = mrio_graph(mrio; direction = :directed)
    @test resd.modularity ≈ Graphs.modularity(gdir, resd.membership; distmx = Graphs.weights(gdir), γ = resd.resolution) rtol =
        1.0e-5
    gg = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6))
    resg = communities(gg; algorithm = :louvain, seed = 11)
    @test gref_same_partition(resg.membership, GRAPH_PLANT_MEMB)
    @test resg.modularity ≈ Graphs.modularity(gg, resg.membership; distmx = Graphs.weights(gg), γ = resg.resolution) rtol =
        1.0e-5

    # 5. Determinism and best-of-nruns: the same seed gives identical
    # memberships, and nruns = 3 keeps a partition at least as good as each
    # single run (run r uses MersenneTwister(seed + r), as in `communities`).
    gd = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6))
    ra = communities(gd; algorithm = :louvain, seed = 13)
    rb = communities(gd; algorithm = :louvain, seed = 13)
    @test ra.membership == rb.membership
    best3 = communities(gd; algorithm = :louvain, seed = 7, nruns = 3)
    @test best3.seed == 7
    for r in 1:3
        single = ext._relabel_first_appearance(ext._louvain_partition(gd, 1.0, MersenneTwister(7 + r)))
        @test best3.modularity >= ext._partition_modularity(gd, single, 1.0) - 1.0e-12
    end

    # 6. Resolution behavior on the planted toy: γ = 3.0 yields at least as
    # many communities as γ = 0.5 (higher resolution ⇒ finer or equal).
    rlow = communities(mrio; algorithm = :louvain, seed = 11, resolution = 0.5)
    rhigh = communities(mrio; algorithm = :louvain, seed = 11, resolution = 3.0)
    @test length(unique(rhigh.membership)) >= length(unique(rlow.membership))

    # 7. Memory guard: one Louvain run on a structured mid-size matrix.
    # Block structure (20 blocks × 100 nodes, strong within-block weights
    # plus weak between-block noise) collapses to ~20 aggregate communities,
    # so the bound measures the level-0 O(n) streaming rather than a
    # legitimate O(n₁²) aggregate: a pure-random matrix could end with ≈ n/2
    # aggregate communities and must not be mis-flagged here.
    nb = 20
    bs = 100
    nbig = nb * bs
    Wbig = 0.05 .* rand(MersenneTwister(99), nbig, nbig)
    for b in 1:nb
        rows = ((b - 1) * bs + 1):(b * bs)
        Wbig[rows, rows] .+= 5.0
    end
    gbig = mrio_graph(Wbig, gref_nodes(nbig); direction = :undirected)
    gref_louvain_alloc(gbig) # warm-up (compilation)
    @test gref_louvain_alloc(gbig) < 25 * 2^20
end

@testset "Leiden" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    @test ext !== nothing

    # Explicit pair-weight matrix of a level-0 view (test-side only).
    function leid_explicit_pairs(V, n::Int)
        return [ext._louvain_pair(V, i, j) for i in 1:n, j in 1:n]
    end

    rng = MersenneTwister(20260925)

    # 1. Connectivity (plan test 2a, mandatory): every Leiden community is
    # connected — planted fixture (both directednesses), ring of cliques,
    # and a seeded random battery (both directednesses, γ ∈ {0.5, 1, 2}).
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        gp = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6); direction = direction)
        rp = communities(gp; algorithm = :leiden, seed = 11)
        gref_assert_connected(rp.membership, gp)
        gr = mrio_graph(copy(gref_ring_w()), gref_nodes(20); direction = direction)
        rr = communities(gr; algorithm = :leiden, seed = 1, nruns = 10)
        gref_assert_connected(rr.membership, gr)
    end
    for t in 1:5
        r = MersenneTwister(t)
        n = 30
        # Strictly-upper random weights in [0, 1] plus a few extra random
        # directed entries (all non-negative, the Leiden domain).
        W = zeros(n, n)
        for i in 1:n, j in (i + 1):n
            W[i, j] = rand(r)
        end
        for _ in 1:10
            i = rand(r, 1:n)
            j = rand(r, 1:n)
            i != j && (W[i, j] += rand(r))
        end
        for directed in (true, false)
            direction = directed ? :directed : :undirected
            g = mrio_graph(copy(W), gref_nodes(n); direction = direction)
            for γ in (0.5, 1.0, 2.0)
                res = communities(g; algorithm = :leiden, resolution = γ, seed = 100 + t)
                gref_assert_connected(res.membership, g)
            end
        end
    end

    # 2. Q_Leiden ≥ Q_Louvain (plan test 2b, mandatory): planted fixture
    # (both directednesses), ring of cliques, and the disjoint-pairs toy
    # (hand-computed optimum Q = 2/3 at γ = 1, see `gref_pairs_w`).
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        gp = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6); direction = direction)
        lp = communities(gp; algorithm = :louvain, nruns = 10, seed = 1)
        dp = communities(gp; algorithm = :leiden, nruns = 10, seed = 1)
        @test dp.modularity >= lp.modularity - 1.0e-12
        @test gref_same_partition(dp.membership, GRAPH_PLANT_MEMB)
    end
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        gr = mrio_graph(copy(gref_ring_w()), gref_nodes(20); direction = direction)
        lr = communities(gr; algorithm = :louvain, nruns = 10, seed = 1)
        dr = communities(gr; algorithm = :leiden, nruns = 10, seed = 1)
        @test dr.modularity >= lr.modularity - 1.0e-12
        @test gref_same_partition(dr.membership, gref_ring_memb())
    end
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        gx = mrio_graph(copy(gref_pairs_w()), gref_nodes(6); direction = direction)
        lx = communities(gx; algorithm = :louvain, nruns = 10, seed = 1)
        dx = communities(gx; algorithm = :leiden, nruns = 10, seed = 1)
        @test dx.modularity >= lx.modularity - 1.0e-12
        @test dx.modularity ≈ 2 / 3 atol = 1.0e-9
        @test gref_same_partition(dx.membership, GRAPH_PAIRS_MEMB)
    end

    # 2c. Fixpoint dual guarantee: `_leiden_partition` outputs admit no
    # strictly improving single-node move (exhaustive naive-Q check mirroring
    # Louvain §3) AND every community is connected — the two together pin the
    # flat-move/split fixpoint.
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        for trial in 1:4
            n = 8
            Wr = rand(rng, n, n)
            Wr[rand(rng, n, n) .< 0.3] .= 0.0
            g = mrio_graph(copy(Wr), gref_nodes(n); direction = direction, self_loops = trial % 2 == 0)
            memb = ext._leiden_partition(g, 1.0, MersenneTwister(200 + trial))
            @test length(memb) == n
            @test all(>=(1), memb)
            gref_assert_connected(memb, g)
            P = leid_explicit_pairs(ext._louvain_view(g), n)
            q0 = gref_q_from_P(P, memb, 1.0)
            for i in 1:n
                dests = [d for d in unique(memb) if d != memb[i]]
                push!(dests, maximum(memb) + 1)
                for d in dests
                    memb2 = copy(memb)
                    memb2[i] = d
                    @test gref_q_from_P(P, memb2, 1.0) <= q0 + 1.0e-9
                end
            end
        end
    end

    # 3. Refinement invariants (the correctness device): `_leiden_refine`
    # refines its input partition, every refined community is connected
    # (adjacency > 0, weak), singletons refine to themselves, and fixed
    # seeds give identical output.
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        for trial in 1:5
            n = rand(rng, 4:9)
            W = rand(rng, n, n)
            W[rand(rng, n, n) .< 0.3] .= 0.0
            g = mrio_graph(copy(W), gref_nodes(n); direction = direction)
            V = ext._louvain_view(g)
            P = leid_explicit_pairs(V, n)
            sum(P) == 0.0 && continue
            kout = vec(sum(P; dims = 2))
            kin = vec(sum(P; dims = 1))
            m = sum(P)
            memb = rand(rng, 1:3, n)
            ref1 = ext._leiden_refine(V, memb, kout, kin, m, 1.0, 0.01, MersenneTwister(77))
            ref2 = ext._leiden_refine(V, memb, kout, kin, m, 1.0, 0.01, MersenneTwister(77))
            @test ref1 == ref2
            # Refines the input: each refined community ⊆ one community.
            for c in unique(ref1)
                parents = Set(memb[i] for i in 1:n if ref1[i] == c)
                @test length(parents) == 1
            end
            # Every refined community is connected (weak adjacency).
            for c in unique(ref1)
                C = [i for i in 1:n if ref1[i] == c]
                seen = Set([C[1]])
                stack = [C[1]]
                while !isempty(stack)
                    u = pop!(stack)
                    for v in C
                        if v ∉ seen && (P[u, v] > 0.0 || P[v, u] > 0.0)
                            push!(seen, v)
                            push!(stack, v)
                        end
                    end
                end
                @test seen == Set(C)
            end
            # Singletons refine to themselves, identically.
            sing = collect(1:n)
            @test ext._leiden_refine(V, sing, kout, kin, m, 1.0, 0.01, MersenneTwister(3)) == sing
        end
    end

    # 3a. Non-positive θ fallback: θ = 0.0 and θ = -1.0 take the documented
    # deterministic smallest-id argmax path, so both agree exactly and the
    # output is a valid refinement (refines the input, every community
    # connected). If float round-off ever leaves the Boltzmann cumulative
    # weight below the draw, the implicit smallest-label fallback keeps the
    # first candidate (documented in `_leiden_refine`).
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        n = 7
        W = rand(rng, n, n)
        W[rand(rng, n, n) .< 0.3] .= 0.0
        g = mrio_graph(copy(W), gref_nodes(n); direction = direction)
        V = ext._louvain_view(g)
        P = leid_explicit_pairs(V, n)
        sum(P) == 0.0 && continue
        kout = vec(sum(P; dims = 2))
        kin = vec(sum(P; dims = 1))
        m = sum(P)
        memb = rand(rng, 1:3, n)
        r0 = ext._leiden_refine(V, memb, kout, kin, m, 1.0, 0.0, MersenneTwister(77))
        rn = ext._leiden_refine(V, memb, kout, kin, m, 1.0, -1.0, MersenneTwister(77))
        @test r0 == rn
        for c in unique(r0)
            parents = Set(memb[i] for i in 1:n if r0[i] == c)
            @test length(parents) == 1
        end
        for c in unique(r0)
            C = [i for i in 1:n if r0[i] == c]
            seen = Set([C[1]])
            stack = [C[1]]
            while !isempty(stack)
                u = pop!(stack)
                for v in C
                    if v ∉ seen && (P[u, v] > 0.0 || P[v, u] > 0.0)
                        push!(seen, v)
                        push!(stack, v)
                    end
                end
            end
            @test seen == Set(C)
        end
    end

    # 3a2. `_leiden_split_disconnected` unit cases: an isolated node pinned
    # inside a community splits with Q unchanged, and a directed asymmetric
    # bridge splits on weak connectivity.
    let
        Wiso = zeros(4, 4)
        Wiso[1, 2] = Wiso[2, 1] = 4.0
        Wiso[2, 3] = Wiso[3, 2] = 4.0
        Wiso[1, 3] = Wiso[3, 1] = 4.0
        giso = mrio_graph(copy(Wiso), gref_nodes(4))
        Viso = ext._louvain_view(giso)
        memb_iso = [1, 1, 1, 1]
        q_before = ext._partition_modularity(giso, Int32.(memb_iso), 1.0)
        split_iso, did_iso = ext._leiden_split_disconnected(Viso, memb_iso)
        @test did_iso == true
        @test count(==(split_iso[4]), split_iso) == 1
        @test length(unique(split_iso)) == 2
        @test ext._partition_modularity(giso, Int32.(split_iso), 1.0) ≈ q_before
    end
    let
        # Directed asymmetric bridge: edges 1→2 and 3→4 only. Weak components
        # of [1,1,1,1] are {1,2} and {3,4}.
        Wd = zeros(4, 4)
        Wd[1, 2] = 3.0
        Wd[3, 4] = 3.0
        gd = mrio_graph(copy(Wd), gref_nodes(4); direction = :directed)
        Vd = ext._louvain_view(gd)
        memb_d = [1, 1, 1, 1]
        split_d, did_d = ext._leiden_split_disconnected(Vd, memb_d)
        @test did_d == true
        @test split_d[1] == split_d[2]
        @test split_d[3] == split_d[4]
        @test split_d[1] != split_d[3]
    end

    # 3b. Well-connectedness predicate on crafted cases: two symmetric nodes
    # with P[1,2] = P[2,1] = 2.0 (m = 4.0, K_1 = K_2 = 2.0, E = 4.0) satisfy
    # wc ⟺ 4 ≥ 2γ ⟺ γ ≤ 2; empty cross weight fails for positive volumes.
    @test ext._leiden_well_connected(4.0, 2.0, 2.0, 2.0, 2.0, 4.0, 1.0)
    @test ext._leiden_well_connected(4.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0)
    @test !ext._leiden_well_connected(4.0, 2.0, 2.0, 2.0, 2.0, 4.0, 3.0)
    @test !ext._leiden_well_connected(0.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0)
    @test ext._leiden_well_connected(0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0)

    # 3c. Local-optimality identity (undirected only): wc({v}, c∖{v}) ⟺
    # ΔQ(v ↦ fresh singleton) ≤ 0. Extracting v gives Δe = -E (the ordered
    # cross weight leaves) and Δk = -2K_Bk_v, so ΔQ = (2γK_Bk_v/m - E)/m ≤ 0
    # ⟺ E ≥ 2γK_Bk_v/m, which is wc for symmetric P. This pins the
    # predicate's scaling against `_louvain_gain`.
    for trial in 1:10
        n = rand(rng, 3:8)
        W = rand(rng, n, n)
        W[rand(rng, n, n) .< 0.3] .= 0.0
        g = mrio_graph(copy(W), gref_nodes(n); direction = :undirected)
        V = ext._louvain_view(g)
        P = leid_explicit_pairs(V, n)
        sum(P) == 0.0 && continue
        kout = vec(sum(P; dims = 2))
        kin = vec(sum(P; dims = 1))
        m = sum(P)
        memb = rand(rng, 1:3, n)
        γ = rand(rng, (0.5, 1.0, 2.0))
        v = rand(rng, 1:n)
        c = memb[v]
        B = [j for j in 1:n if memb[j] == c && j != v]
        C = [j for j in 1:n if memb[j] == c]
        E = sum(P[v, j] + P[j, v] for j in B; init = 0.0)
        KB = sum(kout[j] for j in B; init = 0.0)
        wc = ext._leiden_well_connected(E, kout[v], kin[v], KB, KB, m, Float64(γ))
        a_c = sum(P[v, j] for j in C; init = 0.0)
        Kc = sum(kout[j] for j in C; init = 0.0)
        self_v = Float64(P[v, v])
        gain = ext._louvain_gain(
            kout[v],
            kin[v],
            self_v,
            a_c,
            a_c,
            0.0,
            0.0,
            Kc,
            Kc,
            0.0,
            0.0,
            m,
            Float64(γ),
        )
        @test (gain <= 0.0) == wc
        @test gain ≈ (2.0 * γ * KB * kout[v] / m - E) / m atol = 1.0e-12
    end

    # 4. Initial-partition semantics: after one aggregation step the carried
    # (lifted, non-refined) partition on the aggregate has exactly the
    # quality of the pre-aggregation carried partition.
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        n = 7
        W = rand(rng, n, n)
        W[rand(rng, n, n) .< 0.3] .= 0.0
        g = mrio_graph(copy(W), gref_nodes(n); direction = direction)
        V = ext._louvain_view(g)
        P = leid_explicit_pairs(V, n)
        sum(P) == 0.0 && continue
        kout = vec(sum(P; dims = 2))
        kin = vec(sum(P; dims = 1))
        m = sum(P)
        for γ in (0.5, 1.0, 2.0)
            memb = collect(1:n)
            ext._louvain_local_move!(V, memb, kout, kin, m, Float64(γ), MersenneTwister(9))
            refined = ext._leiden_refine(V, memb, kout, kin, m, Float64(γ), 0.01, MersenneTwister(9))
            idx, k = ext._compact_membership(refined, n)
            p1 = ext._louvain_aggregate(V, idx, k)
            carried = ext._leiden_lift(memb, idx, k)
            @test gref_q_from_P(p1, carried, γ) ≈ gref_q_from_P(P, memb, γ) atol = 1.0e-12
        end
    end

    # 5. Determinism, nruns, and edge cases.
    gd = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6))
    ra = communities(gd; algorithm = :leiden, seed = 13)
    rb = communities(gd; algorithm = :leiden, seed = 13)
    @test ra.membership == rb.membership
    best3 = communities(gd; algorithm = :leiden, seed = 7, nruns = 3)
    @test best3.seed == 7
    for r in 1:3
        single = ext._relabel_first_appearance(ext._leiden_partition(gd, 1.0, MersenneTwister(7 + r)))
        @test best3.modularity >= ext._partition_modularity(gd, single, 1.0) - 1.0e-12
    end
    # Single node (with and without a self-loop) → [1].
    @test communities(mrio_graph(zeros(1, 1), gref_nodes(1)); algorithm = :leiden).membership == Int32[1]
    g1sl = mrio_graph(reshape([2.0], 1, 1), gref_nodes(1); self_loops = true)
    @test communities(g1sl; algorithm = :leiden).membership == Int32[1]
    # All-zero matrix → singletons.
    gzero = mrio_graph(zeros(4, 4), gref_nodes(4))
    @test communities(gzero; algorithm = :leiden, seed = 1).membership == Int32[1, 2, 3, 4]
    # Isolated nodes stay singletons: node 4 has no incident weight.
    Wiso = zeros(4, 4)
    Wiso[1, 2] = Wiso[2, 1] = 4.0
    Wiso[2, 3] = Wiso[3, 2] = 4.0
    Wiso[1, 3] = Wiso[3, 1] = 4.0
    giso = mrio_graph(copy(Wiso), gref_nodes(4))
    riso = communities(giso; algorithm = :leiden, seed = 1)
    @test count(==(riso.membership[4]), riso.membership) == 1
    gref_assert_connected(riso.membership, giso)
    # Label contract: positive Int32 vector of length n.
    @test eltype(riso.membership) == Int32
    @test length(riso.membership) == 4
    @test all(>=(1), riso.membership)

    # 6. Memory guard: one Leiden run on the same n = 2000 block-structured
    # synthetic as the Louvain guard (20 blocks × 100 nodes collapsing to
    # ~20 aggregate communities, so the bound measures level-0 O(n)
    # streaming plus refinement bookkeeping).
    nb = 20
    bs = 100
    nbig = nb * bs
    Wbig = 0.05 .* rand(MersenneTwister(99), nbig, nbig)
    for b in 1:nb
        rows = ((b - 1) * bs + 1):(b * bs)
        Wbig[rows, rows] .+= 5.0
    end
    gbig = mrio_graph(Wbig, gref_nodes(nbig); direction = :undirected)
    gref_leiden_alloc(gbig) # warm-up (compilation)
    @test gref_leiden_alloc(gbig) < 25 * 2^20
end

@testset "Spectral" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    @test ext !== nothing

    # 1. Planted recovery (plan test 4): k = 2, seed = 5 recovers
    # GRAPH_PLANT_MEMB (labels may be permuted) via the directed graph path,
    # the undirected graph path, and the MRIO method (default :undirected).
    for direction in (:directed, :undirected)
        g = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6); direction = direction)
        res = communities(g; algorithm = :spectral, ncommunities = 2, seed = 5)
        @test gref_same_partition(res.membership, GRAPH_PLANT_MEMB)
        @test res.membership isa Vector{Int32}
        @test res.modularity ≈ ext._partition_modularity(g, res.membership, res.resolution)
    end
    resm = communities(gref_plant_mrio(); algorithm = :spectral, ncommunities = 2, seed = 5)
    @test gref_same_partition(resm.membership, GRAPH_PLANT_MEMB)
    @test resm.modularity ≈
        ext._partition_modularity(mrio_graph(gref_plant_mrio(); direction = :undirected), resm.membership, resm.resolution)

    # 2. Auto-k (eigengap): the same planted cases still find 2 communities
    # and recover the planted partition. Eigenvalue evidence on the planted
    # toy: λ ≈ [0, 0.016, 1.476, 1.5, 1.5, 1.508], so the winning gap is
    # g_2 = λ_3 − λ_2 ≈ 1.459 and k = 2.
    for direction in (:directed, :undirected)
        g = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6); direction = direction)
        resa = communities(g; algorithm = :spectral, seed = 5)
        @test length(unique(resa.membership)) == 2
        @test gref_same_partition(resa.membership, GRAPH_PLANT_MEMB)
    end
    resa_m = communities(gref_plant_mrio(); algorithm = :spectral, seed = 5)
    @test length(unique(resa_m.membership)) == 2
    @test gref_same_partition(resa_m.membership, GRAPH_PLANT_MEMB)
    # Ring of cliques with auto-k: 4 weakly coupled 5-cliques give
    # λ_1..λ_4 ≈ [0, 0.0049, 0.0049, 0.0098] with the winning gap
    # g_4 = λ_5 − λ_4 ≈ 1.225, so k = 4 and each clique is recovered.
    for direction in (:directed, :undirected)
        gring = mrio_graph(copy(gref_ring_w()), gref_nodes(20); direction = direction)
        rring = communities(gring; algorithm = :spectral, seed = 5)
        @test length(unique(rring.membership)) == 4
        @test gref_same_partition(rring.membership, gref_ring_memb())
    end

    # 3. Small aggregated graph (plan test 4, end to end): fold the planted
    # MRIO to the 2-node country × country matrix, then spectral k = 2
    # separates country A from country B. NOTE: a single
    # `aggregate(mrio, [:CountryCode])` folds rows only (2 × 6, not a graph);
    # the second `dims = 2` step folds the columns to the square matrix.
    mrio = gref_plant_mrio()
    magg = aggregate(aggregate(mrio, [:CountryCode]), [:CountryCode]; dims = 2)
    @test size(magg.T.data) == (2, 2)
    # Hand value: within-country flows 3 × 20 = 60 per country, bridge 1.0.
    @test magg.T.data ≈ [60.0 1.0; 0.0 60.0]
    gagg = mrio_graph(magg.T.data, magg.T.row_indices)
    ragg = communities(gagg; algorithm = :spectral, ncommunities = 2, seed = 3)
    @test sort(unique(ragg.membership)) == Int32[1, 2]
    @test ragg.membership[1] != ragg.membership[2]

    # 4. Size gate: n = 5000 passes, n = 5001 throws an ArgumentError naming
    # `aggregate` — via the gate helper directly and via the public API on a
    # one-shot 200 MB zeros matrix (freed after the let block).
    @test ext._spectral_check_size(5000) == 5000
    @test_throws ArgumentError ext._spectral_check_size(5001)
    gate_err = try
        ext._spectral_check_size(5001)
        nothing
    catch e
        e
    end
    @test gate_err isa ArgumentError
    @test occursin("aggregate", sprint(showerror, gate_err))
    @test occursin("5000", sprint(showerror, gate_err))
    big_err = try
        let
            gbig = mrio_graph(zeros(5001, 5001), gref_nodes(5001))
            communities(gbig; algorithm = :spectral, seed = 1)
        end
        nothing
    catch e
        e
    end
    @test big_err isa ArgumentError
    @test occursin("aggregate", sprint(showerror, big_err))

    # 5. k-means unit on crafted 2-D data (independent of any graph helper):
    # 3 well-separated clusters × 5 points with tiny N(0, 0.05²) noise.
    rng_data = MersenneTwister(1234)
    X = Matrix{Float64}(undef, 15, 2)
    for (c, (cx, cy)) in enumerate(((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))
        for j in 1:5
            X[(c - 1) * 5 + j, 1] = cx + 0.05 * randn(rng_data)
            X[(c - 1) * 5 + j, 2] = cy + 0.05 * randn(rng_data)
        end
    end
    truth = Int32[v for b in 1:3 for v in fill(b, 5)]
    a1, _, niter1 = ext._spectral_kmeans(X, 3, MersenneTwister(7))
    a2, _, _ = ext._spectral_kmeans(X, 3, MersenneTwister(7))
    @test gref_same_partition(a1, truth)
    @test a1 == a2
    @test 1 <= niter1 <= 300
    # k-means++ seeding alone picks exactly k distinct centers here: with
    # well-separated clusters every seed lands in a fresh cluster.
    seeds = ext._spectral_plusplus_seeds(X, 3, MersenneTwister(7))
    @test length(unique(seeds)) == 3
    @test length(Set([Tuple(X[s, :]) for s in seeds])) == 3
    # Empty clusters are allowed: on three identical points with k = 2 every
    # point ties at distance 0 and goes to cluster 1, leaving 2 unused.
    Xdup = zeros(3, 2)
    adup, _, _ = ext._spectral_kmeans(Xdup, 2, MersenneTwister(7))
    @test adup == [1, 1, 1]
    @test_throws ArgumentError ext._spectral_kmeans(X, 0, MersenneTwister(7))
    @test_throws ArgumentError ext._spectral_kmeans(X, 16, MersenneTwister(7))

    # 5b. Eigengap unit cases (classical semantics): hand-made eigenvalue
    # vectors pin `_spectral_auto_k` — 3-node path λ = [0, 0.5, 1.5] gives
    # k = 2 via gap g_2; an exact tie picks the smallest i; [0.0] gives 1;
    # and a 40-node synthetic strictly increasing λ with the largest gap at
    # position 25 honors the kmax cap (k = min(n′−1, 25) = 25).
    @test ext._spectral_auto_k([0.0, 0.5, 1.5], 3) == 2
    @test ext._spectral_auto_k([0.0], 1) == 1
    @test ext._spectral_auto_k([0.0, 5.0], 2) == 1
    @test ext._spectral_auto_k([0.0, 1.0, 2.0, 3.0], 4) == 1 # tie 1.0 == 1.0 == 1.0 → smallest i
    let
        lam = zeros(40)
        for i in 2:40
            lam[i] = lam[i - 1] + 1.0
        end
        lam[26] += 100.0
        for i in 27:40
            lam[i] += 100.0
        end
        @test ext._spectral_auto_k(lam, 40) == 25
    end

    # 5c. k-means++ D²-seeding distribution: 1-D points x = [0, 1, 2],
    # k = 2, over s in 1:1000 with MersenneTwister(s) the second chosen
    # center has empirical marginals ≈ [0.43333, 0.13333, 0.43333]
    # (atol 0.05). Hand computation: first center uniform (1/3 each);
    # conditional second-center probabilities from squared distances are
    # c1=1 → (0, 1/5, 4/5), c1=2 → (1/2, 0, 1/2), c1=3 → (4/5, 1/5, 0);
    # marginals (1/3)(0 + 1/2 + 4/5) = 0.4333…, (1/3)(1/5 + 0 + 1/5) =
    # 0.1333…, (1/3)(4/5 + 1/2 + 0) = 0.4333…. Uniform seeding would give
    # (1/3, 1/3, 1/3) and fails at this tolerance with 1000 samples.
    let
        X1 = reshape([0.0, 1.0, 2.0], 3, 1)
        counts = zeros(3)
        for s in 1:1000
            seeds = ext._spectral_plusplus_seeds(X1, 2, MersenneTwister(s))
            counts[seeds[2]] += 1
        end
        freq = counts ./ 1000
        @test isapprox(freq, [0.43333, 0.13333, 0.43333]; atol = 0.05)
    end

    # 6. Invariants and edge cases.
    # Isolates: GRAPH_PLANT_Z plus a 7th isolated node — the isolate is its
    # own community, the non-isolates still recover the planted blocks.
    Wiso7 = zeros(7, 7)
    Wiso7[1:6, 1:6] = GRAPH_PLANT_Z
    giso7 = mrio_graph(copy(Wiso7), gref_nodes(7))
    riso7 = communities(giso7; algorithm = :spectral, ncommunities = 2, seed = 5)
    @test count(==(riso7.membership[7]), riso7.membership) == 1
    @test gref_same_partition(riso7.membership[1:6], GRAPH_PLANT_MEMB)
    riso7a = communities(giso7; algorithm = :spectral, seed = 5)
    @test count(==(riso7a.membership[7]), riso7a.membership) == 1
    @test length(unique(riso7a.membership)) == 3
    @test gref_same_partition(riso7a.membership[1:6], GRAPH_PLANT_MEMB)
    # n == 1 (± self-loop) → [1] without any eigen work.
    @test communities(mrio_graph(zeros(1, 1), gref_nodes(1)); algorithm = :spectral).membership == Int32[1]
    g1sl = mrio_graph(reshape([2.0], 1, 1), gref_nodes(1); self_loops = true)
    @test communities(g1sl; algorithm = :spectral).membership == Int32[1]
    # All-zero matrix → singletons.
    gzero = mrio_graph(zeros(4, 4), gref_nodes(4))
    @test communities(gzero; algorithm = :spectral, seed = 1).membership == Int32[1, 2, 3, 4]
    # ncommunities = 5 > n′ = 4 clamps without error: two disjoint pairs,
    # each node its own cluster with contiguous 1:4 labels.
    W4 = [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
    g4 = mrio_graph(copy(W4), gref_nodes(4))
    r4 = communities(g4; algorithm = :spectral, ncommunities = 5, seed = 1)
    @test sort(r4.membership) == Int32[1, 2, 3, 4]
    # Determinism with a fixed seed; best-of-nruns keeps a valid partition.
    gd = mrio_graph(copy(GRAPH_PLANT_Z), gref_nodes(6))
    ra = communities(gd; algorithm = :spectral, ncommunities = 2, seed = 5)
    rb = communities(gd; algorithm = :spectral, ncommunities = 2, seed = 5)
    @test ra.membership == rb.membership
    best3 = communities(gd; algorithm = :spectral, ncommunities = 2, seed = 5, nruns = 3)
    @test length(best3.membership) == 6
    @test best3.membership == ext._relabel_first_appearance(best3.membership)
    @test best3.modularity ≈ ext._partition_modularity(gd, best3.membership, best3.resolution)
    # Float32 input: the wrapped matrix is never converted, the planted
    # blocks are still recovered in both directednesses.
    for direction in (:directed, :undirected)
        g32 = mrio_graph(Float32.(GRAPH_PLANT_Z), gref_nodes(6); direction = direction)
        @test g32.weights isa Matrix{Float32}
        r32 = communities(g32; algorithm = :spectral, ncommunities = 2, seed = 5)
        @test gref_same_partition(r32.membership, GRAPH_PLANT_MEMB)
    end
end

@testset "Compact weights" begin
    # Sparse weight input is stored zero-copy and reads through the whole
    # existing API surface exactly like its dense equivalent.
    Wsp = SparseArrays.sparse(Float32[0 2 0; 1 0 3; 0 0 0])
    gs = mrio_graph(Wsp, gref_nodes(3))
    @test gs.weights === Wsp
    @test Graphs.weights(gs)[1, 2] == 2
    @test Graphs.weights(gs)[2, 1] == 1
    gu = mrio_graph(Wsp, gref_nodes(3); direction = :undirected)
    @test Graphs.weights(gu)[1, 2] == 3
    gd = mrio_graph(Matrix(Wsp), gref_nodes(3); direction = :undirected)
    @test Graphs.ne(gu) == Graphs.ne(gd)
    @test graph_summary(gu).total_weight == graph_summary(gd).total_weight
    @test nrow(graph_summary(gs)) == 1
    gd_dir = mrio_graph(Matrix(Wsp), gref_nodes(3))
    @test pagerank_scores(gs).data ≈ pagerank_scores(gd_dir).data
    rs = communities(gs; algorithm = :label_propagation, seed = 1)
    rd = communities(gd_dir; algorithm = :label_propagation, seed = 1)
    @test rs.modularity ≈ rd.modularity
    simple_s, _ = to_simple_graph(gs)
    @test Graphs.nv(simple_s) == 3

    # Sparse-wrapped similarity ≡ dense similarity bit-for-bit: the panel
    # loops read `W[i, j]` elementwise on whatever `weights` holds, so the
    # sparse and dense forms of `Wsp` must give identical kNN results.
    Wsp_dense = Matrix(Wsp)
    gd_sim = mrio_graph(Wsp_dense, gref_nodes(3))
    for method in (:cosine, :jaccard)
        rsparse = node_similarity(gs; method = method, k = 2)
        rdense = node_similarity(gd_sim; method = method, k = 2)
        @test SparseArrays.nnz(rsparse.weights) == SparseArrays.nnz(rdense.weights)
        srows, scols, svals = SparseArrays.findnz(rsparse.weights)
        drows, dcols, dvals = SparseArrays.findnz(rdense.weights)
        @test srows == drows
        @test scols == dcols
        @test svals == dvals
    end
    rsparse_rw = node_similarity(gs; method = :random_walk, sources = [1], k = 2)
    rdense_rw = node_similarity(gd_sim; method = :random_walk, sources = [1], k = 2)
    @test SparseArrays.nnz(rsparse_rw.weights) == SparseArrays.nnz(rdense_rw.weights)
    srows_rw, scols_rw, svals_rw = SparseArrays.findnz(rsparse_rw.weights)
    drows_rw, dcols_rw, dvals_rw = SparseArrays.findnz(rdense_rw.weights)
    @test srows_rw == drows_rw
    @test scols_rw == dcols_rw
    @test svals_rw == dvals_rw
    gsparse_sym = similarity_graph(gs; symmetrize = :max)
    gdense_sym = similarity_graph(gd_sim; symmetrize = :max)
    @test SparseArrays.nnz(gsparse_sym.weights) == SparseArrays.nnz(gdense_sym.weights)
    srows_s, scols_s, svals_s = SparseArrays.findnz(gsparse_sym.weights)
    drows_s, dcols_s, dvals_s = SparseArrays.findnz(gdense_sym.weights)
    @test srows_s == drows_s
    @test scols_s == dcols_s
    @test svals_s == dvals_s

    # Same sparse ≡ dense contract on a larger seeded Float32 sparse matrix
    # with sprinkled exact zeros — non-vacuous: every result below carries
    # edges, so the bit-for-bit equality is exercised on real values.
    Wsr = rand(MersenneTwister(20260924), Float32, 10, 10)
    Wsr[rand(MersenneTwister(20260925), 10, 10) .< 0.3] .= 0.0f0
    Wsr_sp = SparseArrays.sparse(Wsr)
    gsr_sp = mrio_graph(Wsr_sp, gref_nodes(10))
    gsr_de = mrio_graph(Matrix(Wsr_sp), gref_nodes(10))
    for method in (:cosine, :jaccard)
        rsparse = node_similarity(gsr_sp; method = method, k = 2)
        rdense = node_similarity(gsr_de; method = method, k = 2)
        @test SparseArrays.nnz(rsparse.weights) > 0
        @test SparseArrays.nnz(rsparse.weights) == SparseArrays.nnz(rdense.weights)
        srows, scols, svals = SparseArrays.findnz(rsparse.weights)
        drows, dcols, dvals = SparseArrays.findnz(rdense.weights)
        @test srows == drows
        @test scols == dcols
        @test svals == dvals
    end
    rsparse_rw = node_similarity(gsr_sp; method = :random_walk, sources = [1], k = 2)
    rdense_rw = node_similarity(gsr_de; method = :random_walk, sources = [1], k = 2)
    @test SparseArrays.nnz(rsparse_rw.weights) > 0
    @test SparseArrays.nnz(rsparse_rw.weights) == SparseArrays.nnz(rdense_rw.weights)
    srows_rw, scols_rw, svals_rw = SparseArrays.findnz(rsparse_rw.weights)
    drows_rw, dcols_rw, dvals_rw = SparseArrays.findnz(rdense_rw.weights)
    @test srows_rw == drows_rw
    @test scols_rw == dcols_rw
    @test svals_rw == dvals_rw
    gsparse_sym = similarity_graph(gsr_sp; symmetrize = :max)
    gdense_sym = similarity_graph(gsr_de; symmetrize = :max)
    @test SparseArrays.nnz(gsparse_sym.weights) > 0
    @test SparseArrays.nnz(gsparse_sym.weights) == SparseArrays.nnz(gdense_sym.weights)
    srows_s, scols_s, svals_s = SparseArrays.findnz(gsparse_sym.weights)
    drows_s, dcols_s, dvals_s = SparseArrays.findnz(gdense_sym.weights)
    @test srows_s == drows_s
    @test scols_s == dcols_s
    @test svals_s == dvals_s
end

# Naive dense reference for profile similarity (plan test 5a device): the
# effective matrix via the plain gref_* helpers (filters and self-loops
# included), the full similarity matrix by plain double loops over explicit
# profiles, then top-k with the pinned tie rule (self excluded,
# strictly positive only, descending value with ties to the smaller index,
# first k). Never touches the implementation's kernels.
function gref_similarity_A(W::AbstractMatrix, directed::Bool, tau::Float64, self_loops::Bool)
    n = size(W, 1)
    wfun = directed ? gref_directed_w : gref_undirected_w
    return [wfun(W, i, j, tau, self_loops) for i in 1:n, j in 1:n]
end

function gref_similarity_ref(
        W::AbstractMatrix,
        directed::Bool,
        tau::Float64,
        self_loops::Bool,
        method::Symbol,
        on::Symbol,
        k::Int,
    )
    n = size(W, 1)
    A = gref_similarity_A(W, directed, tau, self_loops)
    function profile(i::Int)
        if on === :out
            return Vector{Float64}(A[i, :])
        elseif on === :in
            return Vector{Float64}(A[:, i])
        else
            return vcat(Vector{Float64}(A[i, :]), Vector{Float64}(A[:, i]))
        end
    end
    S = zeros(n, n)
    for i in 1:n, j in 1:n
        pi = profile(i)
        pj = profile(j)
        if method === :cosine
            ni = sqrt(sum(abs2, pi))
            nj = sqrt(sum(abs2, pj))
            S[i, j] = (ni == 0.0 || nj == 0.0) ? 0.0 : dot(pi, pj) / (ni * nj)
        else
            bi = pi .!= 0.0
            bj = pj .!= 0.0
            inter = count(bi .& bj)
            u = count(bi) + count(bj) - inter
            S[i, j] = u == 0 ? 0.0 : inter / u
        end
    end
    keff = min(k, n - 1)
    edges = Dict{Tuple{Int, Int}, Float64}()
    for i in 1:n
        cands = [(S[i, j], j) for j in 1:n if j != i && S[i, j] > 0.0]
        sort!(cands; lt = (a, b) -> a[1] > b[1] || (a[1] == b[1] && a[2] < b[2]))
        for t in 1:min(keff, length(cands))
            edges[(i, cands[t][2])] = cands[t][1]
        end
    end
    return edges
end

# Dense linear-solve reference for personalized PageRank (random-walk
# device): transition rows from the effective matrix (dangling rows map to
# the teleport vector), then `(I - d·P̃') p = (1 - d)·δ_s` solved directly —
# an independent algorithm from the power iteration.
function gref_ppr(
        W::AbstractMatrix,
        directed::Bool,
        tau::Float64,
        self_loops::Bool,
        s::Int,
        damping::Float64,
    )
    n = size(W, 1)
    A = gref_similarity_A(W, directed, tau, self_loops)
    strengths = vec(sum(A; dims = 2))
    Pt = zeros(n, n)
    for i in 1:n
        if strengths[i] == 0.0
            Pt[i, s] = 1.0
        else
            Pt[i, :] = A[i, :] ./ strengths[i]
        end
    end
    rhs = [(j == s ? 1.0 : 0.0) for j in 1:n]
    return (I - damping * transpose(Pt)) \ ((1.0 - damping) * rhs)
end

# Function-barrier allocation probe for one cosine node_similarity call.
# Kept at top level so `@allocated` measures the call itself rather than
# testset-scope variable capture.
function gref_nsim_alloc(g)
    return @allocated node_similarity(g; method = :cosine, k = 10)
end

@testset "Similarity" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    @test ext !== nothing

    # 5a. Blocked top-k matches the naive dense reference: edge SETS and
    # values across methods x profiles x directedness x filters, on seeded
    # random non-negative matrices with sprinkled exact zeros (jaccard
    # support variety) and, for the self-loop config, a nonzero diagonal.
    sim_filter_cfgs = [(0.0, 0.0, false), (0.5, 0.0, false), (0.0, 0.05, false), (0.25, 0.0, true)]
    for n in (12, 40)
        for (method, on) in Iterators.product((:cosine, :jaccard), (:out, :in, :both))
            for directed in (true, false)
                direction = directed ? :directed : :undirected
                for (threshold, min_share, self_loops) in sim_filter_cfgs
                    rng = MersenneTwister(4000 + n + 100 * Int(method === :jaccard) + 10 * Int(on === :in) + Int(on === :both))
                    W = rand(rng, n, n)
                    W[rand(rng, n, n) .< 0.25] .= 0.0
                    if self_loops
                        for i in 1:n
                            W[i, i] = 0.5 + rand(rng)
                        end
                    end
                    tau = gref_tau(W, threshold, min_share)
                    g = mrio_graph(
                        copy(W),
                        gref_nodes(n);
                        direction = direction,
                        threshold = threshold,
                        min_share = min_share,
                        self_loops = self_loops,
                    )
                    k = 3
                    ref = gref_similarity_ref(W, directed, tau, self_loops, method, on, k)
                    res = node_similarity(g; method = method, on = on, k = k)
                    got_rows, got_cols, got_vals = SparseArrays.findnz(res.weights)
                    got = Dict{Tuple{Int, Int}, Float64}(
                        (Int(got_rows[t]), Int(got_cols[t])) => Float64(got_vals[t]) for t in eachindex(got_vals)
                    )
                    @test Set(keys(ref)) == Set(keys(got))
                    maxdiff = 0.0
                    for e in keys(ref)
                        maxdiff = max(maxdiff, abs(got[e] - ref[e]))
                    end
                    @test maxdiff <= 1.0e-6
                end
            end
        end
    end

    # 5b. kNN symmetrization rules, unit-tested on a hand-built directed kNN
    # matrix: S[1, 2] = 0.5 / S[2, 1] = 0.3 (both directions) and one-sided
    # S[1, 3] = 0.7. :max keeps 0.5, :mean gives (0.5 + 0.3) / 2 = 0.4; the
    # one-sided pair keeps 0.7 under both rules. Storage is one-sided
    # upper-triangular (lower entries read 0).
    Sk = SparseArrays.sparse(Int32[1, 2, 1], Int32[2, 1, 3], Float32[0.5, 0.3, 0.7], 3, 3)
    @test Sk isa SparseArrays.SparseMatrixCSC{Float32, Int32}
    Mmax = ext._symmetrize_knn(Sk, 3, :max)
    @test Mmax isa SparseArrays.SparseMatrixCSC{Float32, Int32}
    @test Mmax[1, 2] ≈ 0.5f0
    @test Mmax[2, 1] == 0.0f0
    @test Mmax[1, 3] ≈ 0.7f0
    @test Mmax[3, 1] == 0.0f0
    Mmean = ext._symmetrize_knn(Sk, 3, :mean)
    @test Mmean[1, 2] ≈ 0.4f0
    @test Mmean[2, 1] == 0.0f0
    @test Mmean[1, 3] ≈ 0.7f0

    # 5b end to end with hand-computed cosine values. W rows (directed, no
    # filter): p1 = [0, 3, 4] (norm 5), p2 = [0, 0, 5] (norm 5),
    # p3 = [0, 0, 0] (norm 0). S[1, 2] = S[2, 1] = (3*0 + 4*5) / 25 = 0.8;
    # everything touching node 3 is 0. k = 2 keeps 1 <-> 2 only.
    Wh = [0.0 3.0 4.0; 0.0 0.0 5.0; 0.0 0.0 0.0]
    gh = mrio_graph(copy(Wh), gref_nodes(3))
    rh = node_similarity(gh; method = :cosine, on = :out, k = 2)
    @test rh.weights[1, 2] ≈ 0.8f0
    @test rh.weights[2, 1] ≈ 0.8f0
    @test SparseArrays.nnz(rh.weights) == 2
    for symmetrize in (:max, :mean)
        gsim = similarity_graph(gh; method = :cosine, on = :out, k = 2, symmetrize = symmetrize)
        @test Graphs.is_directed(gsim) == false
        @test gsim.weights isa SparseArrays.SparseMatrixCSC{Float32, Int32}
        # Pair-weight reads: the FilteredWeights view and the raw one-sided
        # upper-triangular storage both give s = 0.8.
        @test Graphs.weights(gsim)[1, 2] ≈ 0.8
        @test gsim.weights[1, 2] + gsim.weights[2, 1] ≈ 0.8f0
        @test gsim.weights[1, 2] ≈ 0.8f0
        @test gsim.weights[2, 1] == 0.0f0
        @test Graphs.ne(gsim) == 1
    end

    # Top-k edge cases.
    # k >= n - 1 clamps: identical rows ([1, 1, 1] with self_loops = true,
    # so the kept diagonal joins the profiles) give cosine 1.0 for every
    # off-diagonal pair, so k = 100 keeps all 6 edges.
    Wcl = [1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0]
    gcl = mrio_graph(copy(Wcl), gref_nodes(3); self_loops = true)
    rcl = node_similarity(gcl; k = 100)
    @test SparseArrays.nnz(rcl.weights) == 6
    for i in 1:3, j in 1:3
        i == j && continue
        @test rcl.weights[i, j] ≈ 1.0f0
    end
    # An all-zero row (isolated profile) gets no out-edges: row 3 of
    # [0 1 1; 1 0 1; 0 0 0] has norm 0, so row 3 emits nothing (and nothing
    # points at it either, since every S[i, 3] needs its norm).
    Wz = [0.0 1.0 1.0; 1.0 0.0 1.0; 0.0 0.0 0.0]
    gz = mrio_graph(copy(Wz), gref_nodes(3))
    rz = node_similarity(gz; k = 2)
    @test all(rz.weights[3, :] .== 0.0f0)
    @test rz.weights[1, 2] ≈ 0.5f0
    # Self-similarity never appears as an edge, and the exact tie resolves
    # to the smaller index. n = 4, self_loops = false: rows 2 and 3 are
    # both [2, 0, 0, 3] off-diagonally, so p2 == p3 == [2, 0, 0, 3]
    # (norm √13); p1 = [0, 1, 0, 1] (norm √2). S[1, 2] == S[1, 3] ==
    # 3 / √(2·13) ≈ 0.588 (bitwise identical dots), S[2, 2] would be 1.0.
    # k = 1 keeps (1, 2) from row 1 and (2, 3) from row 2 — never (2, 2).
    Wt = [0.0 1.0 0.0 1.0; 2.0 0.0 0.0 3.0; 2.0 0.0 0.0 3.0; 0.0 0.0 0.0 0.0]
    gt = mrio_graph(copy(Wt), gref_nodes(4))
    rt = node_similarity(gt; k = 1)
    @test rt.weights[2, 2] == 0.0f0
    @test rt.weights[1, 2] ≈ Float32(3 / sqrt(26))
    @test rt.weights[1, 3] == 0.0f0
    @test rt.weights[2, 3] ≈ 1.0f0

    # Heap boundary tie with k >= 2 (k = 3): rows 2-5 are raw-identical
    # ([1, 5, 0, 3, 0], self_loops = true so the diagonal joins the
    # profiles), hence p2 == p3 == p4 == p5 bitwise and all four cosine
    # scores to row 1 (p1 = [0, 1, 0, 1, 0], dot 8 > 0) are exactly equal.
    # The tie rule "smaller j first" must keep {2, 3, 4}: candidate 5 ties
    # the heap minimum but its larger index never dislodges a smaller one.
    Wtie = [0.0 1.0 0.0 1.0 0.0; 1.0 5.0 0.0 3.0 0.0; 1.0 5.0 0.0 3.0 0.0; 1.0 5.0 0.0 3.0 0.0; 1.0 5.0 0.0 3.0 0.0]
    gtie = mrio_graph(copy(Wtie), gref_nodes(5); self_loops = true)
    rtie = node_similarity(gtie; method = :cosine, k = 3)
    kept1 = sort([j for j in 1:5 if rtie.weights[1, j] != 0.0f0])
    @test kept1 == [2, 3, 4]
    @test rtie.weights[1, 5] == 0.0f0
    @test rtie.weights[1, 2] == rtie.weights[1, 3] == rtie.weights[1, 4]

    # Result contracts.
    Wc = rand(MersenneTwister(77), 6, 6)
    gc = mrio_graph(copy(Wc), gref_nodes(6))
    rc = node_similarity(gc; k = 3)
    @test rc.nodes === gc.nodes
    @test rc.weights isa SparseArrays.SparseMatrixCSC{Float32, Int32}
    @test SparseArrays.nnz(rc.weights) <= 3 * 6
    @test Graphs.is_directed(rc) == true
    @test Graphs.nv(rc) == 6
    sc = similarity_graph(gc; k = 3)
    @test sc.nodes === gc.nodes
    @test sc.weights isa SparseArrays.SparseMatrixCSC{Float32, Int32}
    @test Graphs.is_directed(sc) == false
    # n == 1 gives an empty result (k clamps to 0).
    g1 = mrio_graph(reshape([2.0], 1, 1), gref_nodes(1))
    r1 = node_similarity(g1)
    @test Graphs.ne(r1) == 0
    @test size(r1.weights) == (1, 1)
    @test Graphs.ne(similarity_graph(g1)) == 0
    # Full determinism: no RNG in the implementation, two runs identical.
    rc2 = node_similarity(gc; method = :jaccard, on = :both, k = 3)
    rc3 = node_similarity(gc; method = :jaccard, on = :both, k = 3)
    @test rc2.weights == rc3.weights
    @test similarity_graph(gc; k = 3).weights == similarity_graph(gc; k = 3).weights
    # MRIO method builds mrio_graph and forwards (default :directed).
    mrio = gref_plant_mrio()
    rm = node_similarity(mrio; k = 2)
    rg = node_similarity(mrio_graph(mrio); k = 2)
    @test rm.weights == rg.weights
    @test rm.nodes === mrio.T.row_indices
    @test node_similarity(mrio; threshold = 5.0, k = 2).weights ==
        node_similarity(mrio_graph(mrio; threshold = 5.0); k = 2).weights
    sm = similarity_graph(mrio; k = 2)
    @test sm.weights == similarity_graph(mrio_graph(mrio); k = 2).weights
    @test Graphs.is_directed(sm) == false

    # Interop: similarity_graph feeds communities (plan: the reusable
    # builder). Two 4-node blocks with strong within-block flows plus weak
    # noise; k = 3 keeps the block neighbors.
    rng_b = MersenneTwister(2026)
    Wb = 0.05 .* rand(rng_b, 8, 8)
    Wb[1:4, 1:4] .+= 4.0
    Wb[5:8, 5:8] .+= 4.0
    for i in 1:8
        Wb[i, i] = 0.0
    end
    gb = mrio_graph(copy(Wb), gref_nodes(8))
    simb = similarity_graph(gb; k = 3)
    resb = communities(simb; algorithm = :louvain, seed = 2)
    @test resb isa CommunityResult
    @test length(resb.membership) == 8
    @test resb.modularity ≈ ext._partition_modularity(simb, resb.membership, resb.resolution)

    # Validation.
    @test_throws ArgumentError node_similarity(gc; method = :bogus)
    @test_throws ArgumentError node_similarity(gc; on = :bogus)
    @test_throws ArgumentError node_similarity(gc; method = :random_walk, on = :bogus)
    @test_throws ArgumentError similarity_graph(gc; symmetrize = :bogus)
    @test_throws ArgumentError node_similarity(gc; k = true)
    @test_throws ArgumentError node_similarity(gc; k = big(2)^80)
    @test_throws ArgumentError node_similarity(gc; k = 0)
    @test_throws ArgumentError node_similarity(gc; k = -2)
    @test_throws ArgumentError node_similarity(gc; damping = 0.0)
    @test_throws ArgumentError node_similarity(gc; damping = 1.0)
    @test_throws ArgumentError node_similarity(gc; tol = 0.0)
    @test_throws ArgumentError node_similarity(gc; max_iter = 0)
    @test_throws ArgumentError node_similarity(gc; method = :cosine, sources = [1])
    @test_throws ArgumentError node_similarity(
        mrio_graph(zeros(0, 0), DataFrame(CountryCode = String[], Sector = String[])),
    )
    # NaN/negative effective weights are rejected; the message names the
    # entry point and the offending pair. NaN sits at the retained
    # off-diagonal (1, 2): a diagonal NaN with the default
    # self_loops = false is dropped by the self-loop rule everywhere
    # (mirroring the PageRank/Community tests).
    @test_throws ArgumentError node_similarity(mrio_graph([0.0 NaN; 0.0 0.0], gref_nodes(2)))
    @test_throws ArgumentError node_similarity(mrio_graph([-1.0 -1.0; 0.0 0.0], gref_nodes(2)))
    gnan_err = try
        node_similarity(mrio_graph([0.0 NaN; 0.0 0.0], gref_nodes(2)))
        nothing
    catch e
        e
    end
    @test gnan_err isa ArgumentError
    @test occursin("node_similarity", sprint(showerror, gnan_err))
    @test occursin("w(1, 2)", sprint(showerror, gnan_err))
    @test_throws ArgumentError similarity_graph(mrio_graph([0.0 NaN; 0.0 0.0], gref_nodes(2)))
    gsim_nan_err = try
        similarity_graph(mrio_graph([0.0 NaN; 0.0 0.0], gref_nodes(2)))
        nothing
    catch e
        e
    end
    @test gsim_nan_err isa ArgumentError
    @test occursin("similarity_graph", sprint(showerror, gsim_nan_err))
    @test occursin("w(1, 2)", sprint(showerror, gsim_nan_err))
    @test_throws ErrorException node_similarity(42)

    # Memory guard (R6): a cosine run on a seeded n = 2000 random
    # non-negative matrix with sprinkled exact zeros must stay far below
    # any n x n scratch (32 MiB at n = 2000 for Float64); expected ~3 MB
    # (block scratch + O(k*n) sparse result).
    Wbig = rand(MersenneTwister(123), 2000, 2000)
    Wbig[rand(MersenneTwister(124), 2000, 2000) .< 0.1] .= 0.0
    gbig = mrio_graph(Wbig, gref_nodes(2000))
    node_similarity(gbig; method = :cosine, k = 10) # warm-up (compilation)
    @test gref_nsim_alloc(gbig) < 12 * 2^20
    rbig = node_similarity(gbig; method = :cosine, k = 10)
    @test SparseArrays.nnz(rbig.weights) <= 10 * 2000
end

@testset "Random walk" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    @test ext !== nothing

    # (a) Hand-computed PPR on the 2-node graph W = [0 1; 2 0] (directed,
    # damping = 0.5). Row strengths s = [1, 2]; row-normalized transitions
    # P = [0 1; 1 0]. Fixed point p = (1 - d)·δ_s + d·P'·p with d = 0.5:
    # source 1: p1 = 0.5 + 0.5·p2, p2 = 0.5·p1 → p = [2/3, 1/3];
    # source 2: p1 = 0.5·p2, p2 = 0.5 + 0.5·p1 → p = [1/3, 2/3].
    W2 = [0.0 1.0; 2.0 0.0]
    g2 = mrio_graph(copy(W2), gref_nodes(2))
    p1 = ext._weighted_pagerank(g2, 0.5, [1.0, 0.0], 1.0e-14, 100_000)
    @test p1 ≈ [2 / 3, 1 / 3] atol = 1.0e-10
    p2 = ext._weighted_pagerank(g2, 0.5, [0.0, 1.0], 1.0e-14, 100_000)
    @test p2 ≈ [1 / 3, 2 / 3] atol = 1.0e-10
    r1 = node_similarity(g2; method = :random_walk, sources = [1], damping = 0.5, tol = 1.0e-14, max_iter = 100_000, k = 1)
    # Top-k of p_1 excluding the source itself (p_1[1] = 2/3 is the max but
    # never becomes an edge): the single edge 1 -> 2 carries 1/3.
    @test SparseArrays.nnz(r1.weights) == 1
    @test r1.weights[1, 2] ≈ Float32(1 / 3) rtol = 1.0e-5
    @test r1.weights[1, 1] == 0.0f0
    r2 = node_similarity(g2; method = :random_walk, sources = [2], damping = 0.5, tol = 1.0e-14, max_iter = 100_000, k = 1)
    @test r2.weights[2, 1] ≈ Float32(1 / 3) rtol = 1.0e-5
    @test SparseArrays.nnz(r2.weights) == 1

    # (b) Dangling-node case: W = [0 2 0; 0 0 3; 0 0 0], source 1,
    # damping = 0.5. Strengths [2, 3, 0]; node 3 is dangling with mass
    # m_d = p3 feeding δ_1. Fixed-point system:
    # p1 = 0.5·(1 + m_d), p2 = 0.5·p1, p3 = 0.5·p2, m_d = p3, so
    # p1 = 0.5·(1 + p1/4) → p = [4/7, 2/7, 1/7] (sums to 1).
    Wd = [0.0 2.0 0.0; 0.0 0.0 3.0; 0.0 0.0 0.0]
    gd = mrio_graph(copy(Wd), gref_nodes(3))
    pd = ext._weighted_pagerank(gd, 0.5, [1.0, 0.0, 0.0], 1.0e-14, 100_000)
    @test pd ≈ [4 / 7, 2 / 7, 1 / 7] atol = 1.0e-10
    rd = node_similarity(gd; method = :random_walk, sources = [1], damping = 0.5, tol = 1.0e-14, max_iter = 100_000, k = 1)
    @test rd.weights[1, 2] ≈ Float32(2 / 7) rtol = 1.0e-5
    @test SparseArrays.nnz(rd.weights) == 1
    rd2 = node_similarity(gd; method = :random_walk, sources = [1], damping = 0.5, tol = 1.0e-14, max_iter = 100_000, k = 2)
    @test rd2.weights[1, 2] ≈ Float32(2 / 7) rtol = 1.0e-5
    @test rd2.weights[1, 3] ≈ Float32(1 / 7) rtol = 1.0e-5

    # Cross-check against the dense linear-solve reference on seeded random
    # non-negative matrices (both directednesses, self-loops on/off): the
    # internal iterate and the top-k edges of the public result.
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        for self_loops in (false, true)
            rng = MersenneTwister(900 + 10 * Int(directed) + Int(self_loops))
            n = 9
            W = rand(rng, n, n)
            W[rand(rng, n, n) .< 0.3] .= 0.0
            if self_loops
                for i in 1:n
                    W[i, i] = 0.5 + rand(rng)
                end
            end
            g = mrio_graph(copy(W), gref_nodes(n); direction = direction, self_loops = self_loops)
            for s in (1, 5, 9)
                pref = gref_ppr(W, directed, 0.0, self_loops, s, 0.5)
                pgot = ext._weighted_pagerank(g, 0.5, [j == s ? 1.0 : 0.0 for j in 1:n], 1.0e-14, 100_000)
                @test pgot ≈ pref atol = 1.0e-9
                @test sum(pgot) ≈ 1.0 atol = 1.0e-9
                r = node_similarity(g; method = :random_walk, sources = [s], damping = 0.5, tol = 1.0e-14, max_iter = 100_000, k = 3)
                cands = [(pref[j], j) for j in 1:n if j != s && pref[j] > 0.0]
                sort!(cands; lt = (a, b) -> a[1] > b[1] || (a[1] == b[1] && a[2] < b[2]))
                keept = first(cands, min(3, length(cands)))
                frw = SparseArrays.findnz(r.weights)
                got_pairs = Set((Int(i), Int(j)) for (i, j) in zip(frw[1], frw[2]))
                @test Set([(s, c[2]) for c in keept]) == got_pairs
                for c in keept
                    @test r.weights[s, c[2]] ≈ Float32(c[1]) rtol = 1.0e-5
                end
            end
        end
    end

    # (c) sources validation: required with :random_walk, rejected with
    # :cosine/:jaccard, out-of-range rejected, duplicates deduped.
    @test_throws ArgumentError node_similarity(g2; method = :random_walk)
    @test_throws ArgumentError node_similarity(g2; method = :cosine, sources = [1])
    @test_throws ArgumentError node_similarity(g2; method = :jaccard, sources = [1])
    @test_throws ArgumentError node_similarity(g2; method = :random_walk, sources = [0])
    @test_throws ArgumentError node_similarity(g2; method = :random_walk, sources = [3])
    @test_throws ArgumentError node_similarity(g2; method = :random_walk, sources = Int[])
    @test_throws ArgumentError node_similarity(g2; method = :random_walk, sources = 1)
    @test_throws ArgumentError node_similarity(g2; method = :random_walk, sources = [true])
    @test_throws ArgumentError node_similarity(g2; method = :random_walk, sources = [big(2)^80])
    @test_throws ArgumentError similarity_graph(g2; method = :random_walk)
    # Duplicates dedupe to first occurrences: sources = [2, 1, 2] emits
    # edges from rows {1, 2} exactly like sources = [2, 1].
    W3 = [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]
    g3 = mrio_graph(copy(W3), gref_nodes(3))
    kw = (; method = :random_walk, damping = 0.5, tol = 1.0e-14, max_iter = 100_000, k = 2)
    rdup = node_similarity(g3; kw..., sources = [2, 1, 2])
    rded = node_similarity(g3; kw..., sources = [2, 1])
    @test rdup.weights == rded.weights
    @test Set(Int(i) for (i, _, _) in zip(SparseArrays.findnz(rdup.weights)...)) == Set([1, 2])
    @test all(rdup.weights[3, :] .== 0.0f0)

    # (d) Top-k excludes the source node itself even when k is ample: with
    # k = 5 on the 3-node clique-like W3 every non-source row member is
    # kept, but no self edge appears.
    rk = node_similarity(g3; kw..., sources = [1], k = 5)
    @test rk.weights[1, 1] == 0.0f0
    @test SparseArrays.nnz(rk.weights) == 2
end

# ---- Cross-network / cross-partition comparison references ----
# Independent naive references for `compare_networks`/`compare_partitions`.
# Plain loops over explicit index/item pairs built with `Dict` key matching
# (overlap/strengths/density/scale) and O(m^2) brute-force pair counting
# (ARI) plus a naive double loop over label pairs (NMI) — structurally
# unlike the production contingency-table/index-map code.

function gref_compare_key_maps(nodes1::DataFrame, nodes2::DataFrame, cols::Vector{Symbol})
    index2 = Dict{Any, Int}()
    for r in 1:nrow(nodes2)
        index2[ntuple(i -> nodes2[r, cols[i]], length(cols))] = r
    end
    map1 = Int[]
    map2 = Int[]
    for r in 1:nrow(nodes1)
        k = ntuple(i -> nodes1[r, cols[i]], length(cols))
        if haskey(index2, k)
            push!(map1, r)
            push!(map2, index2[k])
        end
    end
    return map1, map2
end

function gref_compare_ref(
        W1::AbstractMatrix,
        directed1::Bool,
        tau1::Float64,
        sl1::Bool,
        W2::AbstractMatrix,
        directed2::Bool,
        tau2::Float64,
        sl2::Bool,
        map1::Vector{Int},
        map2::Vector{Int},
    )
    m = length(map1)
    sum_min = 0.0
    sum_max = 0.0
    sout1 = zeros(m)
    sout2 = zeros(m)
    sin1 = zeros(m)
    sin2 = zeros(m)
    cnt1 = 0
    cnt2 = 0
    sc1 = 0.0
    sc2 = 0.0
    for j in 1:m, i in 1:m
        v1 = directed1 ? gref_directed_w(W1, map1[i], map1[j], tau1, sl1) :
            gref_undirected_w(W1, map1[i], map1[j], tau1, sl1)
        v2 = directed2 ? gref_directed_w(W2, map2[i], map2[j], tau2, sl2) :
            gref_undirected_w(W2, map2[i], map2[j], tau2, sl2)
        sum_min += min(v1, v2)
        sum_max += max(v1, v2)
        sout1[i] += v1
        sout2[i] += v2
        sin1[j] += v1
        sin2[j] += v2
        v1 != 0.0 && (cnt1 += 1)
        v2 != 0.0 && (cnt2 += 1)
        sc1 += v1
        sc2 += v2
    end
    return (;
        sum_min = sum_min,
        sum_max = sum_max,
        sout1 = sout1,
        sout2 = sout2,
        sin1 = sin1,
        sin2 = sin2,
        cnt1 = cnt1,
        cnt2 = cnt2,
        sc1 = sc1,
        sc2 = sc2,
    )
end

function gref_count_ranks(x::AbstractVector{<:Real})
    # O(m^2) average ranks by counting (structurally unlike the sort-based
    # `_tied_ranks`): rank[i] = #{x[j] < x[i]} + (#{x[j] == x[i]} + 1) / 2.
    n = length(x)
    r = Vector{Float64}(undef, n)
    for i in 1:n
        less = count(j -> x[j] < x[i], 1:n)
        equal = count(j -> x[j] == x[i], 1:n)
        r[i] = less + (equal + 1) / 2
    end
    return r
end

function gref_ari_nmi_ref(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    # ARI from pair counting (loop over all item pairs: `same_both`,
    # `same_1`, `same_2` give `ARI = (same_both − E) / (M − E)` with
    # `E = same_1 * same_2 / C(m, 2)`, `M = (same_1 + same_2) / 2`) and NMI
    # from a naive double loop over label pairs — O(m^2) brute force,
    # structurally unlike the production contingency code.
    m = length(a)
    @assert length(b) == m
    if m < 2
        return 1.0, 1.0
    end
    same_both = 0
    same_1 = 0
    same_2 = 0
    for i in 1:m, j in (i + 1):m
        s1 = a[i] == a[j]
        s2 = b[i] == b[j]
        s1 && (same_1 += 1)
        s2 && (same_2 += 1)
        (s1 && s2) && (same_both += 1)
    end
    P = m * (m - 1) / 2
    E = same_1 * same_2 / P
    M = (same_1 + same_2) / 2
    ari = M == E ? 1.0 : (same_both - E) / (M - E)
    I = 0.0
    H1 = 0.0
    H2 = 0.0
    for u in unique(a), v in unique(b)
        n = count(t -> a[t] == u && b[t] == v, 1:m)
        ru = count(==(u), a)
        cv = count(==(v), b)
        n > 0 && (I += (n / m) * log(n * m / (ru * cv)))
    end
    for u in unique(a)
        ru = count(==(u), a)
        H1 -= (ru / m) * log(ru / m)
    end
    for v in unique(b)
        cv = count(==(v), b)
        H2 -= (cv / m) * log(cv / m)
    end
    nmi = (H1 + H2) == 0.0 ? 1.0 : 2I / (H1 + H2)
    return Float64(ari), Float64(nmi)
end

# Function-barrier allocation probes for `compare_networks` (with and
# without the PageRank option). Kept at top level so `@allocated` measures
# the call itself rather than testset-scope variable capture.
function gref_compare_alloc(g1, g2)
    return @allocated compare_networks(g1, g2)
end

function gref_compare_pr_alloc(g1, g2)
    return @allocated compare_networks(g1, g2; pagerank = true)
end

@testset "Network comparison" begin
    ext = Base.get_extension(Juliora, :JulioraGraphsExt)
    @test ext !== nothing

    # Column contract: exact names and order without `pagerank`, two
    # appended PageRank columns with `pagerank = true`.
    base_cols = [
        "n_matched",
        "n_1",
        "n_2",
        "edge_overlap",
        "pearson_out",
        "spearman_out",
        "pearson_in",
        "spearman_in",
        "density_1",
        "density_2",
        "density_ratio",
        "scale_1",
        "scale_2",
        "scale_ratio",
    ]

    # 1. Identical inputs: two separate `mrio_graph` objects from equal
    # matrices. W = [0 2 1; 0 0 3; 1 0 0] (directed, no filter) has
    # sout = [3, 3, 1] and sin = [1, 2, 4] — nonzero strength variance, so
    # all four correlations are exactly 1.0 (Statistics.cor of a vector
    # with itself).
    WI = Float64[0 2 1; 0 0 3; 1 0 0]
    for directed in (true, false)
        direction = directed ? :directed : :undirected
        ni = gref_nodes(3)
        gi1 = mrio_graph(copy(WI), ni; direction = direction)
        gi2 = mrio_graph(copy(WI), ni; direction = direction)
        df = compare_networks(gi1, gi2)
        @test names(df) == base_cols
        @test only(df.n_matched) == 3
        @test only(df.n_1) == 3
        @test only(df.n_2) == 3
        @test only(df.edge_overlap) == 1.0
        @test only(df.pearson_out) == 1.0
        @test only(df.spearman_out) == 1.0
        @test only(df.pearson_in) == 1.0
        @test only(df.spearman_in) == 1.0
        @test only(df.density_ratio) == 1.0
        @test only(df.scale_ratio) == 1.0
        @test only(df.density_1) == only(df.density_2)
        @test only(df.scale_1) == only(df.scale_2)
        @test eltype(df.n_matched) == Int
        @test eltype(df.edge_overlap) == Float64
        # Identical graphs through the PageRank path: bitwise-identical
        # score vectors, so the PageRank correlations are exactly 1.0 too.
        dfpr = compare_networks(gi1, gi2; pagerank = true)
        @test names(dfpr) == [base_cols; "pearson_pagerank"; "spearman_pagerank"]
        @test only(dfpr.pearson_pagerank) == 1.0
        @test only(dfpr.spearman_pagerank) == 1.0
        @test only(dfpr.edge_overlap) == 1.0
    end

    # 4. Hand-computed intersection case. Nodes share C1/C2/C3 (CX/CY are
    # unmatched), directed graphs with defaults (no filter, self_loops =
    # false, so diagonals are dropped).
    #   W1 = [0 2 0 9; 0 0 1 8; 4 0 0 7; 6 5 3 0]
    #   W2 = [0 2 3 1; 0 0 1 2; 4 0 0 0; 0 0 9 0]
    # Matched 3x3 blocks (rows/cols 1:3, diagonals dropped):
    #   V1 = [0 2 0; 0 0 1; 4 0 0], V2 = [0 2 3; 0 0 1; 4 0 0].
    # Nonzero ordered pairs: V1 has (1,2)=2, (2,3)=1, (3,1)=4;
    # V2 adds (1,3)=3.
    #   edge_overlap = (2 + 0 + 1 + 4) / (2 + 3 + 1 + 4) = 7/10.
    #   scale_1 = 7, scale_2 = 10, scale_ratio = 0.7.
    #   density_1 = 3/9, density_2 = 4/9, density_ratio = 3/4 = 0.75.
    #   sout_1 = [2,1,4], sout_2 = [5,1,4]:
    #     pearson_out = 11/sqrt(364) ≈ 0.57656 (Σxy form: with Σx = 7,
    #     Σy = 10, Σxy = 27, Σx^2 = 21, Σy^2 = 42:
    #     (3*27 − 70)/sqrt((63 − 49)(126 − 100)) = 11/sqrt(14*26)).
    #     spearman_out = 1/2 (ranks [2,1,3] vs [3,1,2]: Σxy = 13,
    #     (3*13 − 36)/6 = 1/2).
    #   sin_1 = [4,2,1] (V1 columns), sin_2 = [4,2,4] (V2 col 3 is
    #   3 + 1 + 0 = 4):
    #     pearson_in = 1/(2*sqrt(7)) ≈ 0.18898 (Σx = 7, Σy = 10, Σxy = 24,
    #     Σx^2 = 21, Σy^2 = 36: (72 − 70)/sqrt(14*8) = 2/sqrt(112)).
    #     spearman_in = 0.0 (ranks [3,2,1] vs [2.5,1,2.5]: Σxy = 12,
    #     numerator 3*12 − 36 = 0).
    # The unmatched row/column 4 values (9, 8, 7, 6, 5, 3, ...) appear
    # nowhere: scale_1 == 7 pins intersection semantics (any inclusion of
    # row/column 4 would add ≥ 3 to a scale).
    nodes1 = DataFrame(CountryCode = ["C1", "C2", "C3", "CX"], Sector = fill("s", 4))
    nodes2 = DataFrame(CountryCode = ["C1", "C2", "C3", "CY"], Sector = fill("s", 4))
    W1 = Float64[0 2 0 9; 0 0 1 8; 4 0 0 7; 6 5 3 0]
    W2 = Float64[0 2 3 1; 0 0 1 2; 4 0 0 0; 0 0 9 0]
    g1 = mrio_graph(copy(W1), nodes1)
    g2 = mrio_graph(copy(W2), nodes2)
    df = compare_networks(g1, g2)
    @test names(df) == base_cols
    @test only(df.n_matched) == 3
    @test only(df.n_1) == 4
    @test only(df.n_2) == 4
    @test only(df.edge_overlap) ≈ 7 / 10
    @test only(df.scale_1) ≈ 7.0
    @test only(df.scale_2) ≈ 10.0
    @test only(df.scale_ratio) ≈ 0.7
    @test only(df.density_1) ≈ 3 / 9
    @test only(df.density_2) ≈ 4 / 9
    @test only(df.density_ratio) ≈ 0.75
    @test only(df.pearson_out) ≈ 11 / sqrt(364)
    @test only(df.spearman_out) ≈ 0.5
    @test only(df.pearson_in) ≈ 1 / (2 * sqrt(7))
    @test only(df.spearman_in) ≈ 0.0 atol = 1.0e-15
    # Cross-check every column against the naive reference (explicit index
    # pairs, counting-based ranks).
    map1, map2 = gref_compare_key_maps(nodes1, nodes2, [:CountryCode, :Sector])
    @test map1 == [1, 2, 3]
    @test map2 == [1, 2, 3]
    ref = gref_compare_ref(W1, true, 0.0, false, W2, true, 0.0, false, map1, map2)
    @test only(df.edge_overlap) ≈ ref.sum_min / ref.sum_max
    @test only(df.pearson_out) ≈ Statistics.cor(ref.sout1, ref.sout2)
    @test only(df.spearman_out) ≈ Statistics.cor(gref_count_ranks(ref.sout1), gref_count_ranks(ref.sout2))
    @test only(df.pearson_in) ≈ Statistics.cor(ref.sin1, ref.sin2)
    @test only(df.spearman_in) ≈ Statistics.cor(gref_count_ranks(ref.sin1), gref_count_ranks(ref.sin2))
    @test only(df.density_1) ≈ ref.cnt1 / 9
    @test only(df.density_2) ≈ ref.cnt2 / 9
    @test only(df.density_ratio) ≈ (ref.cnt1 / 9) / (ref.cnt2 / 9)
    @test only(df.scale_1) ≈ ref.sc1
    @test only(df.scale_2) ≈ ref.sc2
    @test only(df.scale_ratio) ≈ ref.sc1 / ref.sc2

    # 5a. Reordering: g2 with row-shuffled nodes and identically permuted
    # W rows/cols gives the same metrics (keys, not positions).
    perm = [3, 1, 4, 2]
    nodes2s = nodes2[perm, :]
    W2s = W2[perm, perm]
    g2s = mrio_graph(copy(W2s), nodes2s)
    dfs = compare_networks(g1, g2s)
    @test only(dfs.n_matched) == 3
    for col in base_cols[4:end]
        @test only(dfs[!, col]) ≈ only(df[!, col])
    end

    # 5b. Multi-column composite keys: two nodes sharing CountryCode are
    # distinguished only by Sector.
    n1b = DataFrame(CountryCode = ["A", "A", "B"], Sector = ["x", "y", "x"])
    n2b = DataFrame(CountryCode = ["B", "A", "A"], Sector = ["x", "x", "y"])
    Wb = Float64[0 1 2; 3 0 4; 5 6 0]
    # n2b row order is [B/x, A/x, A/y] = n1b rows [3, 1, 2]; permute W
    # identically so the pair describes the same network.
    Wb2 = Wb[[3, 1, 2], [3, 1, 2]]
    gb1 = mrio_graph(copy(Wb), n1b)
    gb2 = mrio_graph(copy(Wb2), n2b)
    dfb = compare_networks(gb1, gb2)
    @test only(dfb.n_matched) == 3
    @test only(dfb.edge_overlap) == 1.0
    @test only(dfb.pearson_out) == 1.0
    @test only(dfb.pearson_in) == 1.0
    # A single-column key would collide on "A" and throw.
    @test_throws ArgumentError compare_networks(gb1, gb2; match = [:CountryCode])

    # 5c. Explicit-column form on a fixture where `:keys` picks more
    # columns: Tags differ across graphs, so `:keys` (CountryCode, Sector,
    # Tag) matches nothing while `match = [:CountryCode]` aligns on codes.
    n1c = DataFrame(CountryCode = ["A", "B"], Sector = ["x", "x"], Tag = ["p", "q"])
    n2c = DataFrame(CountryCode = ["B", "A"], Sector = ["x", "x"], Tag = ["Q", "P"])
    Wc = Float64[0 1; 2 0]
    gc1 = mrio_graph(copy(Wc), n1c)
    gc2 = mrio_graph(copy(Wc[[2, 1], [2, 1]]), n2c)
    dfc0 = compare_networks(gc1, gc2)
    @test only(dfc0.n_matched) == 0
    dfc = compare_networks(gc1, gc2; match = [:CountryCode])
    @test only(dfc.n_matched) == 2
    @test only(dfc.edge_overlap) == 1.0

    # 5h. Missing-safe (`isequal`) key matching: `missing` keys pair with
    # `missing` keys. nodes1 order is [C1, missing, C3]; nodes2 carries the
    # same three keys as [C3, C1, missing] with W rows/cols permuted
    # identically (perm [3, 1, 2]).
    #   Wmiss1   = [0 2 0; 0 0 1; 4 0 0] (nodes1 order),
    #   W2base   = [0 5 0; 0 0 1; 4 0 0] (nodes1 order; the (C1, missing)
    #   pair is 5 instead of 2, so the differing pair touches the
    #   missing-key node).
    # Matched blocks (diagonals dropped): V1 has (1,2)=2, (2,3)=1,
    # (3,1)=4; V2 has (1,2)=5, (2,3)=1, (3,1)=4, so
    #   edge_overlap = (2 + 1 + 4) / (5 + 1 + 4) = 7/10,
    #   scale_1 = 7, scale_2 = 10.
    # A `==`-based (non-missing-safe) matcher would drop the missing pair
    # and report n_matched == 2.
    nmiss1 = DataFrame(CountryCode = Union{String, Missing}["C1", missing, "C3"], Sector = fill("s", 3))
    nmiss2 = nmiss1[[3, 1, 2], :]
    Wmiss1 = Float64[0 2 0; 0 0 1; 4 0 0]
    Wmiss2 = (Float64[0 5 0; 0 0 1; 4 0 0])[[3, 1, 2], [3, 1, 2]]
    gmiss1 = mrio_graph(copy(Wmiss1), nmiss1)
    gmiss2 = mrio_graph(copy(Wmiss2), nmiss2)
    dfmiss = compare_networks(gmiss1, gmiss2)
    @test only(dfmiss.n_matched) == 3
    @test only(dfmiss.n_1) == 3
    @test only(dfmiss.n_2) == 3
    @test only(dfmiss.edge_overlap) ≈ 7 / 10
    @test only(dfmiss.scale_1) ≈ 7.0
    @test only(dfmiss.scale_2) ≈ 10.0
    @test only(dfmiss.scale_ratio) ≈ 0.7
    mmap1, mmap2 = gref_compare_key_maps(nmiss1, nmiss2, [:CountryCode, :Sector])
    @test mmap1 == [1, 2, 3]
    @test mmap2 == [2, 3, 1]

    # 5d. Duplicate keys throw (both inputs).
    ndup = DataFrame(CountryCode = ["A", "A"], Sector = ["x", "x"])
    gdup = mrio_graph(Float64[0 1; 1 0], ndup)
    gok = mrio_graph(Float64[0 1; 1 0], DataFrame(CountryCode = ["A", "B"], Sector = ["x", "x"]))
    err_dup1 = try
        compare_networks(gdup, gok)
        nothing
    catch e
        e
    end
    @test err_dup1 isa ArgumentError
    @test occursin("duplicate", sprint(showerror, err_dup1))
    @test occursin("A", sprint(showerror, err_dup1))
    @test_throws ArgumentError compare_networks(gok, gdup)

    # 5e/5f. Disjoint schemas (`:keys` finds zero shared columns) and
    # unknown match values throw.
    nd1 = DataFrame(CodeA = ["a", "b"])
    nd2 = DataFrame(CodeB = ["a", "b"])
    gd1 = mrio_graph(Float64[0 1; 1 0], nd1)
    gd2 = mrio_graph(Float64[0 1; 1 0], nd2)
    @test_throws ArgumentError compare_networks(gd1, gd2)
    @test_throws ArgumentError compare_networks(g1, g2; match = :bogus)
    @test_throws ArgumentError compare_networks(g1, g2; match = "keys")
    @test_throws ArgumentError compare_networks(g1, g2; match = Symbol[])
    @test_throws ArgumentError compare_networks(g1, g2; match = [:Nope])
    @test_throws ArgumentError compare_networks(g1, g2; match = 42)

    # 5g. Zero matched nodes: one row with n_matched == 0, NaN edge
    # overlap/correlations/densities/ratios, and zero (empty-sum) scales.
    nz1 = DataFrame(CountryCode = ["A"], Sector = ["s"])
    nz2 = DataFrame(CountryCode = ["B"], Sector = ["s"])
    gz1 = mrio_graph(zeros(1, 1), nz1)
    gz2 = mrio_graph(zeros(1, 1), nz2)
    dfz = compare_networks(gz1, gz2)
    @test only(dfz.n_matched) == 0
    @test only(dfz.n_1) == 1
    @test only(dfz.n_2) == 1
    @test isnan(only(dfz.edge_overlap))
    @test isnan(only(dfz.pearson_out))
    @test isnan(only(dfz.spearman_out))
    @test isnan(only(dfz.pearson_in))
    @test isnan(only(dfz.spearman_in))
    @test isnan(only(dfz.density_1))
    @test isnan(only(dfz.density_2))
    @test isnan(only(dfz.density_ratio))
    @test only(dfz.scale_1) == 0.0
    @test only(dfz.scale_2) == 0.0
    @test isnan(only(dfz.scale_ratio))

    # 6. Spearman ties: the helper pins average ranks, and tied strength
    # vectors agree end to end with the counting-based reference.
    # `_tied_ranks([1,2,2,4]) == [1,2.5,2.5,4]`.
    @test ext._tied_ranks([1.0, 2.0, 2.0, 4.0]) == [1.0, 2.5, 2.5, 4.0]
    @test ext._tied_ranks([3.0, 1.0, 3.0, 1.0, 2.0]) ≈ gref_count_ranks([3.0, 1.0, 3.0, 1.0, 2.0])
    # Crafted graphs with tied out-strengths: sout_1 = [2,2,5,1],
    # sout_2 = [1,3,3,3] (diagonals dropped, directed, no filter).
    Wt1 = Float64[0 2 0 0; 2 0 0 0; 0 0 0 5; 0 0 1 0]
    Wt2 = Float64[0 1 0 0; 0 0 3 0; 0 0 0 3; 3 0 0 0]
    nt = gref_nodes(4)
    gt1 = mrio_graph(copy(Wt1), nt)
    gt2 = mrio_graph(copy(Wt2), nt)
    dft = compare_networks(gt1, gt2)
    @test gref_count_ranks([2.0, 2.0, 5.0, 1.0]) == [2.5, 2.5, 4.0, 1.0]
    @test only(dft.spearman_out) ≈ Statistics.cor([2.5, 2.5, 4.0, 1.0], [1.0, 3.0, 3.0, 3.0])
    @test only(dft.spearman_out) ≈
        Statistics.cor(gref_count_ranks([2.0, 2.0, 5.0, 1.0]), gref_count_ranks([1.0, 3.0, 3.0, 3.0]))

    # 7. `pagerank = true`: appends exactly the two PageRank columns
    # (absent otherwise), matching an independent construction — the 3x3
    # matched blocks built by explicit loops, solved via `mrio_graph` +
    # `pagerank_scores`, correlated with `Statistics.cor`.
    # Fixture (matched C1/C2/C3, damping 0.7):
    #   WA = [0 1 1 5; 0 0 2 0; 3 0 0 0; 0 0 0 0]
    #   WB = [0 2 0 1; 0 0 1 0; 1 1 0 4; 0 0 0 0]
    # give non-constant PageRank vectors on both matched subgraphs.
    @test !("pearson_pagerank" in names(df))
    @test !("spearman_pagerank" in names(df))
    WAp = Float64[0 1 1 5; 0 0 2 0; 3 0 0 0; 0 0 0 0]
    WBp = Float64[0 2 0 1; 0 0 1 0; 1 1 0 4; 0 0 0 0]
    ga = mrio_graph(copy(WAp), nodes1)
    gb = mrio_graph(copy(WBp), nodes2)
    dfp = compare_networks(ga, gb; pagerank = true, damping = 0.7)
    @test names(dfp) == [base_cols; "pearson_pagerank"; "spearman_pagerank"]
    @test isfinite(only(dfp.pearson_pagerank))
    @test isfinite(only(dfp.spearman_pagerank))
    map_a, map_b = gref_compare_key_maps(nodes1, nodes2, [:CountryCode, :Sector])
    VAm = [gref_directed_w(WAp, map_a[i], map_a[j], 0.0, false) for i in 1:3, j in 1:3]
    VBm = [gref_directed_w(WBp, map_b[i], map_b[j], 0.0, false) for i in 1:3, j in 1:3]
    sub = DataFrame(CountryCode = ["C1", "C2", "C3"], Sector = fill("s", 3))
    pa = pagerank_scores(mrio_graph(VAm, sub; direction = :directed); damping = 0.7).data
    pb = pagerank_scores(mrio_graph(VBm, sub; direction = :directed); damping = 0.7).data
    @test only(dfp.pearson_pagerank) ≈ Statistics.cor(pa, pb)
    @test only(dfp.spearman_pagerank) ≈ Statistics.cor(gref_count_ranks(pa), gref_count_ranks(pb))
    # The lazy matched view reads back exactly the source pair weights
    # (the invariant the PageRank option relies on).
    gma = ext._matched_graph(ga, map_a)
    @test size(gma.weights) == (3, 3)
    for i in 1:3, j in 1:3
        @test Graphs.weights(gma)[i, j] == Graphs.weights(ga)[map_a[i], map_a[j]]
    end
    # m == 0 skips the solves (NaN PageRank columns, no throw).
    dfpz = compare_networks(gz1, gz2; pagerank = true)
    @test only(dfpz.n_matched) == 0
    @test isnan(only(dfpz.pearson_pagerank))
    @test isnan(only(dfpz.spearman_pagerank))
    # m == 1: correlations NaN (length-1), and the matched solve itself is
    # the [1.0] single-node fixed point.
    n1m = DataFrame(CountryCode = ["A", "B"], Sector = fill("s", 2))
    n2m = DataFrame(CountryCode = ["A", "C"], Sector = fill("s", 2))
    g1m = mrio_graph(Float64[0 2; 3 0], n1m)
    g2m = mrio_graph(Float64[0 4; 1 0], n2m)
    dfm = compare_networks(g1m, g2m; pagerank = true)
    @test only(dfm.n_matched) == 1
    @test isnan(only(dfm.edge_overlap))
    @test isnan(only(dfm.pearson_out))
    @test isnan(only(dfm.pearson_pagerank))
    @test isnan(only(dfm.spearman_pagerank))
    gm1 = ext._matched_graph(g1m, [1])
    @test ext._weighted_pagerank(gm1, 0.85, [1.0], 1.0e-6, 100) ≈ [1.0]

    # 9 (networks). Validation: NaN at a retained off-diagonal throws an
    # ArgumentError naming the entry point and the pair (mirroring the
    # PageRank/Community validation tests); non-Bool `pagerank` and bad
    # `damping` throw (damping is always validated, even without pagerank).
    gnan1 = mrio_graph(Float64[0 NaN; 0 0], gref_nodes(2))
    gnan2 = mrio_graph(Float64[0 1; 0 0], gref_nodes(2))
    @test_throws ArgumentError compare_networks(gnan1, gnan2)
    @test_throws ArgumentError compare_networks(gnan2, gnan1)
    gnan_err = try
        compare_networks(gnan1, gnan2)
        nothing
    catch e
        e
    end
    @test gnan_err isa ArgumentError
    @test occursin("compare_networks", sprint(showerror, gnan_err))
    @test occursin("w(1, 2)", sprint(showerror, gnan_err))
    @test_throws ArgumentError compare_networks(g1, g2; pagerank = 1)
    @test_throws ArgumentError compare_networks(g1, g2; damping = 0.0)
    @test_throws ArgumentError compare_networks(g1, g2; damping = 1.0)
    @test_throws ArgumentError compare_networks(g1, g2; damping = -0.1)
    @test_throws ArgumentError compare_networks(g1, g2; pagerank = true, damping = 1.5)

    # 10. Memory guard (the plan's O(1)-extra rule): two seeded n = 2000
    # graphs must stay far below an m×m submatrix copy (≥ 32 MiB for
    # Float64) — with and without `pagerank = true` (which adds only O(m)
    # iteration vectors).
    Wbig1 = rand(MersenneTwister(1001), 2000, 2000)
    Wbig2 = rand(MersenneTwister(1002), 2000, 2000)
    gbig1 = mrio_graph(Wbig1, gref_nodes(2000))
    gbig2 = mrio_graph(Wbig2, gref_nodes(2000))
    compare_networks(gbig1, gbig2) # warm-up (compilation)
    @test gref_compare_alloc(gbig1, gbig2) < 12 * 2^20
    compare_networks(gbig1, gbig2; pagerank = true) # warm-up (compilation)
    @test gref_compare_pr_alloc(gbig1, gbig2) < 12 * 2^20
end

@testset "Partition comparison" begin
    part_cols = [
        "n_matched",
        "n_1",
        "n_2",
        "ari",
        "nmi",
        "n_communities_1",
        "n_communities_2",
        "modularity_1",
        "modularity_2",
    ]

    # 1. Identical inputs give ARI = NMI = 1, in both raw-vector and
    # CommunityResult forms (exact 1.0: identical partitions make the
    # numerator and denominator bitwise equal, and the 0/0 rules cover the
    # trivial cases).
    dfi = compare_partitions([1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3])
    @test names(dfi) == part_cols
    @test only(dfi.n_matched) == 6
    @test only(dfi.ari) == 1.0
    @test only(dfi.nmi) == 1.0
    @test only(dfi.n_communities_1) == 3
    @test only(dfi.n_communities_2) == 3
    @test isnan(only(dfi.modularity_1))
    @test isnan(only(dfi.modularity_2))

    # 2. Hand-computed ARI/NMI case (6 nodes):
    #   A = [1,1,1,2,2,2], B = [1,1,2,2,3,3].
    # Contingency (rows A, cols B): n_11 = 2 (items 1, 2), n_12 = 1
    # (item 3), n_22 = 1 (item 4), n_23 = 2 (items 5, 6).
    #   ΣC(n,2) = C(2,2) + C(2,2) = 2; ΣC(a,2) = 3 + 3 = 6;
    #   ΣC(b,2) = 1 + 1 + 1 = 3; C(6,2) = 15.
    # ARI = (2 − 6*3/15) / ((6 + 3)/2 − 6*3/15) = (2 − 1.2)/(4.5 − 1.2)
    #     = 0.8/3.3 = 8/33 ≈ 0.242424.
    # NMI: I = (1/3)log(12/6) + (1/3)log(12/6) = 2/3*log 2 (the two
    # size-2 cells; the size-1 cells contribute log(6/6) = 0),
    # H_1 = log 2, H_2 = log 3, so NMI = 2I/(H_1+H_2) = 4log2/(3log6)
    #     ≈ 0.5158037.
    A6 = [1, 1, 1, 2, 2, 2]
    B6 = [1, 1, 2, 2, 3, 3]
    df6 = compare_partitions(A6, B6)
    @test only(df6.ari) ≈ 8 / 33
    @test only(df6.nmi) ≈ 4 * log(2) / (3 * log(6))
    ari_ref, nmi_ref = gref_ari_nmi_ref(A6, B6)
    @test only(df6.ari) ≈ ari_ref
    @test only(df6.nmi) ≈ nmi_ref
    @test ari_ref ≈ 8 / 33
    @test nmi_ref ≈ 4 * log(2) / (3 * log(6))

    # 3. ARI/NMI edge cases (exact pinned values).
    # Both one-cluster: 0/0 rules give 1.0/1.0.
    df_one = compare_partitions([1, 1, 1, 1], [7, 7, 7, 7])
    @test only(df_one.ari) == 1.0
    @test only(df_one.nmi) == 1.0
    # Both all-singletons: 0/0 rules give 1.0/1.0.
    df_single = compare_partitions([1, 2, 3], [4, 5, 6])
    @test only(df_single.ari) == 1.0
    @test only(df_single.nmi) == 1.0
    # All-singletons vs one-cluster: ARI = 0/denom = 0, NMI = 0/H = 0.
    df_mix = compare_partitions([1, 2, 3], [1, 1, 1])
    @test only(df_mix.ari) == 0.0
    @test only(df_mix.nmi) == 0.0
    # Constant vs non-constant: ΣC(cells) = expected exactly, so ARI = 0;
    # I = 0, so NMI = 0.
    df_const = compare_partitions([1, 1, 1, 1], [1, 1, 2, 2])
    @test only(df_const.ari) == 0.0
    @test only(df_const.nmi) == 0.0
    # Independent 2-block partitions [1,1,2,2] vs [1,2,1,2]: the 2x2
    # contingency is all ones, so ΣC(cells) = 0, ΣC(a) = ΣC(b) = 2,
    # C(4,2) = 6, and ARI = (0 − 4/6)/(2 − 4/6) = −1/2 (below chance is
    # negative — independence gives 0 only in expectation). NMI = 0
    # (I = 0 over uniform cells).
    df_ind = compare_partitions([1, 1, 2, 2], [1, 2, 1, 2])
    @test only(df_ind.ari) ≈ -0.5
    @test only(df_ind.nmi) == 0.0
    ari_ind_ref, nmi_ind_ref = gref_ari_nmi_ref([1, 1, 2, 2], [1, 2, 1, 2])
    @test only(df_ind.ari) ≈ ari_ind_ref
    @test only(df_ind.nmi) ≈ nmi_ind_ref
    # Label-permutation invariance: relabeling both sides (including to
    # negative and huge labels) changes nothing.
    a7 = [1, 1, 2, 2, 3, 3, 3]
    b7 = [1, 2, 1, 2, 3, 1, 3]
    df7 = compare_partitions(a7, b7)
    df7r = compare_partitions([10, 10, -5, -5, 10^12, 10^12, 10^12], [7, 8, 7, 8, 9, 7, 9])
    @test only(df7r.ari) ≈ only(df7.ari)
    @test only(df7r.nmi) ≈ only(df7.nmi)
    ari7_ref, nmi7_ref = gref_ari_nmi_ref(a7, b7)
    @test only(df7.ari) ≈ ari7_ref
    @test only(df7.nmi) ≈ nmi7_ref
    # Negative and large labels are accepted, with correct counts.
    df_neg = compare_partitions(Int64[-10^12, -10^12, 10^12], Int32[-3, -3, -3])
    @test only(df_neg.n_communities_1) == 2
    @test only(df_neg.n_communities_2) == 1
    @test only(df_neg.ari) == 0.0
    @test only(df_neg.nmi) == 0.0

    # 8. CommunityResult inputs: two results with differently ordered node
    # tables (rows and membership permuted consistently) describe the same
    # partition, so key alignment (not positions) gives ARI = NMI = 1.0,
    # with the stored modularities passed through.
    Wp = copy(GRAPH_PLANT_Z)
    np1 = DataFrame(
        CountryCode = ["A", "A", "A", "B", "B", "B"],
        Sector = ["s1", "s2", "s3", "s1", "s2", "s3"],
    )
    c1 = CommunityResult(
        Int32[1, 1, 1, 2, 2, 2],
        0.42,
        :louvain,
        1.0,
        7,
        np1,
        Wp,
        (0.0, 0.0, false, 0.0),
        true,
    )
    perm = [1, 4, 2, 5, 3, 6]
    np2 = np1[perm, :]
    # np2 rows are A/s1, B/s1, A/s2, B/s2, A/s3, B/s3, so the consistently
    # permuted membership is [1,2,1,2,1,2] — the same partition as c1's
    # [1,1,1,2,2,2] seen through reordered rows.
    c2 = CommunityResult(
        Int32[1, 2, 1, 2, 1, 2],
        0.43,
        :leiden,
        1.0,
        nothing,
        np2,
        Wp,
        (0.0, 0.0, false, 0.0),
        true,
    )
    dfc = compare_partitions(c1, c2)
    @test names(dfc) == part_cols
    @test only(dfc.n_matched) == 6
    @test only(dfc.n_1) == 6
    @test only(dfc.n_2) == 6
    @test only(dfc.ari) == 1.0
    @test only(dfc.nmi) == 1.0
    @test only(dfc.n_communities_1) == 2
    @test only(dfc.n_communities_2) == 2
    @test only(dfc.modularity_1) == 0.42
    @test only(dfc.modularity_2) == 0.43
    # Positional-vs-key proof: the same two membership vectors compared
    # positionally (ignoring node order) are different partitions.
    dfc_pos = compare_partitions(c1.membership, c2.membership)
    @test only(dfc_pos.ari) < 1.0
    # Mixed forms are positional with a length check; modularity is NaN on
    # the vector side.
    dfm1 = compare_partitions(c1, Vector{Int}(c1.membership))
    @test only(dfm1.ari) == 1.0
    @test only(dfm1.nmi) == 1.0
    @test only(dfm1.modularity_1) == 0.42
    @test isnan(only(dfm1.modularity_2))
    @test only(dfm1.n_1) == 6
    @test only(dfm1.n_2) == 6
    dfm2 = compare_partitions(Vector{Int}(c2.membership), c2)
    @test only(dfm2.ari) == 1.0
    @test isnan(only(dfm2.modularity_1))
    @test only(dfm2.modularity_2) == 0.43
    @test_throws ArgumentError compare_partitions(c1, [1, 1, 1])
    @test_throws ArgumentError compare_partitions([1, 1, 1], c1)
    # Community counts on the matched subset: c3 has nodes A..D with
    # labels [1,2,3,3] (3 communities); c4 has nodes A,B with labels
    # [7,7] (1 community). Matched A,B give labels [1,2] vs [7,7]:
    # n_communities_1 == 2 (not 3), n_communities_2 == 1, and the m = 2
    # minimal case gives ARI = NMI = 0.0.
    n3 = DataFrame(Code = ["A", "B", "C", "D"])
    n4 = DataFrame(Code = ["A", "B"])
    W34 = zeros(4, 4)
    c3 = CommunityResult(Int32[1, 2, 3, 3], 0.1, :louvain, 1.0, nothing, n3, W34, (0.0, 0.0, false, 0.0), true)
    c4 = CommunityResult(
        Int32[7, 7],
        0.2,
        :louvain,
        1.0,
        nothing,
        n4,
        zeros(2, 2),
        (0.0, 0.0, false, 0.0),
        true,
    )
    df34 = compare_partitions(c3, c4)
    @test only(df34.n_matched) == 2
    @test only(df34.n_1) == 4
    @test only(df34.n_2) == 2
    @test only(df34.n_communities_1) == 2
    @test only(df34.n_communities_2) == 1
    @test only(df34.ari) == 0.0
    @test only(df34.nmi) == 0.0
    # 8b. Missing-safe key alignment: both node tables carry a `missing`
    # key, paired via `isequal`. c5 order is [A, missing, C] with labels
    # [1, 1, 2]; c6 order is [C, A, missing] with the consistently
    # permuted labels [2, 1, 1] — the same partition, so ARI = NMI = 1.0
    # aligned through the `missing` pair (a non-missing-safe matcher would
    # align only 2 items).
    nm5 = DataFrame(Code = Union{String, Missing}["A", missing, "C"])
    nm6 = nm5[[3, 1, 2], :]
    Wm56 = zeros(3, 3)
    c5 = CommunityResult(Int32[1, 1, 2], 0.11, :louvain, 1.0, nothing, nm5, Wm56, (0.0, 0.0, false, 0.0), true)
    c6 = CommunityResult(Int32[2, 1, 1], 0.12, :louvain, 1.0, nothing, nm6, Wm56, (0.0, 0.0, false, 0.0), true)
    dfm56 = compare_partitions(c5, c6)
    @test only(dfm56.n_matched) == 3
    @test only(dfm56.n_1) == 3
    @test only(dfm56.n_2) == 3
    @test only(dfm56.ari) == 1.0
    @test only(dfm56.nmi) == 1.0
    @test only(dfm56.n_communities_1) == 2
    @test only(dfm56.n_communities_2) == 2
    # Matching-rule errors mirror the network side.
    @test_throws ArgumentError compare_partitions(c3, c3; match = :bogus)
    ndup3 = DataFrame(Code = ["A", "A"])
    cdup = CommunityResult(
        Int32[1, 1],
        0.0,
        :louvain,
        1.0,
        nothing,
        ndup3,
        zeros(2, 2),
        (0.0, 0.0, false, 0.0),
        true,
    )
    @test_throws ArgumentError compare_partitions(cdup, cdup)
    nx = DataFrame(Other = ["A", "B", "C", "D"])
    cx = CommunityResult(
        Int32[1, 1, 2, 2],
        0.0,
        :louvain,
        1.0,
        nothing,
        nx,
        W34,
        (0.0, 0.0, false, 0.0),
        true,
    )
    @test_throws ArgumentError compare_partitions(c3, cx)

    # 9 (partitions). Validation: Bool/non-Integer vectors and unequal
    # lengths throw ArgumentError.
    @test_throws ArgumentError compare_partitions([true, false], [true, false])
    @test_throws ArgumentError compare_partitions([1.0, 2.0], [1, 2])
    @test_throws ArgumentError compare_partitions([1, 2], ["a", "b"])
    @test_throws ArgumentError compare_partitions([1, 1, 2], [1, 2])
    @test_throws ArgumentError compare_partitions(c1, [true, true, true, true, true, true])
    @test_throws ArgumentError compare_partitions(c1, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    gbool_err = try
        compare_partitions([true, false], [true, false])
        nothing
    catch e
        e
    end
    @test gbool_err isa ArgumentError
    @test occursin("Bool", sprint(showerror, gbool_err))
end
