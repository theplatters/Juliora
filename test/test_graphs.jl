using Test
using Juliora
using Graphs
using DataFrames
using LinearAlgebra
using SparseArrays
using Random

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
