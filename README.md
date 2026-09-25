# Juliora

Juliora is a Julia package, with R bindings, for parsing and analyzing
multi-regional input-output (MRIO) databases. It provides labeled matrix and
series containers, MRIO database constructors, environmental extensions,
aggregation tools, and helpers for tidy analysis workflows.

The package is designed for working with large economic input-output systems
where matrix values need to stay attached to country, sector, final demand,
value added, or environmental stressor metadata.

## Capabilities

- Load complete Eora and Gloria MRIO databases from local data files.
- Parse Gloria input-output and supply-use table files.
- Store labeled matrices with `MatrixEntry` and labeled vectors with
  `SeriesEntry`.
- Construct complete `MRIO` objects from transaction, final demand, and value
  added matrices.
- Compute technical coefficients, total output, and Leontief factorizations.
- Attach environmental extensions with direct impacts and impact intensities.
- Calculate environmental impacts from production vectors or production
  scenario matrices.
- Estimate induced production using the Leontief inverse for selected consumer
  and producer countries.
- Filter, subset, drop, group, and aggregate matrices by country, sector, or
  other index metadata.
- Convert matrices to and from long-form data frames for tidy analysis.
- Analyze flow tables as graphs — PageRank, community detection, node
  similarity, and cross-network comparison on a zero-copy view of the data.
- Produce country, sector, bilateral flow, and matrix summary tables.
- Use Julia `Tidier` macros and R `dplyr` methods for metadata-oriented
  workflows.

## Core Data Model

Juliora centers on a few typed containers:

- `MatrixEntry`: a numeric matrix plus row and column metadata tables.
- `SeriesEntry`: a numeric vector plus element metadata.
- `EnvironmentalExtension`: direct environmental impacts `F` and
  environmental intensity matrix `A`.
- `MRIO`: a complete input-output database containing technical coefficients
  `A`, transactions `T`/`Z`, value added `VA`, final demand `FD`/`Y`, Leontief
  factorization `L`, total output `X`, and environmental data `env`.

Because matrix dimensions are tied to metadata dimensions, constructors validate
that data sizes match the supplied row and column indices.

## Julia Usage

Instantiate dependencies:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Load the package from this repository:

```julia
using Juliora
using DataFrames
```

Create a labeled matrix:

```julia
data = [1.0 2.0; 3.0 4.0]

indices = DataFrame(
    CountryCode = ["AUT", "DEU"],
    Sector = ["Agriculture", "Manufacturing"],
)

z = MatrixEntry(data, indices, indices)
```

Use named metadata to access values:

```julia
z[(CountryCode = "AUT", Sector = "Agriculture"),
  (CountryCode = "DEU", Sector = "Manufacturing")]
```

Load external MRIO data:

```julia
eora = Eora("data/2017/")
gloria = Gloria("data/GLORIA/", 60, 2019)

# Lower-level Gloria parsers are also available.
gloria_io = parse_gloria("data/GLORIA/", 2019; version = 60)
gloria_sut = parse_gloria_sut("data/GLORIA/", 2019; version = 60)
```

Query metadata and summarize flows:

```julia
countries(gloria)
sectors(gloria)
stressors(gloria)

sum_by_country(gloria.Z; dimension = :both)
sum_by_sector(gloria.Z; dimension = :rows)
country_summary(gloria.Z)
matrix_summary(gloria.Z)
```

Filter and aggregate matrices:

```julia
aut_rows = filter_rows(gloria.Z, row -> row.CountryCode == "AUT")
manufacturing_cols = filter_cols(gloria.Z, col -> col.Sector == "Manufacturing")

sector_totals = aggregate(groupby(gloria.Z, :Sector; dims = 1), sum)
aggregated_mrio = aggregate(gloria, [:CountryCode]; dims = 2)
```

Run environmental and Leontief-based analysis:

```julia
impact = environmental_impact(gloria, gloria.X.data)

production = induced_production(
    gloria;
    consumer_countries = ["AUT", "DEU"],
    producer_countries = ["CHN", "IND"],
)
```

Convert between matrix and tabular forms:

```julia
long = to_long_dataframe(gloria.Z; value_name = "flow")
wide = pivot_matrix_to_wide(gloria.Z, [:CountryCode], :Sector, "flow")
rebuilt = from_long_dataframe(long; value_col = "flow")
```

## Graph Analysis

The graph extension turns the dense flow tables into a `Graphs.jl` graph
without copying them. It loads with `using Graphs` (Graphs is a weak
dependency, so it must be installed in the active environment) and works on
any square flow matrix plus a node-metadata `DataFrame`.

### Quick start

Build a graph directly from a flow matrix and node metadata. An edge `i → j`
is the monetary flow supplied by node `i` to buyer `j` (`W[i, j]`):

```julia
using Juliora
using Graphs
using DataFrames

nodes = DataFrame(
    CountryCode = ["A", "A", "B", "B"],
    Sector = ["s1", "s2", "s1", "s2"],
)
W = [0.0 5.0 4.0 0.0;
     5.0 0.0 0.0 4.0;
     4.0 0.0 0.0 5.0;
     0.0 4.0 5.0 0.0]

g = mrio_graph(W, nodes)
```

`graph_summary` returns a one-row `DataFrame` (post-filter edge count,
retained weight share, referenced memory):

```julia
graph_summary(g)
```

`pagerank_scores` returns a `SeriesEntry` of flow-weighted PageRank scores
over the node metadata:

```julia
scores = pagerank_scores(g; damping = 0.85)
```

`communities` returns a `CommunityResult`; `community_table` joins the
node-to-community mapping with the node metadata, and `community_summary`
reports one row per community:

```julia
result = communities(g; algorithm = :louvain, seed = 1)
community_table(result)
community_summary(result)
```

`node_similarity` returns a directed top-k similarity graph over a compact
`SparseMatrixCSC{Float32, Int32}`; `similarity_graph` symmetrizes it into an
undirected kNN graph ready for `communities`:

```julia
sim = node_similarity(g; method = :cosine, k = 2)
knn = similarity_graph(g; method = :cosine, k = 2)
communities(knn; algorithm = :louvain, seed = 1)
```

`compare_networks` and `compare_partitions` each return a one-row `DataFrame`:

```julia
g2 = mrio_graph(2 .* W, nodes)
compare_networks(g, g2; match = :keys)
compare_partitions(result, communities(g; algorithm = :leiden, seed = 1))
```

### Semantics worth knowing

- `mrio_graph(W, nodes; direction = :directed, threshold = 0.0, min_share = 0.0,
  self_loops = false)` drops self-flows `Z[i, i]` by default; the filter
  `τ = max(threshold, min_share · Σ|W|)` applies as `|w| < τ → 0` on the fly,
  never as a pruned copy.
- Per-feature direction defaults: `mrio_graph`, `pagerank_scores`, and the
  `node_similarity` / `similarity_graph` MRIO methods default to `:directed`
  for the source graph; `communities(mrio)` defaults to `:undirected`
  (symmetrized flow weights).
- `communities` algorithms are `:louvain`, `:leiden`, `:label_propagation`,
  and `:spectral`; `ncommunities` is only accepted with `:spectral`.
  `:spectral` needs an O(n²) workspace and is gated to graphs with n ≤ 5 000 —
  aggregate first (see recipes).
- `node_similarity` methods are `:cosine`, `:jaccard`, and `:random_walk`;
  `:random_walk` needs `sources` (one personalized-PageRank solve per source).
  `similarity_graph` (`symmetrize = :max` or `:mean`) returns an undirected kNN
  graph usable directly as `communities` input.
- `compare_networks` / `compare_partitions` align nodes via the shared key
  columns (`match = :keys`, or explicit columns). `compare_networks` reports
  every metric on the matched node set; `compare_partitions` scores ARI, NMI,
  and the community counts on the matched items (its `modularity` columns are
  each partition's own stored value).
- The full contract lives in the docstrings (`?mrio_graph`, `?communities`, `?node_similarity`, …).

### Memory strategy

The dense matrix is the source of truth: the graph layer wraps the resident
`Matrix{Float64}` and never allocates a second full-size copy of the n×n
data. A sparse copy of the full matrix would be counter-effective: MRIO
tables are near-complete graphs (`nnz ≈ n²`), so CSC (`Float64` values plus
`Int64` indices) is about 2× larger than dense (16 B/entry vs 8 B/entry),
plus a conversion pass and a transient double allocation; sparse and
edge-list representations only pay off after the problem is reduced (pruned
or aggregated).

| # | Rule | Meaning |
|---|---|---|
| R1 Zero-copy wrapping | `mrio_graph` wraps the existing matrix — no copy, no `Float32` conversion. |
| R2 No symmetrized copy | Undirected kernels read `W[i, j] + W[j, i]` on the fly; `W + Wᵀ` is never materialized. |
| R3 On-the-fly weight filtering | `threshold` / `min_share` are applied inside the kernels, never as a pruned matrix. |
| R4 Sparse/edge extraction opt-in only | Only for Graphs.jl ecosystem interop and weight-blind algorithms (label propagation); a single streaming pass emits already-pruned compact `(src::Int32, dst::Int32, w::Float32)` edges. |
| R5 Multilevel shrinks itself | Level 0 works on dense tiles with O(n) bookkeeping; level ≥ 1 aggregates are dense `n₁×n₁` with `n₁ ≈ #communities ≪ n`. |
| R6 Block n×n-shaped results | Similarity never allocates an n×n result: row-block × n scratch plus top-k per row. |

Extra memory budget with n ≈ 25 000 (Gloria) and `W` already resident:

| Operation | Extra memory |
|---|---|
| `mrio_graph` construction | **0** |
| PageRank | 3n `Float64` ≈ 0.6 MB |
| Louvain / Leiden | O(n) + tile scratch (≈ 512²·8 B ≈ 2 MB) |
| Similarity (k = 10) | block·n·8 B + k·n·12 B ≈ 20 MB |
| Spectral | O(n²) workspace, **gated** to n ≤ 5 000 (aggregated graphs) |
| `to_simple_graph` (opt-in) | 2·m·8 B for m extracted edges, post-pruning |

Do not route graph workflows through the Leontief inverse:
`LeontiefFactorization.data` materializes a dense n×n inverse (~5 GB per
matrix at n = 25 000), so graph code never touches it. Graph-only workflows
should build `mrio_graph(W, nodes)` from their flow matrix and skip `MRIO` /
Leontief construction entirely. Zero-copy covers `Matrix` and sparse inputs
(any other `AbstractMatrix` is converted once via `Matrix(W)`).

### Recipes

Graph-only workflow, no Leontief — only the flow matrix and node metadata:

```julia
g = mrio_graph(W, nodes)
scores = pagerank_scores(g)
```

Prune and aggregate before scaling. `threshold` / `min_share` filtering is
free (applied on the fly); shrinking the node count needs `aggregate` — e.g.
Eora at ≈ 4.9k nodes (189 countries × 26 sectors) aggregates to 189 country
nodes. Aggregate both dimensions to keep a square flow table, before
`:spectral` (size gate) and before similarity runs at Gloria scale
(20–40k nodes):

```julia
by_country = aggregate(aggregate(mrio, [:CountryCode]; dims = 1), [:CountryCode]; dims = 2)
g_small = mrio_graph(by_country)
communities(g_small; algorithm = :spectral, ncommunities = 10, seed = 1)
```

Extract a sparse graph only when needed — for Graphs.jl ecosystem algorithms
that need an edge list (one streaming pass over the already-pruned matrix):

```julia
simple, distmx = to_simple_graph(g)
```

Compare two networks or partitions on matched key columns — e.g. two years of
the same database, or two databases sharing `CountryCode` / `Sector` keys:

```julia
compare_networks(g_2019, g_2020; match = :keys)
compare_partitions(communities(g_2019), communities(g_2020))
```

## R Usage

The repository also exposes R bindings through `JuliaConnectoR`. The R package
wraps Julia objects as S3 objects and provides R functions with names aligned to
the Julia API.

Run R tests:

```sh
Rscript -e 'devtools::test()'
```

Check that Julia is available to R:

```r
library(Juliora)

is_julia_available()
```

Create and inspect labeled data:

```r
idx <- data.frame(
  CountryCode = c("AUT", "DEU"),
  Sector = c("Agriculture", "Manufacturing")
)

z <- MatrixEntry(matrix(c(1, 3, 2, 4), nrow = 2), idx, idx)

matrix_summary(z)
as.data.frame(z)
```

Load and analyze MRIO data:

```r
gloria <- Gloria("data/GLORIA/", version = 60, year = 2019)

countries(gloria)
sectors(gloria)
stressors(gloria)

sum_by_country(gloria$Z, dimension = "both")
country_summary(gloria$Z)
```

Use `dplyr` verbs on metadata:

```r
library(dplyr)

filtered <- gloria$Z |>
  filter(CountryCode == "AUT", .dims = 1) |>
  mutate(region_group = "Austria", .dims = 1)
```

## Development Commands

Run the Julia test suite:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run R package tests:

```sh
Rscript -e 'devtools::test()'
```

Regenerate R documentation after roxygen changes:

```sh
Rscript -e 'devtools::document()'
```

Run the full R package check:

```sh
R CMD check .
```

Format Julia source:

```sh
runic --inplace src/*
```

## Data Requirements

Juliora expects MRIO source data to be available locally. Eora loaders look for
files such as `T.txt`, `VA.txt`, `FD.txt`, `Q.txt`, and their label files in the
provided directory. Gloria loaders expect the corresponding Gloria data release
files for the requested version and year.

Large MRIO datasets are not bundled with this repository.

## License

Juliora is licensed under the MIT license. See `LICENSE` for details.
