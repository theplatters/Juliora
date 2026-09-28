# Changelog

All notable changes to Juliora are documented in this file.

## [0.3.0] - 2026-09-25

This release introduces graph analysis of MRIO networks: a Graphs.jl
extension on the Julia side and R bindings for the whole graph API, plus
JLD2-based caching of parsed GLORIA data.

### Added

- `mrio_graph` — zero-copy graph views over the dense MRIO matrices
  (`Z`/`T` transactions or the `A` technical matrix; directed or
  undirected), with on-the-fly edge filtering (`threshold`, `min_share`,
  `self_loops`). `graph_summary` reports size, retained weight share and
  referenced memory; `to_simple_graph` extracts a `Graphs.SimpleGraph`/
  `SimpleDiGraph` for the wider Graphs.jl ecosystem.
- `pagerank_scores` — flow-weighted PageRank (with an unweighted mode and
  dangling-node handling) over the graph, returned as a `SeriesEntry` on
  the node metadata.
- `communities` — community detection with Louvain, Leiden, label
  propagation and spectral clustering (`ncommunities` for the spectral
  method), returning a `CommunityResult`; `community_table` and
  `community_summary` turn results into data frames.
- `node_similarity` and `similarity_graph` — top-k structural node
  similarity (cosine, Jaccard or personalized random walk) and symmetric
  kNN similarity graphs, computed in blocks without ever allocating an
  n×n similarity matrix.
- `compare_networks` and `compare_partitions` — cross-network comparison
  on matched nodes (weighted edge overlap, Pearson/Spearman correlations
  of node strengths and PageRank, density and scale ratios) and partition
  comparison (adjusted Rand index, normalized mutual information,
  community counts and modularities).
- R bindings for the complete graph API in `R/graph_bindings.R`:
  `mrio_graph()`, `graph_summary()`, `communities()`, `community_table()`,
  `community_summary()`, `pagerank_scores()`, `node_similarity()`,
  `similarity_graph()`, `compare_networks()`, `compare_partitions()` and
  the `ensure_graphs_loaded()` helper, with `MRIOGraph` and
  `CommunityResult` S3 classes and testthat coverage.
- Aggregation and helper utilities: `aggregate`/`groupby` of MRIO tables
  by country or sector columns, plus sector/country accessors.
- `save_gloria_cache()`/`load_gloria_cache()` — JLD2-based caching of
  parsed GLORIA MRIO data with a versioned, validated cache schema; a
  cache can be loaded directly via `Gloria(path)` or
  `Gloria(path, version, year)` when given a `.jld2`/`.jdl2` path.
- `juliora_reset()` — R helper that clears the cached Julia connection
  state so the next Juliora call re-runs project discovery and reloads
  the Julia packages (optionally stopping the Julia server).

### Notes

- Graph features load through a Graphs.jl package extension: they become
  available as soon as Graphs is installed in the active Julia project
  (`import Pkg; Pkg.add("Graphs")`); the rest of Juliora does not need it.
- The graph kernels are dense-first and memory-conscious: filtering is
  applied on the fly, matrices are wrapped by reference (never copied),
  similarity results are compact top-k, and the spectral method is gated
  to n ≤ 5 000. The README's "Graph Analysis" section documents the
  memory budget and recommended aggregation recipes.
- From R, note that R matrices and data frames are transferred to Julia
  first; the zero-copy guarantees apply on the Julia side.

## [0.2.0] - 2026-07-05

Initial development: MRIO construction and parsing (Eora, Gloria),
`MatrixEntry`/`SeriesEntry` types, Leontief factorization helpers,
matrix analysis and filtering utilities, and the first R bindings
including dplyr integrations.
