library(testthat)
library(Juliora)

if (!JuliaConnectoR::juliaSetupOk()) {
  skip("Julia environment not available for testing")
}

# Arrange a graph-capable Julia session. The repo root Julia project only has
# Graphs as a weak dependency, so `using Graphs` can fail there even though the
# package extension works in production setups. On failure, fall back to a
# temporary Julia environment that resolves Graphs and Juliora from the local
# depot via a copied manifest.
setup_graph_session <- function() {
  loaded <- tryCatch({
    Juliora:::ensure_graphs_loaded()
    TRUE
  }, error = function(e) {
    FALSE
  })
  if (loaded) {
    return(TRUE)
  }

  tryCatch({
    env_dir <- file.path(tempdir(), "juliora-graph-test-env")
    dir.create(env_dir, recursive = TRUE, showWarnings = FALSE)
    root <- normalizePath(Juliora:::find_julia_project(), winslash = "/")

    writeLines(c(
      "[deps]",
      "Graphs = \"86223c79-3864-5bf0-83f7-82e725a168b6\"",
      "Juliora = \"b4cbaa25-ff22-403a-b853-176105f9e354\""
    ), file.path(env_dir, "Project.toml"))

    manifest <- readLines(file.path(root, "Manifest.toml"))
    manifest <- sub("path = \".\"", paste0("path = \"", root, "\""), manifest, fixed = TRUE)
    writeLines(manifest, file.path(env_dir, "Manifest.toml"))

    env_dir_jl <- normalizePath(env_dir, winslash = "/")
    JuliaConnectoR::juliaEval("using Pkg")
    JuliaConnectoR::juliaEval(sprintf("Pkg.activate(\"%s\")", env_dir_jl))
    JuliaConnectoR::juliaEval("using Graphs")
    Juliora:::ensure_graphs_loaded()
    TRUE
  }, error = function(e) {
    FALSE
  })
}

if (!setup_graph_session()) {
  skip("Graphs.jl not available for graph binding tests")
}

# 4 nodes in two 2-node blocks with strong intra-block flows and weak cross
# flows: community detection with seed = 1 recovers the planted partition
# {1, 2} and {3, 4}.
make_graph_fixture <- function() {
  industry_indices <- data.frame(
    CountryCode = c("AUT", "DEU", "CHN", "GBR"),
    Sector = c("A", "B", "A", "B")
  )
  final_demand_indices <- data.frame(Category = "Households")
  value_added_indices <- data.frame(Category = "Value added")

  transactions <- matrix(c(
    0, 8, 1, 1,
    8, 0, 1, 1,
    1, 1, 0, 7,
    1, 1, 7, 0
  ), nrow = 4, byrow = TRUE)
  final_demand <- matrix(c(2, 2, 2, 2), nrow = 4)
  value_added <- matrix(c(4, 6, 4, 6), nrow = 1)

  mrio <- MRIO(
    MatrixEntry(transactions, industry_indices, industry_indices),
    MatrixEntry(final_demand, final_demand_indices, industry_indices),
    MatrixEntry(value_added, industry_indices, value_added_indices)
  )

  list(
    mrio = mrio,
    nodes = industry_indices,
    W = transactions
  )
}

test_that("mrio_graph builds MRIOGraph objects from matrices and MRIO objects", {
  fixture <- make_graph_fixture()

  g <- mrio_graph(fixture$W, fixture$nodes)
  expect_s3_class(g, "MRIOGraph")
  expect_false(inherits(g, "MRIO"))
  expect_equal(g$nodes$CountryCode, fixture$nodes$CountryCode)
  expect_equal(g$nodes$Sector, fixture$nodes$Sector)

  g_mrio <- mrio_graph(fixture$mrio, source = "Z", weights = "flows")
  expect_s3_class(g_mrio, "MRIOGraph")
  expect_false(inherits(g_mrio, "MRIO"))
  expect_equal(graph_summary(g_mrio)$nodes, 4)

  g_mrio_default <- mrio_graph(fixture$mrio)
  expect_s3_class(g_mrio_default, "MRIOGraph")
  expect_equal(graph_summary(g_mrio_default)$edges, 12)
})

test_that("mrio_graph validates inputs and translates Julia errors", {
  fixture <- make_graph_fixture()

  expect_error(mrio_graph(fixture$W), "must be a data.frame")
  expect_error(mrio_graph(fixture$W, "nodes"), "must be a data.frame")
  expect_error(mrio_graph(list(), fixture$nodes), "must be a MRIO object or a numeric matrix")
  expect_error(mrio_graph(matrix(c("a", "b"), nrow = 1), fixture$nodes[1, , drop = FALSE]),
               "must be a MRIO object or a numeric matrix")
  expect_error(mrio_graph(fixture$W, fixture$nodes, source = "Z"), "must be NULL")
  expect_error(mrio_graph(fixture$W, fixture$nodes, weights = "flows"), "must be NULL")
  expect_error(mrio_graph(fixture$mrio, nodes = fixture$nodes), "must be NULL")
  expect_error(mrio_graph(fixture$W, fixture$nodes, direction = "both"), "must be one of")
  expect_error(mrio_graph(fixture$W, fixture$nodes, self_loops = NA), "must be TRUE or FALSE")
  expect_error(mrio_graph(fixture$mrio, source = "Q"), "Julia Error")
})

test_that("graph_summary reports nodes, edges and weights", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  gs <- graph_summary(g)
  expect_s3_class(gs, "data.frame")
  expect_equal(nrow(gs), 1)
  expect_equal(
    names(gs),
    c(
      "nodes", "edges", "directed", "threshold", "min_share", "self_loops",
      "total_weight", "retained_weight", "retained_share", "memory_bytes"
    )
  )
  expect_equal(gs$nodes, 4)
  expect_equal(gs$edges, 12)
  expect_true(gs$directed)
  expect_equal(gs$self_loops, FALSE)
  expect_equal(gs$total_weight, 38)
  expect_equal(gs$retained_weight, 38)
  expect_equal(gs$retained_share, 1)

  # threshold prunes the four unit cross flows in each direction
  gs_threshold <- graph_summary(mrio_graph(fixture$W, fixture$nodes, threshold = 2))
  expect_equal(gs_threshold$edges, 4)
  expect_equal(gs_threshold$retained_weight, 30)
  expect_lt(gs_threshold$retained_share, 1)

  expect_error(graph_summary(list()), "must be a MRIOGraph object")
  expect_error(graph_summary(fixture$mrio), "must be a MRIOGraph object")
})

test_that("communities recovers the planted partition", {
  fixture <- make_graph_fixture()

  cr <- communities(fixture$mrio, seed = 1)
  expect_s3_class(cr, "CommunityResult")
  expect_equal(cr$membership, c(1L, 1L, 2L, 2L))
  expect_type(cr$membership, "integer")
  expect_true(is.numeric(cr$modularity))
  expect_equal(cr$algorithm, "louvain")

  g <- mrio_graph(fixture$W, fixture$nodes)
  cr_graph <- communities(g, seed = 1)
  expect_s3_class(cr_graph, "CommunityResult")
  expect_equal(cr_graph$membership, c(1L, 1L, 2L, 2L))

  cr_spectral <- communities(g, algorithm = "spectral", ncommunities = 2, seed = 1)
  expect_s3_class(cr_spectral, "CommunityResult")
  expect_equal(length(unique(cr_spectral$membership)), 2)
  expect_equal(cr_spectral$algorithm, "spectral")
})

test_that("communities validates inputs and translates Julia errors", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  expect_error(communities(list()), "must be a MRIO object or a MRIOGraph object")
  expect_error(communities(g, algorithm = "spin"), "must be one of")
  expect_error(communities(g, resolution = "wide"), "must be a single number")
  expect_error(communities(g, nruns = "many"), "must be a single integer")
  expect_error(communities(g, source = "Z"), "must be NULL")
  expect_error(communities(g, ncommunities = 2), "Julia Error")
})

test_that("community_table maps nodes to communities", {
  fixture <- make_graph_fixture()
  cr <- communities(fixture$mrio, seed = 1)

  ct <- community_table(cr)
  expect_s3_class(ct, "data.frame")
  expect_equal(nrow(ct), 4)
  expect_true("community" %in% names(ct))
  expect_equal(as.integer(ct$community), cr$membership)
  expect_equal(as.character(ct$CountryCode), fixture$nodes$CountryCode)
  expect_equal(as.character(ct$Sector), fixture$nodes$Sector)

  expect_error(community_table(list()), "must be a CommunityResult object")
  expect_error(community_table(fixture$mrio), "must be a CommunityResult object")
})

test_that("community_summary summarizes each community", {
  fixture <- make_graph_fixture()
  cr <- communities(fixture$mrio, seed = 1)

  cs <- community_summary(cr)
  expect_s3_class(cs, "data.frame")
  expect_equal(nrow(cs), 2)
  expect_equal(
    names(cs)[1:5],
    c("community", "size", "internal_flow", "external_flow", "internal_share")
  )
  expect_equal(sort(as.integer(cs$community)), c(1L, 2L))
  expect_equal(sum(cs$size), 4)

  expect_error(community_summary(list()), "must be a CommunityResult object")
  expect_error(community_summary(cr$membership), "must be a CommunityResult object")
})

test_that("pagerank_scores returns a SeriesEntry with scores summing to one", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  pr <- pagerank_scores(g)
  expect_s3_class(pr, "SeriesEntry")
  pr_df <- as.data.frame(pr)
  expect_equal(names(pr_df), c("CountryCode", "Sector", "value"))
  expect_equal(sum(pr_df$value), 1, tolerance = 1e-6)

  pr_mrio <- pagerank_scores(fixture$mrio, source = "Z", weights = "flows")
  expect_s3_class(pr_mrio, "SeriesEntry")
  expect_equal(sum(as.data.frame(pr_mrio)$value), 1, tolerance = 1e-6)

  pr_unweighted <- pagerank_scores(g, weighted = FALSE)
  expect_s3_class(pr_unweighted, "SeriesEntry")
  expect_equal(sum(as.data.frame(pr_unweighted)$value), 1, tolerance = 1e-6)
})

test_that("pagerank_scores validates inputs and translates Julia errors", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  expect_error(pagerank_scores(list()), "must be a MRIO object or a MRIOGraph object")
  expect_error(pagerank_scores(g, damping = 1.5), "Julia Error")
  expect_error(pagerank_scores(g, damping = "fast"), "must be a single number")
  expect_error(pagerank_scores(g, weighted = "yes"), "must be TRUE or FALSE")
  expect_error(pagerank_scores(g, max_iter = "many"), "must be a single integer")
  expect_error(pagerank_scores(g, source = "Z"), "must be NULL")
})

test_that("node_similarity builds directed similarity graphs", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  sg <- node_similarity(g, k = 2)
  expect_s3_class(sg, "MRIOGraph")
  expect_equal(sg$nodes$CountryCode, fixture$nodes$CountryCode)

  sg_jaccard <- node_similarity(g, method = "jaccard", k = 2)
  expect_s3_class(sg_jaccard, "MRIOGraph")

  sg_walk <- node_similarity(g, method = "random_walk", sources = c(1L, 2L), k = 2)
  expect_s3_class(sg_walk, "MRIOGraph")

  sg_mrio <- node_similarity(fixture$mrio, source = "Z", weights = "flows", k = 2)
  expect_s3_class(sg_mrio, "MRIOGraph")
})

test_that("node_similarity validates inputs and translates Julia errors", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  expect_error(node_similarity(list()), "must be a MRIO object or a MRIOGraph object")
  expect_error(node_similarity(g, method = "euclidean"), "must be one of")
  expect_error(node_similarity(g, on = "sideways"), "must be one of")
  expect_error(node_similarity(g, sources = "node"), "must be a non-empty numeric vector")
  expect_error(node_similarity(g, sources = c(1L, NA)), "must be a non-empty numeric vector")
  expect_error(node_similarity(g, k = 0), "Julia Error")
  expect_error(node_similarity(g, method = "cosine", sources = c(1L, 2L)), "Julia Error")
  expect_error(node_similarity(g, method = "random_walk"), "Julia Error")
})

test_that("similarity_graph builds undirected graphs usable for clustering", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  sg <- similarity_graph(g, k = 2)
  expect_s3_class(sg, "MRIOGraph")
  expect_false(graph_summary(sg)$directed)

  cr <- communities(sg, seed = 1)
  expect_s3_class(cr, "CommunityResult")
  expect_equal(length(cr$membership), 4)

  sg_mean <- similarity_graph(g, k = 2, symmetrize = "mean")
  expect_s3_class(sg_mean, "MRIOGraph")

  sg_mrio <- similarity_graph(fixture$mrio, source = "Z", weights = "flows", k = 2)
  expect_s3_class(sg_mrio, "MRIOGraph")
})

test_that("similarity_graph validates inputs and translates Julia errors", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  expect_error(similarity_graph(list()), "must be a MRIO object or a MRIOGraph object")
  expect_error(similarity_graph(g, symmetrize = "median"), "must be one of")
  expect_error(similarity_graph(g, method = "euclidean"), "must be one of")
  expect_error(similarity_graph(g, on = "sideways"), "must be one of")
  expect_error(similarity_graph(g, k = 0), "Julia Error")
})

test_that("compare_networks compares identical graphs exactly", {
  fixture <- make_graph_fixture()
  g1 <- mrio_graph(fixture$W, fixture$nodes)
  g2 <- mrio_graph(fixture$W, fixture$nodes)

  cn <- compare_networks(g1, g2)
  expect_s3_class(cn, "data.frame")
  expect_equal(
    names(cn),
    c(
      "n_matched", "n_1", "n_2", "edge_overlap", "pearson_out", "spearman_out",
      "pearson_in", "spearman_in", "density_1", "density_2", "density_ratio",
      "scale_1", "scale_2", "scale_ratio"
    )
  )
  expect_equal(cn$n_matched, 4)
  expect_equal(cn$n_1, 4)
  expect_equal(cn$n_2, 4)
  expect_equal(cn$edge_overlap, 1, tolerance = 1e-12)

  cn_pagerank <- compare_networks(g1, g2, pagerank = TRUE)
  expect_equal(
    names(cn_pagerank),
    c(
      "n_matched", "n_1", "n_2", "edge_overlap", "pearson_out", "spearman_out",
      "pearson_in", "spearman_in", "density_1", "density_2", "density_ratio",
      "scale_1", "scale_2", "scale_ratio", "pearson_pagerank", "spearman_pagerank"
    )
  )
  expect_equal(cn_pagerank$pearson_pagerank, 1, tolerance = 1e-12)

  cn_keys <- compare_networks(g1, g2, match = c("CountryCode"))
  expect_equal(cn_keys$n_matched, 4)
  expect_equal(cn_keys$edge_overlap, 1, tolerance = 1e-12)
})

test_that("compare_networks validates inputs", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  expect_error(compare_networks(list(), g), "must be a MRIOGraph object")
  expect_error(compare_networks(g, list()), "must be a MRIOGraph object")
  expect_error(compare_networks(g, g, match = 5), "must be \"keys\" or a non-empty character vector")
  expect_error(compare_networks(g, g, match = character(0)), "must be \"keys\" or a non-empty character vector")
  expect_error(compare_networks(g, g, pagerank = NA), "must be TRUE or FALSE")
})

test_that("compare_partitions compares community partitions", {
  fixture <- make_graph_fixture()
  cr <- communities(fixture$mrio, seed = 1)

  cp <- compare_partitions(cr, cr)
  expect_s3_class(cp, "data.frame")
  expect_equal(
    names(cp),
    c(
      "n_matched", "n_1", "n_2", "ari", "nmi", "n_communities_1",
      "n_communities_2", "modularity_1", "modularity_2"
    )
  )
  expect_equal(cp$ari, 1, tolerance = 1e-12)
  expect_equal(cp$nmi, 1, tolerance = 1e-12)

  cp_match <- compare_partitions(cr, cr, match = c("CountryCode"))
  expect_equal(cp_match$ari, 1, tolerance = 1e-12)
})

test_that("compare_partitions compares raw label vectors positionally", {
  cp <- compare_partitions(c(1, 1, 1, 2, 2, 2), c(1, 1, 2, 2, 3, 3))
  expect_equal(cp$ari, 8 / 33, tolerance = 1e-12)
  expect_equal(cp$nmi, 4 * log(2) / (3 * log(6)), tolerance = 1e-12)

  cp2 <- compare_partitions(c(1, 1, 2, 2), c(1, 2, 2, 2))
  expect_equal(cp2$ari, 0, tolerance = 1e-12)
  expect_equal(cp2$nmi, 0.3437107, tolerance = 1e-6)
  expect_true(is.nan(cp2$modularity_1))
  expect_true(is.nan(cp2$modularity_2))

  fixture <- make_graph_fixture()
  cr <- communities(fixture$mrio, seed = 1)
  cp_mixed <- compare_partitions(cr, cr$membership)
  expect_equal(cp_mixed$ari, 1, tolerance = 1e-12)
  expect_true(is.nan(cp_mixed$modularity_2))
  expect_false(is.nan(cp_mixed$modularity_1))
})

test_that("compare_partitions validates inputs and translates Julia errors", {
  fixture <- make_graph_fixture()
  cr <- communities(fixture$mrio, seed = 1)

  expect_error(compare_partitions(list(), list()), "must be a CommunityResult object or a numeric vector")
  expect_error(compare_partitions(c("1", "2"), c("1", "2")), "must be a CommunityResult object or a numeric vector")
  expect_error(compare_partitions(cr, list()), "must be a CommunityResult object or a numeric vector")
  expect_error(compare_partitions(c(1, NA), c(1, 2)), "must not contain missing values")
  expect_error(compare_partitions(c(1, 1, 2), c(1, 2)), "Julia Error")
  expect_error(compare_partitions(cr, cr, match = 5), "must be \"keys\" or a non-empty character vector")
  expect_error(
    compare_partitions(c(1, 2), c(1, 2), match = c("CountryCode")),
    "only supported for CommunityResult"
  )
})

test_that("MRIOGraph and CommunityResult objects print summaries", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)
  expect_output(print(g), "MRIOGraph")
  expect_output(print(g), "4 nodes")

  cr <- communities(g, seed = 1)
  expect_output(print(cr), "CommunityResult")
  expect_output(print(cr), "2 communities")
  expect_output(print(cr), "louvain")
})

test_that("ensure_graphs_loaded returns invisible NULL when graphs are ready", {
  expect_invisible(ensure_graphs_loaded())
  expect_null(ensure_graphs_loaded())
})

# Node metadata with hostile-looking but legitimate column names: a literal
# "$(2019)" sequence (which Julia string interpolation would evaluate), a name
# that would raise a Julia error if interpolated into Julia source, and a name
# containing backtick and quote characters. All must round-trip as inert data.
make_graph_fixture_hostile_names <- function() {
  fixture <- make_graph_fixture()
  nodes <- data.frame(
    CountryCode = c("AUT", "DEU", "CHN", "GBR"),
    Sector = c("A", "B", "A", "B"),
    `GDP$(2019)` = c(10, 20, 30, 40),
    check.names = FALSE
  )
  nodes[['wage `"quoted"`']] <- c(4, 3, 2, 1)
  nodes[['x$(error("boom"))']] <- c(1, 2, 3, 4)
  list(mrio = fixture$mrio, nodes = nodes, W = fixture$W)
}

test_that("column names with $(), quotes and backticks never reach Julia code", {
  fixture <- make_graph_fixture_hostile_names()
  g <- mrio_graph(fixture$W, fixture$nodes)

  hostile <- c("GDP$(2019)", 'wage `"quoted"`', 'x$(error("boom"))')
  expect_true(all(hostile %in% names(g$nodes)))

  cn <- compare_networks(g, g, match = "GDP$(2019)")
  expect_equal(cn$n_matched, 4)
  expect_equal(cn$edge_overlap, 1, tolerance = 1e-12)

  cn_quote <- compare_networks(g, g, match = 'wage `"quoted"`')
  expect_equal(cn_quote$n_matched, 4)

  # This name would raise "Julia Error: ..." if it were interpolated into a
  # Julia string literal: error("boom") would execute during Symbol construction.
  cn_exec <- compare_networks(g, g, match = 'x$(error("boom"))')
  expect_equal(cn_exec$n_matched, 4)

  cn_multi <- compare_networks(g, g, match = c("GDP$(2019)", 'wage `"quoted"`'))
  expect_equal(cn_multi$n_matched, 4)

  cr <- communities(g, seed = 1)
  cp <- compare_partitions(cr, cr, match = "GDP$(2019)")
  expect_equal(cp$n_matched, 4)
  expect_equal(cp$ari, 1, tolerance = 1e-12)
})

test_that("integer validators reject fractions, non-finite and out-of-range values", {
  fixture <- make_graph_fixture()
  g <- mrio_graph(fixture$W, fixture$nodes)

  # All of these are R-side validation errors, not translated "Julia Error"
  # conditions, and none of them may be silently coerced by as.integer.
  expect_error(communities(g, nruns = 1.9), "must be a single integer")
  expect_error(communities(g, seed = Inf), "must be a single integer")
  expect_error(communities(g, ncommunities = 2.5), "must be a single integer")
  expect_error(node_similarity(g, k = Inf), "must be a single integer")
  expect_error(node_similarity(g, max_iter = 2147483648), "must be a single integer")
  expect_error(
    node_similarity(g, method = "random_walk", sources = c(1.5, 2)),
    "must be a non-empty numeric vector"
  )
  expect_error(
    compare_partitions(c(1.5, 1.5, 2, 2), c(1, 1, 2, 2)),
    "whole numbers within the integer range"
  )
  expect_error(
    compare_partitions(c(1, 1, 2, 2), c(1.5, 1.5, 2, 2)),
    "whole numbers within the integer range"
  )

  # Whole labels including negatives stay legal per the Julia contract
  cp_negative <- compare_partitions(c(-1, -1, 2, 2), c(-1, -1, 2, 2))
  expect_equal(cp_negative$ari, 1, tolerance = 1e-12)

  err <- tryCatch(communities(g, nruns = 1.9), error = identity)
  expect_s3_class(err, "error")
  expect_false(grepl("Julia Error", conditionMessage(err), fixed = TRUE))
})
