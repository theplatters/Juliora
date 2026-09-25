#' Ensure the Graphs.jl package extension is loaded
#'
#' @title Ensure graph support is loaded
#' @description Establish the Julia connection and load the Graphs.jl package
#'   extension that provides the graph API. The result is cached: re-running
#'   `using Graphs` re-resolves Graphs against the active Julia project and can
#'   fail even when Graphs is already loaded in the session, so the load is
#'   only attempted once per R session.
#'
#' @return Invisible NULL.
#' @export
#'
#' @examples
#' \dontrun{
#' ensure_graphs_loaded()
#' }
ensure_graphs_loaded <- function() {
  get_julia_connection()
  if (!isTRUE(.juliora_env$graphs_loaded)) {
    tryCatch(JuliaConnectoR::juliaEval("using Graphs"),
      error = function(e) {
        stop(
          "Graph features require the Graphs.jl package extension, which could not be loaded. ",
          "Install Graphs into the active Julia project ",
          "(import Pkg; Pkg.add(\"Graphs\")) and try again.",
          call. = FALSE
        )
      }
    )
    .juliora_env$graphs_loaded <- TRUE
  }
  invisible(NULL)
}

# --- Internal argument helpers ---

#' Convert a character string to a Julia Symbol
#'
#' The value is passed as a call argument (never interpolated into Julia
#' source), so arbitrary user strings are safe. Julia Symbols transfer as R
#' name objects; pass them to `juliaCall` directly or through
#' `do.call(..., quote = TRUE)`.
#'
#' @param value A single character string.
#' @return A Julia Symbol for `Symbol(value)`.
#' @noRd
graph_symbol <- function(value) {
  JuliaConnectoR::juliaCall("Base.Symbol", value)
}

#' Convert a character vector to a Julia Vector\{Symbol\}
#'
#' The values are passed as call arguments (never interpolated into Julia
#' source), so arbitrary user strings are safe.
#'
#' @param values A character vector of names.
#' @return A Julia proxy for `Vector{Symbol}` of `Symbol.(values)`.
#' @noRd
graph_symbols <- function(values) {
  JuliaConnectoR::juliaCall(
    "Vector{Symbol}",
    lapply(values, function(v) JuliaConnectoR::juliaCall("Base.Symbol", v))
  )
}

#' Build the Julia keyword arguments selecting matrix and graph filter
#'
#' @param source Optional source matrix name.
#' @param weights Optional weights matrix name.
#' @param direction "directed" or "undirected".
#' @param threshold Absolute weight cutoff.
#' @param min_share Minimum weight share.
#' @param self_loops Whether to keep self-flows.
#' @return A named list of Julia keyword arguments.
#' @noRd
graph_filter_kwargs <- function(source, weights, direction, threshold, min_share, self_loops) {
  kwargs <- list(
    direction = graph_symbol(direction),
    threshold = threshold,
    min_share = min_share,
    self_loops = self_loops
  )
  if (!is.null(source)) {
    kwargs$source <- graph_symbol(source)
  }
  if (!is.null(weights)) {
    kwargs$weights <- graph_symbol(weights)
  }
  kwargs
}

#' Validate a single character string argument
#'
#' @param value The value to check.
#' @param name The argument name used in error messages.
#' @return The validated value.
#' @noRd
graph_check_string <- function(value, name) {
  if (!is.character(value) || length(value) != 1 || is.na(value)) {
    stop("Argument '", name, "' must be a single character string.", call. = FALSE)
  }
  value
}

#' Validate a single string against a whitelist
#'
#' @param value The value to check.
#' @param name The argument name used in error messages.
#' @param choices The allowed values.
#' @return The validated value.
#' @noRd
graph_check_choice <- function(value, name, choices) {
  if (!is.character(value) || length(value) != 1 || is.na(value) || !value %in% choices) {
    stop(
      "Argument '", name, "' must be one of ",
      paste0('"', choices, '"', collapse = ", "), ".",
      call. = FALSE
    )
  }
  value
}

#' Validate a single numeric scalar argument
#'
#' @param value The value to check.
#' @param name The argument name used in error messages.
#' @return The value as a numeric scalar.
#' @noRd
graph_check_number <- function(value, name) {
  if (!is.numeric(value) || length(value) != 1 || is.na(value)) {
    stop("Argument '", name, "' must be a single number.", call. = FALSE)
  }
  as.numeric(value)
}

#' Validate a single integer scalar argument
#'
#' Requires a whole number (no fractional part) within R's integer range so
#' the value round-trips as a 32-bit integer; rejects `NA`, `Inf` and
#' fractions instead of silently coercing them.
#'
#' @param value The value to check.
#' @param name The argument name used in error messages.
#' @return The value as an integer scalar.
#' @noRd
graph_check_integer <- function(value, name) {
  if (!is.numeric(value) || length(value) != 1 || is.na(value) ||
    !is.finite(value) || value != trunc(value) || abs(value) > .Machine$integer.max) {
    stop(
      "Argument '", name,
      "' must be a single integer (a whole number within the integer range).",
      call. = FALSE
    )
  }
  as.integer(value)
}

#' Validate a single logical scalar argument
#'
#' @param value The value to check.
#' @param name The argument name used in error messages.
#' @return The validated value.
#' @noRd
graph_check_flag <- function(value, name) {
  if (!is.logical(value) || length(value) != 1 || is.na(value)) {
    stop("Argument '", name, "' must be TRUE or FALSE.", call. = FALSE)
  }
  value
}

#' Validate a non-empty integer vector argument
#'
#' Requires whole numbers (no fractional part) within R's integer range so
#' the values round-trip as 32-bit integers; rejects `NA`, `Inf` and fractions
#' instead of silently coercing them.
#'
#' @param value The value to check.
#' @param name The argument name used in error messages.
#' @return The value as an integer vector.
#' @noRd
graph_check_integer_vector <- function(value, name) {
  if (!is.numeric(value) || length(value) < 1 || anyNA(value) ||
    any(!is.finite(value)) || any(value != trunc(value)) || any(abs(value) > .Machine$integer.max)) {
    stop(
      "Argument '", name,
      "' must be a non-empty numeric vector of whole numbers within the integer range, ",
      "without missing values.",
      call. = FALSE
    )
  }
  as.integer(value)
}

#' Check that x is a MRIO or MRIOGraph object and that source/weights are
#' compatible with it
#'
#' @param x The input object.
#' @param source Optional source matrix name.
#' @param weights Optional weights matrix name.
#' @return TRUE when `x` is a MRIOGraph (no filter keywords to forward),
#'   FALSE when `x` is a MRIO object.
#' @noRd
graph_check_input <- function(x, source, weights) {
  if (inherits(x, "MRIOGraph")) {
    if (!is.null(source)) {
      stop("Argument 'source' must be NULL when 'x' is a MRIOGraph object.", call. = FALSE)
    }
    if (!is.null(weights)) {
      stop("Argument 'weights' must be NULL when 'x' is a MRIOGraph object.", call. = FALSE)
    }
    return(TRUE)
  }
  if (inherits(x, "MRIO")) {
    return(FALSE)
  }
  stop("Argument 'x' must be a MRIO object or a MRIOGraph object.", call. = FALSE)
}

#' Validate a partition label vector argument
#'
#' Labels must be whole numbers within R's integer range (arbitrary whole
#' labels including negatives are legal); fractional labels are rejected
#' instead of silently collapsing distinct labels through `as.integer`.
#'
#' @param value The value to check.
#' @param name The argument name used in error messages.
#' @return The value as an integer vector.
#' @noRd
graph_check_partition <- function(value, name) {
  if (!is.numeric(value) || !is.null(dim(value))) {
    stop("Argument '", name, "' must be a CommunityResult object or a numeric vector of partition labels.", call. = FALSE)
  }
  if (anyNA(value)) {
    stop("Argument '", name, "' must not contain missing values.", call. = FALSE)
  }
  if (any(!is.finite(value)) || any(value != trunc(value)) || any(abs(value) > .Machine$integer.max)) {
    stop(
      "Argument '", name,
      "' must contain only whole numbers within the integer range.",
      call. = FALSE
    )
  }
  as.integer(value)
}

#' Validate the node matching keys argument
#'
#' @param value The value to check.
#' @return "keys" or a character vector of column names.
#' @noRd
graph_check_match <- function(value) {
  if (is.character(value) && length(value) == 1 && !is.na(value) && identical(value, "keys")) {
    return(value)
  }
  if (!is.character(value) || length(value) < 1 || anyNA(value) || !all(nzchar(value))) {
    stop("Argument 'match' must be \"keys\" or a non-empty character vector of column names.", call. = FALSE)
  }
  value
}

#' Convert a Julia DataFrame result to a plain R data.frame
#'
#' Drops the JuliaConnectoR round-trip attributes (`JLDIM`, `JLTYPE`) that
#' otherwise leak onto transferred columns (they are only needed to send
#' arrays back to Julia intact). Also restores the exact Julia column names:
#' the JuliaConnectoR conversion runs `make.names()`, which would mangle
#' non-syntactic names such as `GDP$(2019)`.
#'
#' @param res A Julia DataFrame proxy.
#' @return A plain data.frame.
#' @noRd
graph_as_dataframe <- function(res) {
  df <- as.data.frame(res)
  names(df) <- JuliaConnectoR::juliaCall("names", res)
  df[] <- lapply(df, function(col) {
    attr(col, "JLDIM") <- NULL
    attr(col, "JLTYPE") <- NULL
    col
  })
  df
}

# --- Graph wrappers ---

#' Build a graph view over MRIO data
#'
#' @title Build MRIO graph
#' @description Build a graph view over a dense MRIO matrix or an
#'   MRIO database; R inputs are transferred to Julia first, and only the
#'   Julia-side construction adds no further copy. An edge `i -> j` means
#'   monetary flow supplied by node `i` to buyer `j` (`Z[i, j]` of the wrapped
#'   transactions matrix). Self-flows are dropped by default. The wrapped
#'   matrix is stored by reference on the Julia side, never copied.
#'
#' @param x A MRIO object, or a numeric matrix of pairwise flows.
#' @param nodes A data.frame of node metadata with one row per node. Required
#'   when `x` is a numeric matrix, must be NULL for MRIO input.
#' @param source A character string selecting the source matrix for MRIO
#'   input: "Z", "T" or "A" (default NULL selects the transactions matrix).
#'   Must be NULL for matrix input. Values are validated by Julia; an invalid
#'   value raises a Julia error.
#' @param weights A character string selecting the weights matrix for MRIO
#'   input: "flows" or "technical" (default NULL selects "flows"). Must be
#'   NULL for matrix input. Values are validated by Julia; an invalid value
#'   raises a Julia error.
#' @param direction A character string: "directed" (default) or "undirected".
#' @param threshold A single number: absolute cutoff on `|w|` (default 0).
#' @param min_share A single number: minimum share of the total absolute
#'   weight (default 0).
#' @param self_loops A logical value: keep self-flows (default FALSE).
#'
#' @return An MRIOGraph object wrapping the Julia MRIOGraph.
#' @export
#'
#' @examples
#' \dontrun{
#' g <- mrio_graph(mrio, source = "Z", weights = "flows")
#' g <- mrio_graph(flow_matrix, nodes)
#' }
mrio_graph <- function(x, nodes = NULL, source = NULL, weights = NULL, direction = "directed",
                       threshold = 0, min_share = 0, self_loops = FALSE) {
  direction <- graph_check_choice(direction, "direction", c("directed", "undirected"))
  threshold <- graph_check_number(threshold, "threshold")
  min_share <- graph_check_number(min_share, "min_share")
  self_loops <- graph_check_flag(self_loops, "self_loops")
  if (!is.null(source)) {
    source <- graph_check_string(source, "source")
  }
  if (!is.null(weights)) {
    weights <- graph_check_string(weights, "weights")
  }

  matrix_input <- is.matrix(x) && is.numeric(x)
  if (!matrix_input && !inherits(x, "MRIO")) {
    stop("Argument 'x' must be a MRIO object or a numeric matrix.", call. = FALSE)
  }
  if (matrix_input) {
    if (!is.data.frame(nodes)) {
      stop("Argument 'nodes' must be a data.frame when 'x' is a numeric matrix.", call. = FALSE)
    }
    if (!is.null(source)) {
      stop("Argument 'source' must be NULL when 'x' is a numeric matrix.", call. = FALSE)
    }
    if (!is.null(weights)) {
      stop("Argument 'weights' must be NULL when 'x' is a numeric matrix.", call. = FALSE)
    }
  } else if (!is.null(nodes)) {
    stop("Argument 'nodes' must be NULL when 'x' is a MRIO object.", call. = FALSE)
  }

  ensure_graphs_loaded()

  filter_kwargs <- graph_filter_kwargs(source, weights, direction, threshold, min_share, self_loops)

  res <- tryCatch({
    # quote = TRUE keeps the Julia Symbol keyword arguments (R name objects)
    # from being evaluated as variables by do.call
    if (matrix_input) {
      nodes_jl <- JuliaConnectoR::juliaCall("Juliora.safe_dataframe", nodes)
      do.call(JuliaConnectoR::juliaCall, c(list("Juliora.mrio_graph", x, nodes_jl), filter_kwargs), quote = TRUE)
    } else {
      do.call(JuliaConnectoR::juliaCall, c(list("Juliora.mrio_graph", unwrap_julia_object(x)), filter_kwargs), quote = TRUE)
    }
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  wrap_julia_object(res)
}

#' Summarize an MRIO graph
#'
#' @title Summarize MRIO graph
#' @description Summarize an MRIOGraph in a one-row data.frame with the node
#'   and edge counts, the graph filter, the total and retained weights and the
#'   memory the wrapped matrix references.
#'
#' @param g An MRIOGraph object.
#'
#' @return A one-row data.frame with columns `nodes`, `edges`, `directed`,
#'   `threshold`, `min_share`, `self_loops`, `total_weight`,
#'   `retained_weight`, `retained_share` and `memory_bytes`.
#' @export
#'
#' @examples
#' \dontrun{
#' graph_summary(g)
#' }
graph_summary <- function(g) {
  if (!inherits(g, "MRIOGraph")) {
    stop("Argument 'g' must be a MRIOGraph object.", call. = FALSE)
  }

  ensure_graphs_loaded()

  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.graph_summary", unwrap_julia_object(g))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  graph_as_dataframe(res)
}

#' Detect communities in an MRIO network
#'
#' @title Detect communities
#' @description Detect communities in an MRIO network and return the partition
#'   with its modularity. Community detection runs on the symmetrized flow
#'   weights by default: for MRIO input `direction` defaults to "undirected"
#'   here (unlike `mrio_graph` and `pagerank_scores`).
#'
#' @param x A MRIO object or an MRIOGraph object. For a MRIOGraph the graph
#'   filter is taken as built; for a MRIO object the graph is built from
#'   `source`, `weights`, `direction`, `threshold`, `min_share` and
#'   `self_loops`.
#' @param algorithm A character string: "louvain" (default), "leiden",
#'   "label_propagation" or "spectral".
#' @param resolution A single number: modularity resolution parameter
#'   (default 1).
#' @param nruns A single integer: number of independent runs (default 1); the
#'   partition with the largest modularity is kept.
#' @param seed NULL (default) or a single integer RNG seed.
#' @param ncommunities NULL (default) or a single integer target community
#'   count; only valid with `algorithm = "spectral"` (Julia raises an error
#'   otherwise).
#' @param source A character string selecting the source matrix for MRIO
#'   input: "Z", "T" or "A" (default NULL selects the transactions matrix).
#'   Must be NULL for MRIOGraph input.
#' @param weights A character string selecting the weights matrix for MRIO
#'   input: "flows" or "technical" (default NULL selects "flows"). Must be
#'   NULL for MRIOGraph input.
#' @param direction A character string: "directed" or "undirected" (default).
#'   Ignored for MRIOGraph input.
#' @param threshold A single number: absolute cutoff on `|w|` (default 0).
#'   Ignored for MRIOGraph input.
#' @param min_share A single number: minimum share of the total absolute
#'   weight (default 0). Ignored for MRIOGraph input.
#' @param self_loops A logical value: keep self-flows (default FALSE). Ignored
#'   for MRIOGraph input.
#'
#' @return A CommunityResult object holding the partition.
#' @export
#'
#' @examples
#' \dontrun{
#' cr <- communities(mrio, seed = 1)
#' cr <- communities(g, algorithm = "spectral", ncommunities = 2)
#' }
communities <- function(x, algorithm = "louvain", resolution = 1, nruns = 1, seed = NULL,
                        ncommunities = NULL, source = NULL, weights = NULL,
                        direction = "undirected", threshold = 0, min_share = 0,
                        self_loops = FALSE) {
  algorithm <- graph_check_choice(
    algorithm, "algorithm",
    c("louvain", "leiden", "label_propagation", "spectral")
  )
  resolution <- graph_check_number(resolution, "resolution")
  nruns <- graph_check_integer(nruns, "nruns")
  if (!is.null(seed)) {
    seed <- graph_check_integer(seed, "seed")
  }
  if (!is.null(ncommunities)) {
    ncommunities <- graph_check_integer(ncommunities, "ncommunities")
  }
  if (!is.null(source)) {
    source <- graph_check_string(source, "source")
  }
  if (!is.null(weights)) {
    weights <- graph_check_string(weights, "weights")
  }
  direction <- graph_check_choice(direction, "direction", c("directed", "undirected"))
  threshold <- graph_check_number(threshold, "threshold")
  min_share <- graph_check_number(min_share, "min_share")
  self_loops <- graph_check_flag(self_loops, "self_loops")
  graph_input <- graph_check_input(x, source, weights)

  ensure_graphs_loaded()

  call_args <- list("Juliora.communities", unwrap_julia_object(x))
  if (!graph_input) {
    call_args <- c(call_args, graph_filter_kwargs(source, weights, direction, threshold, min_share, self_loops))
  }
  call_args <- c(call_args, list(
    algorithm = graph_symbol(algorithm),
    resolution = resolution,
    nruns = nruns
  ))
  if (!is.null(seed)) {
    call_args$seed <- seed
  }
  if (!is.null(ncommunities)) {
    call_args$ncommunities <- ncommunities
  }

  res <- tryCatch({
    do.call(JuliaConnectoR::juliaCall, call_args, quote = TRUE)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  wrap_julia_object(res)
}

#' Node-to-community mapping of a community partition
#'
#' @title Community table
#' @description Return the node-to-community mapping of a CommunityResult
#'   joined with the node metadata.
#'
#' @param result A CommunityResult object.
#'
#' @return A data.frame with the node metadata columns plus a `community`
#'   column.
#' @export
#'
#' @examples
#' \dontrun{
#' community_table(cr)
#' }
community_table <- function(result) {
  if (!inherits(result, "CommunityResult")) {
    stop("Argument 'result' must be a CommunityResult object.", call. = FALSE)
  }

  ensure_graphs_loaded()

  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.community_table", unwrap_julia_object(result))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  graph_as_dataframe(res)
}

#' Summarize a community partition
#'
#' @title Community summary
#' @description Summarize a CommunityResult with one row per community and
#'   columns `community`, `size`, `internal_flow`, `external_flow` and
#'   `internal_share`, plus country and sector breakdown columns when the
#'   corresponding metadata columns exist in the node table.
#'
#' @param result A CommunityResult object.
#'
#' @return A data.frame with one row per community.
#' @export
#'
#' @examples
#' \dontrun{
#' community_summary(cr)
#' }
community_summary <- function(result) {
  if (!inherits(result, "CommunityResult")) {
    stop("Argument 'result' must be a CommunityResult object.", call. = FALSE)
  }

  ensure_graphs_loaded()

  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.community_summary", unwrap_julia_object(result))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  graph_as_dataframe(res)
}

#' PageRank scores over an MRIO network
#'
#' @title PageRank scores
#' @description Compute flow-weighted PageRank scores over the directed MRIO
#'   graph. Scores are non-negative and sum to 1.
#'
#' @param x A MRIO object or an MRIOGraph object. For a MRIOGraph the graph
#'   filter is taken as built; for a MRIO object the graph is built from
#'   `source`, `weights`, `direction`, `threshold`, `min_share` and
#'   `self_loops`.
#' @param source A character string selecting the source matrix for MRIO
#'   input: "Z", "T" or "A" (default NULL selects the transactions matrix).
#'   Must be NULL for MRIOGraph input.
#' @param weights A character string selecting the weights matrix for MRIO
#'   input: "flows" or "technical" (default NULL selects "flows"). Must be
#'   NULL for MRIOGraph input.
#' @param direction A character string: "directed" (default) or "undirected".
#'   Ignored for MRIOGraph input.
#' @param threshold A single number: absolute cutoff on `|w|` (default 0).
#'   Ignored for MRIOGraph input.
#' @param min_share A single number: minimum share of the total absolute
#'   weight (default 0). Ignored for MRIOGraph input.
#' @param self_loops A logical value: keep self-flows (default FALSE). Ignored
#'   for MRIOGraph input.
#' @param damping A single number strictly between 0 and 1 (default 0.85);
#'   validated by Julia.
#' @param weighted A logical value: use the flow-weighted power iteration
#'   (default TRUE) or the unweighted topology (FALSE).
#' @param tol A single number: convergence tolerance (default 1e-6).
#' @param max_iter A single integer: iteration cap (default 100).
#'
#' @return A SeriesEntry with the node metadata and one score per node.
#' @export
#'
#' @examples
#' \dontrun{
#' pr <- pagerank_scores(mrio)
#' pr <- pagerank_scores(g, weighted = FALSE)
#' }
pagerank_scores <- function(x, source = NULL, weights = NULL, direction = "directed",
                            threshold = 0, min_share = 0, self_loops = FALSE,
                            damping = 0.85, weighted = TRUE, tol = 1e-6, max_iter = 100) {
  if (!is.null(source)) {
    source <- graph_check_string(source, "source")
  }
  if (!is.null(weights)) {
    weights <- graph_check_string(weights, "weights")
  }
  direction <- graph_check_choice(direction, "direction", c("directed", "undirected"))
  threshold <- graph_check_number(threshold, "threshold")
  min_share <- graph_check_number(min_share, "min_share")
  self_loops <- graph_check_flag(self_loops, "self_loops")
  damping <- graph_check_number(damping, "damping")
  weighted <- graph_check_flag(weighted, "weighted")
  tol <- graph_check_number(tol, "tol")
  max_iter <- graph_check_integer(max_iter, "max_iter")
  graph_input <- graph_check_input(x, source, weights)

  ensure_graphs_loaded()

  call_args <- list("Juliora.pagerank_scores", unwrap_julia_object(x))
  if (!graph_input) {
    call_args <- c(call_args, graph_filter_kwargs(source, weights, direction, threshold, min_share, self_loops))
  }
  call_args <- c(call_args, list(
    damping = damping,
    weighted = weighted,
    tol = tol,
    max_iter = max_iter
  ))

  res <- tryCatch({
    do.call(JuliaConnectoR::juliaCall, call_args, quote = TRUE)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  wrap_julia_object(res)
}

#' Structural node similarity over an MRIO network
#'
#' @title Node similarity
#' @description Top-k structural node similarity over the graph's effective
#'   weights: cosine or Jaccard similarity of the flow profiles, or
#'   personalized-PageRank relatedness for selected source nodes. Returns a
#'   directed kNN similarity graph whose edge weights are the similarities.
#'
#' @param x A MRIO object or an MRIOGraph object. For a MRIOGraph the graph
#'   filter is taken as built; for a MRIO object the graph is built from
#'   `source`, `weights`, `direction`, `threshold`, `min_share` and
#'   `self_loops`.
#' @param source A character string selecting the source matrix for MRIO
#'   input: "Z", "T" or "A" (default NULL selects the transactions matrix).
#'   Must be NULL for MRIOGraph input.
#' @param weights A character string selecting the weights matrix for MRIO
#'   input: "flows" or "technical" (default NULL selects "flows"). Must be
#'   NULL for MRIOGraph input.
#' @param direction A character string: "directed" (default) or "undirected".
#'   Ignored for MRIOGraph input.
#' @param threshold A single number: absolute cutoff on `|w|` (default 0).
#'   Ignored for MRIOGraph input.
#' @param min_share A single number: minimum share of the total absolute
#'   weight (default 0). Ignored for MRIOGraph input.
#' @param self_loops A logical value: keep self-flows (default FALSE). Ignored
#'   for MRIOGraph input.
#' @param method A character string: "cosine" (default), "jaccard" or
#'   "random_walk".
#' @param on A character string selecting the flow profiles: "out" (default),
#'   "in" or "both". Validated always, ignored by "random_walk".
#' @param k A single integer: neighbors kept per row (default 10); validated
#'   by Julia.
#' @param sources NULL (default), or a non-empty integer vector of node
#'   indices. Required with `method = "random_walk"` and rejected with any
#'   other method (Julia raises an error otherwise).
#' @param damping A single number strictly between 0 and 1 (default 0.85),
#'   used by "random_walk" only; validated by Julia.
#' @param tol A single number: convergence tolerance (default 1e-6), used by
#'   "random_walk" only.
#' @param max_iter A single integer: iteration cap (default 100), used by
#'   "random_walk" only.
#'
#' @return An MRIOGraph object: a directed kNN similarity graph over the node
#'   metadata of `x`.
#' @export
#'
#' @examples
#' \dontrun{
#' sg <- node_similarity(g, k = 5)
#' sg <- node_similarity(mrio, method = "random_walk", sources = c(1, 2))
#' }
node_similarity <- function(x, source = NULL, weights = NULL, direction = "directed",
                            threshold = 0, min_share = 0, self_loops = FALSE,
                            method = "cosine", on = "out", k = 10, sources = NULL,
                            damping = 0.85, tol = 1e-6, max_iter = 100) {
  if (!is.null(source)) {
    source <- graph_check_string(source, "source")
  }
  if (!is.null(weights)) {
    weights <- graph_check_string(weights, "weights")
  }
  direction <- graph_check_choice(direction, "direction", c("directed", "undirected"))
  threshold <- graph_check_number(threshold, "threshold")
  min_share <- graph_check_number(min_share, "min_share")
  self_loops <- graph_check_flag(self_loops, "self_loops")
  method <- graph_check_choice(method, "method", c("cosine", "jaccard", "random_walk"))
  on <- graph_check_choice(on, "on", c("out", "in", "both"))
  k <- graph_check_integer(k, "k")
  if (!is.null(sources)) {
    sources <- graph_check_integer_vector(sources, "sources")
  }
  damping <- graph_check_number(damping, "damping")
  tol <- graph_check_number(tol, "tol")
  max_iter <- graph_check_integer(max_iter, "max_iter")
  graph_input <- graph_check_input(x, source, weights)

  ensure_graphs_loaded()

  call_args <- list("Juliora.node_similarity", unwrap_julia_object(x))
  if (!graph_input) {
    call_args <- c(call_args, graph_filter_kwargs(source, weights, direction, threshold, min_share, self_loops))
  }
  call_args <- c(call_args, list(
    method = graph_symbol(method),
    on = graph_symbol(on),
    k = k,
    damping = damping,
    tol = tol,
    max_iter = max_iter
  ))
  if (!is.null(sources)) {
    call_args$sources <- sources
  }

  res <- tryCatch({
    do.call(JuliaConnectoR::juliaCall, call_args, quote = TRUE)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  wrap_julia_object(res)
}

#' Symmetric kNN similarity graph over an MRIO network
#'
#' @title Similarity graph
#' @description Run `node_similarity` and symmetrize the kept directed edges
#'   into an undirected kNN similarity graph. The result is intended as input
#'   to `communities` for clustering nodes with similar flow profiles.
#'
#' @param x A MRIO object or an MRIOGraph object. For a MRIOGraph the graph
#'   filter is taken as built; for a MRIO object the graph is built from
#'   `source`, `weights`, `direction`, `threshold`, `min_share` and
#'   `self_loops`.
#' @param source A character string selecting the source matrix for MRIO
#'   input: "Z", "T" or "A" (default NULL selects the transactions matrix).
#'   Must be NULL for MRIOGraph input.
#' @param weights A character string selecting the weights matrix for MRIO
#'   input: "flows" or "technical" (default NULL selects "flows"). Must be
#'   NULL for MRIOGraph input.
#' @param direction A character string: "directed" (default) or "undirected".
#'   Ignored for MRIOGraph input.
#' @param threshold A single number: absolute cutoff on `|w|` (default 0).
#'   Ignored for MRIOGraph input.
#' @param min_share A single number: minimum share of the total absolute
#'   weight (default 0). Ignored for MRIOGraph input.
#' @param self_loops A logical value: keep self-flows (default FALSE). Ignored
#'   for MRIOGraph input.
#' @param method A character string: "cosine" (default), "jaccard" or
#'   "random_walk".
#' @param on A character string selecting the flow profiles: "out" (default),
#'   "in" or "both". Validated always, ignored by "random_walk".
#' @param k A single integer: neighbors kept per row (default 10); validated
#'   by Julia.
#' @param sources NULL (default), or a non-empty integer vector of node
#'   indices. Required with `method = "random_walk"` and rejected with any
#'   other method (Julia raises an error otherwise).
#' @param damping A single number strictly between 0 and 1 (default 0.85),
#'   used by "random_walk" only; validated by Julia.
#' @param tol A single number: convergence tolerance (default 1e-6), used by
#'   "random_walk" only.
#' @param max_iter A single integer: iteration cap (default 100), used by
#'   "random_walk" only.
#' @param symmetrize A character string: "max" (default) or "mean", how the
#'   two directed values of a kept pair are combined.
#'
#' @return An MRIOGraph object: an undirected kNN similarity graph over the
#'   node metadata of `x`.
#' @export
#'
#' @examples
#' \dontrun{
#' sg <- similarity_graph(g, k = 5)
#' cr <- communities(sg, seed = 1)
#' }
similarity_graph <- function(x, source = NULL, weights = NULL, direction = "directed",
                             threshold = 0, min_share = 0, self_loops = FALSE,
                             method = "cosine", on = "out", k = 10, sources = NULL,
                             damping = 0.85, tol = 1e-6, max_iter = 100, symmetrize = "max") {
  if (!is.null(source)) {
    source <- graph_check_string(source, "source")
  }
  if (!is.null(weights)) {
    weights <- graph_check_string(weights, "weights")
  }
  direction <- graph_check_choice(direction, "direction", c("directed", "undirected"))
  threshold <- graph_check_number(threshold, "threshold")
  min_share <- graph_check_number(min_share, "min_share")
  self_loops <- graph_check_flag(self_loops, "self_loops")
  method <- graph_check_choice(method, "method", c("cosine", "jaccard", "random_walk"))
  on <- graph_check_choice(on, "on", c("out", "in", "both"))
  k <- graph_check_integer(k, "k")
  if (!is.null(sources)) {
    sources <- graph_check_integer_vector(sources, "sources")
  }
  damping <- graph_check_number(damping, "damping")
  tol <- graph_check_number(tol, "tol")
  max_iter <- graph_check_integer(max_iter, "max_iter")
  symmetrize <- graph_check_choice(symmetrize, "symmetrize", c("max", "mean"))
  graph_input <- graph_check_input(x, source, weights)

  ensure_graphs_loaded()

  call_args <- list("Juliora.similarity_graph", unwrap_julia_object(x))
  if (!graph_input) {
    call_args <- c(call_args, graph_filter_kwargs(source, weights, direction, threshold, min_share, self_loops))
  }
  call_args <- c(call_args, list(
    method = graph_symbol(method),
    on = graph_symbol(on),
    k = k,
    damping = damping,
    tol = tol,
    max_iter = max_iter,
    symmetrize = graph_symbol(symmetrize)
  ))
  if (!is.null(sources)) {
    call_args$sources <- sources
  }

  res <- tryCatch({
    do.call(JuliaConnectoR::juliaCall, call_args, quote = TRUE)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  wrap_julia_object(res)
}

#' Compare two MRIO networks
#'
#' @title Compare networks
#' @description Compare two MRIO networks on their shared (matched) nodes:
#'   weighted edge overlap, out/in strength correlations, densities and scale
#'   ratios, and optionally PageRank correlations. Graphs only — build the
#'   graphs first with `mrio_graph`.
#'
#' @param g1 An MRIOGraph object.
#' @param g2 An MRIOGraph object.
#' @param match The node matching keys: the single string "keys" (the default;
#'   matches on every column present in both node tables) or a non-empty
#'   character vector of column names to match on. A column literally named
#'   "keys" cannot be selected explicitly.
#' @param pagerank A logical value: add `pearson_pagerank` and
#'   `spearman_pagerank` columns from a PageRank solve per matched subgraph
#'   (default FALSE).
#' @param damping A single number strictly between 0 and 1 (default 0.85),
#'   used when `pagerank` is TRUE; validated by Julia.
#'
#' @return A one-row data.frame with the comparison metrics.
#' @export
#'
#' @examples
#' \dontrun{
#' compare_networks(g1, g2)
#' compare_networks(g1, g2, match = c("CountryCode"), pagerank = TRUE)
#' }
compare_networks <- function(g1, g2, match = "keys", pagerank = FALSE, damping = 0.85) {
  if (!inherits(g1, "MRIOGraph")) {
    stop("Argument 'g1' must be a MRIOGraph object.", call. = FALSE)
  }
  if (!inherits(g2, "MRIOGraph")) {
    stop("Argument 'g2' must be a MRIOGraph object.", call. = FALSE)
  }
  match <- graph_check_match(match)
  pagerank <- graph_check_flag(pagerank, "pagerank")
  damping <- graph_check_number(damping, "damping")

  ensure_graphs_loaded()

  match_jl <- if (identical(match, "keys")) graph_symbol(match) else graph_symbols(match)

  res <- tryCatch({
    JuliaConnectoR::juliaCall(
      "Juliora.compare_networks",
      unwrap_julia_object(g1),
      unwrap_julia_object(g2),
      match = match_jl,
      pagerank = pagerank,
      damping = damping
    )
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  graph_as_dataframe(res)
}

#' Compare two community partitions
#'
#' @title Compare partitions
#' @description Compare two community partitions (adjusted Rand index,
#'   normalized mutual information, community counts, modularities). Each
#'   input is either a CommunityResult or a numeric vector of partition
#'   labels. Raw label vectors (and mixed CommunityResult/vector forms) are
#'   compared positionally and must have equal length.
#'
#' @param c1 A CommunityResult object or a numeric vector of partition labels.
#' @param c2 A CommunityResult object or a numeric vector of partition labels.
#' @param match The node matching keys when both inputs are CommunityResult
#'   objects: the single string "keys" (the default; matches on every column
#'   present in both node tables) or a non-empty character vector of column
#'   names to match on. A column literally named "keys" cannot be selected
#'   explicitly. Must be "keys" when either input is a label vector.
#'
#' @return A one-row data.frame with columns `n_matched`, `n_1`, `n_2`, `ari`,
#'   `nmi`, `n_communities_1`, `n_communities_2`, `modularity_1` and
#'   `modularity_2`.
#' @export
#'
#' @examples
#' \dontrun{
#' compare_partitions(cr1, cr2)
#' compare_partitions(c(1, 1, 2, 2), c(1, 2, 2, 2))
#' }
compare_partitions <- function(c1, c2, match = "keys") {
  c1_is_result <- inherits(c1, "CommunityResult")
  c2_is_result <- inherits(c2, "CommunityResult")
  if (!c1_is_result) {
    c1 <- graph_check_partition(c1, "c1")
  }
  if (!c2_is_result) {
    c2 <- graph_check_partition(c2, "c2")
  }
  match <- graph_check_match(match)

  positional <- !c1_is_result || !c2_is_result
  if (positional && !identical(match, "keys")) {
    stop("Argument 'match' is only supported for CommunityResult inputs.", call. = FALSE)
  }

  ensure_graphs_loaded()

  call_args <- list(
    "Juliora.compare_partitions",
    if (c1_is_result) unwrap_julia_object(c1) else c1,
    if (c2_is_result) unwrap_julia_object(c2) else c2
  )
  if (!positional) {
    call_args$match <- if (identical(match, "keys")) graph_symbol(match) else graph_symbols(match)
  }

  res <- tryCatch({
    do.call(JuliaConnectoR::juliaCall, call_args, quote = TRUE)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })

  graph_as_dataframe(res)
}

# --- S3 Constructors and Methods ---

#' Create an MRIOGraph R object
#'
#' @param proxy A JuliaProxy to wrap.
#' @return An MRIOGraph S3 object.
new_mrio_graph <- function(proxy) {
  nodes <- graph_as_dataframe(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":nodes")))

  structure(
    list(
      nodes = nodes,
      proxy = proxy
    ),
    class = "MRIOGraph"
  )
}

#' Print an MRIOGraph object
#'
#' @title Print MRIOGraph
#' @description Print a short summary of an MRIOGraph object and the first
#'   rows of its node table.
#'
#' @param x An MRIOGraph object.
#' @param ... Unused.
#'
#' @return The object, invisibly.
#' @export
#'
#' @examples
#' \dontrun{
#' print(g)
#' }
print.MRIOGraph <- function(x, ...) {
  cat("MRIOGraph (", nrow(x$nodes), " nodes)\n", sep = "")
  cat("\nNodes (first 6 rows):\n")
  print(head(x$nodes))
  invisible(x)
}

#' Create a CommunityResult R object
#'
#' @param proxy A JuliaProxy to wrap.
#' @return A CommunityResult S3 object.
new_community_result <- function(proxy) {
  membership <- JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":membership"))
  modularity <- JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":modularity"))
  algorithm <- JuliaConnectoR::juliaCall(
    "string",
    JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":algorithm"))
  )

  structure(
    list(
      membership = as.integer(membership),
      modularity = as.numeric(modularity),
      algorithm = algorithm,
      proxy = proxy
    ),
    class = "CommunityResult"
  )
}

#' Print a CommunityResult object
#'
#' @title Print CommunityResult
#' @description Print the community count, modularity and algorithm of a
#'   CommunityResult object and the first elements of its membership vector.
#'
#' @param x A CommunityResult object.
#' @param ... Unused.
#'
#' @return The object, invisibly.
#' @export
#'
#' @examples
#' \dontrun{
#' print(cr)
#' }
print.CommunityResult <- function(x, ...) {
  cat(
    "CommunityResult (", length(unique(x$membership)), " communities, algorithm ",
    x$algorithm, ")\n",
    sep = ""
  )
  cat("\nModularity: ", x$modularity, "\n", sep = "")
  cat("\nMembership (first 6 elements):\n")
  print(head(x$membership))
  invisible(x)
}
