.juliora_env <- new.env(parent = emptyenv())

#' Find the directory containing Project.toml
#'
#' Looks for a Julia project in the following order: the
#' `juliora.julia_project` option, the `JULIORA_JULIA_PROJECT` environment
#' variable, the installed package directory, the current working directory
#' and its parents (including the filesystem root), and finally `"."` as a
#' fallback.
#'
#' @return A character string representing the directory path.
#' @noRd
#' @keywords internal
find_julia_project <- function() {
  # 1. Check option
  opt <- getOption("juliora.julia_project")
  if (!is.null(opt) && length(opt) == 1 && nzchar(opt)) {
    if (file.exists(file.path(opt, "Project.toml"))) {
      return(opt)
    }
    warning("Option 'juliora.julia_project' points to '", opt,
      "' which does not contain a Project.toml; ignoring it.", call. = FALSE)
  }

  # 2. Check environment variable
  env_val <- Sys.getenv("JULIORA_JULIA_PROJECT")
  if (nzchar(env_val)) {
    if (file.exists(file.path(env_val, "Project.toml"))) {
      return(env_val)
    }
    warning("Environment variable 'JULIORA_JULIA_PROJECT' points to '", env_val,
      "' which does not contain a Project.toml; ignoring it.", call. = FALSE)
  }

  # 3. Check system.file (dev mode)
  pkg_dir <- system.file(package = "Juliora")
  if (nzchar(pkg_dir) && file.exists(file.path(pkg_dir, "Project.toml"))) {
    return(pkg_dir)
  }

  # 4. Check working directory and parents, including the filesystem root
  dir <- getwd()
  repeat {
    if (file.exists(file.path(dir, "Project.toml"))) {
      return(dir)
    }
    parent <- dirname(dir)
    if (identical(parent, dir)) {
      break
    }
    dir <- parent
  }

  # 5. Default fallback
  return(".")
}

#' Get or initialize the Julia connection and load the Juliora package
#'
#' @return Invisible NULL.
#' @noRd
#' @keywords internal
get_julia_connection <- function() {
  if (is.null(.juliora_env$juliora)) {
    if (!JuliaConnectoR::juliaSetupOk()) {
      stop(
        "Julia environment is not available or not properly configured. ",
        "Ensure Julia is installed and reachable, and that the Julia project ",
        "containing Juliora can be found. Set ",
        "options(juliora.julia_project = <path>) or the JULIORA_JULIA_PROJECT ",
        "environment variable to the directory containing Project.toml.",
        call. = FALSE
      )
    }

    proj_dir <- find_julia_project()
    proj_dir <- normalizePath(proj_dir, winslash = "/", mustWork = FALSE)

    # Activate Julia environment and load Juliora
    tryCatch({
      JuliaConnectoR::juliaEval("using Pkg")
      JuliaConnectoR::juliaCall("Pkg.activate", proj_dir)
      JuliaConnectoR::juliaEval("using Juliora")
      JuliaConnectoR::juliaEval("using Statistics")
    }, error = function(e) {
      stop(
        "Failed to load Juliora in Julia: ", conditionMessage(e), " ",
        "Ensure the Julia project containing Juliora can be found. Set ",
        "options(juliora.julia_project = <path>) or the JULIORA_JULIA_PROJECT ",
        "environment variable to the directory containing Project.toml.",
        call. = FALSE
      )
    })

    .juliora_env$juliora <- TRUE
  }
  invisible(NULL)
}

#' Reset the cached Julia connection state
#'
#' @title Reset Juliora Julia connection
#' @description Clears the cached Julia initialization flag so the next
#'   Juliora call re-runs project discovery and reloads the Julia packages.
#'   Optionally shuts down the Julia server process as well.
#'
#' @param julia_stop A logical value indicating whether to also stop the Julia
#'   server via `JuliaConnectoR::stopJulia()` (default: FALSE).
#'
#' @return Invisible NULL.
#' @export
#'
#' @examples
#' \dontrun{
#' juliora_reset()
#' juliora_reset(julia_stop = TRUE)
#' }
juliora_reset <- function(julia_stop = FALSE) {
  .juliora_env$juliora <- NULL
  if (isTRUE(julia_stop)) {
    tryCatch({
      JuliaConnectoR::stopJulia()
    }, error = function(e) {
      warning("Failed to stop the Julia server: ", conditionMessage(e), call. = FALSE)
    })
  }
  invisible(NULL)
}

.onLoad <- function(libname, pkgname) {
  if (identical(Sys.getenv("JULIA_NUM_THREADS", unset = ""), "")) {
    Sys.setenv(JULIA_NUM_THREADS = "auto")
  }
}

# --- Type Conversion Helpers ---

#' Wrap Julia proxies in R S3 classes
#'
#' Unknown Julia types pass through unchanged deliberately: only the known
#' Juliora container types are wrapped, everything else is returned as-is.
#'
#' @param proxy A JuliaProxy object.
#' @return A wrapped object or the proxy itself.
#' @noRd
#' @keywords internal
wrap_julia_object <- function(proxy) {
  if (!inherits(proxy, "JuliaProxy")) {
    return(proxy)
  }

  jl_type <- tryCatch({
    JuliaConnectoR::juliaCall("typeof", proxy)
  }, error = function(e) {
    NULL
  })
  if (is.null(jl_type)) {
    return(proxy)
  }

  # Normalize the type name, then match exactly so that e.g.
  # "GroupedMatrixEntry" cannot collide with "MatrixEntry" via substring
  # matching. Type parameters are stripped FIRST (they may themselves
  # contain module-qualified dots), then any leading module prefix.
  type_name <- tryCatch({
    s <- paste(as.character(jl_type), collapse = "")
    s <- sub("\\{.*$", "", s)
    s <- sub("^.*\\.", "", s)
    trimws(s)
  }, error = function(e) {
    NULL
  })
  if (is.null(type_name) || length(type_name) != 1 || !nzchar(type_name)) {
    return(proxy)
  }

  if (identical(type_name, "GroupedMatrixEntry")) {
    return(structure(list(proxy = proxy), class = "GroupedMatrixEntry"))
  } else if (identical(type_name, "GroupedSeriesEntry")) {
    return(structure(list(proxy = proxy), class = "GroupedSeriesEntry"))
  } else if (identical(type_name, "MatrixEntry")) {
    return(new_matrix_entry(proxy))
  } else if (identical(type_name, "SeriesEntry")) {
    return(new_series_entry(proxy))
  } else if (identical(type_name, "EnvironmentalExtension")) {
    return(new_environmental_extension(proxy))
  } else if (identical(type_name, "LeontiefFactorization")) {
    return(new_leontief_factorization(proxy))
  } else if (grepl("MRIOGraph", jl_type)) {
    return(new_mrio_graph(proxy))
  } else if (grepl("CommunityResult", jl_type)) {
    return(new_community_result(proxy))
  } else if (grepl("MRIO", jl_type)) {
    return(new_mrio(proxy))
  }

  # Unknown Julia types pass through unchanged deliberately.
  return(proxy)
}

#' Unwrap wrapped R S3 classes back to Julia proxies
#'
#' @param x An object.
#' @return The underlying JuliaProxy or the object itself.
#' @noRd
#' @keywords internal
unwrap_julia_object <- function(x) {
  if (inherits(x, "MatrixEntry")) {
    return(x$proxy)
  } else if (inherits(x, "SeriesEntry")) {
    return(x$proxy)
  } else if (inherits(x, "EnvironmentalExtension")) {
    return(x$proxy)
  } else if (inherits(x, "LeontiefFactorization")) {
    return(x$proxy)
  } else if (inherits(x, "MRIO")) {
    return(attr(x, "julia_proxy"))
  } else if (inherits(x, "MRIOGraph")) {
    return(x$proxy)
  } else if (inherits(x, "CommunityResult")) {
    return(x$proxy)
  } else if (inherits(x, "GroupedMatrixEntry")) {
    return(x$proxy)
  } else if (inherits(x, "GroupedSeriesEntry")) {
    return(x$proxy)
  }
  return(x)
}

#' Convert R named list to Julia NamedTuple proxy
#'
#' @param x A named list.
#' @return A Julia proxy object representing a NamedTuple.
#' @noRd
#' @keywords internal
to_named_tuple <- function(x) {
  if (!is.list(x) || is.null(names(x))) {
    stop("NamedTuple representation in R must be a named list.", call. = FALSE)
  }
  keys <- names(x)
  vals <- unname(x)
  JuliaConnectoR::juliaCall("Juliora.make_named_tuple", keys, as.list(vals))
}

#' Convert R list of named lists to Julia Vector\{NamedTuple\} proxy
#'
#' @param x A list of named lists.
#' @return A Julia proxy object representing a Vector of NamedTuples.
#' @noRd
#' @keywords internal
to_named_tuple_vector <- function(x) {
  if (!is.list(x)) {
    stop("Array of NamedTuples must be represented as a list of named lists.", call. = FALSE)
  }
  if (!is.null(names(x))) {
    warning("Input is a named list; treating it as a single NamedTuple.", call. = FALSE)
    return(to_named_tuple(x))
  }
  
  keys_list <- lapply(x, names)
  vals_list <- lapply(x, function(item) as.list(unname(item)))
  
  JuliaConnectoR::juliaCall("Juliora.make_named_tuple_vector", keys_list, vals_list)
}

# Map common R aggregation functions to Juliora aggregation-name strings, so
# that e.g. `aggregate(gm, sum)` and `groupby_matrix(m, :C, agg_func = mean)`
# use Julia's fast, dimension-aware reductions (Julia calls a raw R callback as
# `func(block; dims)` and expects a dim-reduced *matrix* back, which base R
# `sum`/`mean` do not provide). Returns NULL for unrecognized functions; the
# caller then falls back to passing the closure as a JuliaConnectoR callback,
# which must accept `(matrix, dims)` and return a matrix reduced along `dims`.
#' @noRd
#' @keywords internal
.julia_agg_name <- function(func) {
  if (!is.function(func)) {
    return(NULL)
  }
  b <- baseenv()
  s <- asNamespace("stats")
  if (identical(func, get("sum", b))) return("sum")
  if (identical(func, get("mean", b))) return("mean")
  if (identical(func, get("min", b))) return("min")
  if (identical(func, get("max", b))) return("max")
  if (identical(func, get("median", s))) return("median")
  if (identical(func, get("var", s))) return("var")
  if (identical(func, get("sd", s))) return("std")
  NULL
}

# JuliaConnectoR's `as.data.frame` for a Julia DataFrame sanitizes non-syntactic
# column names (e.g. "VA share (%)" -> "VA.share....", via R's default
# check.names = TRUE). Fetch the original names from Julia (strings round-trip
# unsanitized) and restore them so index data.frames keep the exact labels used
# on the Julia side. Falls back to the sanitized frame if names cannot be read.
#' @noRd
#' @keywords internal
.julia_df_to_dataframe <- function(df_proxy) {
  df <- as.data.frame(df_proxy)
  nm <- tryCatch(
    as.character(JuliaConnectoR::juliaCall("names", df_proxy)),
    error = function(e) NULL
  )
  if (!is.null(nm) && length(nm) == ncol(df)) {
    names(df) <- nm
  }
  df
}

# --- S3 Constructors and Methods ---

#' Create a MatrixEntry R object
#'
#' @param proxy A JuliaProxy to wrap.
#' @return A MatrixEntry S3 object.
#' @noRd
#' @keywords internal
new_matrix_entry <- function(proxy) {
  col_indices <- .julia_df_to_dataframe(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":col_indices")))
  row_indices <- .julia_df_to_dataframe(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":row_indices")))
  
  structure(
    list(
      col_indices = col_indices,
      row_indices = row_indices,
      proxy = proxy
    ),
    class = "MatrixEntry"
  )
}

#' @export
print.MatrixEntry <- function(x, ...) {
  cat("MatrixEntry (", nrow(x$row_indices), "x", nrow(x$col_indices), ")\n", sep = "")
  cat("\nRow Indices (first 6 rows):\n")
  print(head(x$row_indices))
  cat("\nColumn Indices (first 6 rows):\n")
  print(head(x$col_indices))
  cat("\nData Matrix (first 6 rows):\n")
  n_r <- min(6, nrow(x$row_indices))
  n_c <- min(6, nrow(x$col_indices))
  if (n_r == 0 || n_c == 0) {
    cat("(empty data matrix)\n")
    return(invisible(x))
  }
  data_proxy <- JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data"))
  sub_data <- JuliaConnectoR::juliaCall("Base.getindex", data_proxy, seq_len(n_r), seq_len(n_c))
  # Convert to matrix for print formatting
  if (is.vector(sub_data)) {
    sub_data <- matrix(sub_data, nrow = n_r, ncol = n_c)
  }
  print(sub_data)
  invisible(x)
}

#' Create a SeriesEntry R object
#'
#' @param proxy A JuliaProxy to wrap.
#' @return A SeriesEntry S3 object.
#' @noRd
#' @keywords internal
new_series_entry <- function(proxy) {
  col_indices <- .julia_df_to_dataframe(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":col_indices")))
  
  structure(
    list(
      col_indices = col_indices,
      proxy = proxy
    ),
    class = "SeriesEntry"
  )
}

#' @export
print.SeriesEntry <- function(x, ...) {
  cat("SeriesEntry (length ", nrow(x$col_indices), ")\n", sep = "")
  cat("\nCol Indices (first 6 rows):\n")
  print(head(x$col_indices))
  cat("\nData Vector (first 6 elements):\n")
  n_e <- min(6, nrow(x$col_indices))
  if (n_e == 0) {
    cat("(empty data vector)\n")
    return(invisible(x))
  }
  data_proxy <- JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data"))
  sub_data <- JuliaConnectoR::juliaCall("Base.getindex", data_proxy, seq_len(n_e))
  print(sub_data)
  invisible(x)
}

#' Create an EnvironmentalExtension R object
#'
#' @param proxy A JuliaProxy to wrap.
#' @return An EnvironmentalExtension S3 object.
#' @noRd
#' @keywords internal
new_environmental_extension <- function(proxy) {
  f_proxy <- JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":F"))
  a_proxy <- JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":A"))
  
  structure(
    list(
      F = new_matrix_entry(f_proxy),
      A = new_matrix_entry(a_proxy),
      proxy = proxy
    ),
    class = "EnvironmentalExtension"
  )
}

#' @export
print.EnvironmentalExtension <- function(x, ...) {
  cat("EnvironmentalExtension wrapper\n")
  cat("Fields: F (direct impacts), A (intensities)\n")
  invisible(x)
}

#' Create a LeontiefFactorization R object
#'
#' @param proxy A JuliaProxy to wrap.
#' @return A LeontiefFactorization S3 object.
#' @noRd
#' @keywords internal
new_leontief_factorization <- function(proxy) {
  col_indices <- .julia_df_to_dataframe(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":col_indices")))
  row_indices <- .julia_df_to_dataframe(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":row_indices")))
  
  structure(
    list(
      col_indices = col_indices,
      row_indices = row_indices,
      proxy = proxy
    ),
    class = "LeontiefFactorization"
  )
}

#' @export
print.LeontiefFactorization <- function(x, ...) {
  cat("LeontiefFactorization wrapper\n")
  invisible(x)
}

#' Dimensions of LeontiefFactorization
#'
#' @param x A LeontiefFactorization object.
#' @return An integer vector of length 2.
#' @export
dim.LeontiefFactorization <- function(x) {
  c(nrow(x$row_indices), nrow(x$col_indices))
}

#' Create an MRIO R object
#'
#' @param proxy A JuliaProxy to wrap.
#' @return An MRIO S3 object.
#' @noRd
#' @keywords internal
new_mrio <- function(proxy) {
  structure(
    list(),
    julia_proxy = proxy,
    class = "MRIO"
  )
}

#' @export
`$.MRIO` <- function(x, name) {
  valid_fields <- c("A", "T", "Z", "VA", "FD", "Y", "L", "X", "env")
  if (!name %in% valid_fields) {
    stop("MRIO has no field '", name, "'. Valid fields: ",
      paste(valid_fields, collapse = ", "), ".", call. = FALSE)
  }
  proxy <- attr(x, "julia_proxy")

  # `name` is validated against the fixed whitelist above, so interpolating
  # it into a constant Julia Symbol expression is safe.
  prop_proxy <- JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(paste0(":", name)))
  # Fields that are `nothing` in Julia (e.g. `$env` or `$L` on MRIO objects
  # built without them) arrive as NULL and pass through unchanged.
  wrap_julia_object(prop_proxy)
}

#' @export
`[[.MRIO` <- function(x, i, ...) {
  if (is.character(i)) {
    return(`$.MRIO`(x, i))
  }
  NextMethod()
}

#' Get names of MRIO fields
#'
#' @param x An MRIO object.
#' @return A character vector of field names.
#' @export
names.MRIO <- function(x) {
  c("A", "T", "Z", "VA", "FD", "Y", "L", "X", "env")
}

#' Autocomplete names for MRIO
#'
#' @param x An MRIO object.
#' @param pattern A character string to match.
#' @return A character vector of matching field names.
#' @export
.DollarNames.MRIO <- function(x, pattern = "") {
  fields <- c("A", "T", "Z", "VA", "FD", "Y", "L", "X", "env")
  grep(pattern, fields, value = TRUE)
}

#' @export
print.MRIO <- function(x, ...) {
  proxy <- attr(x, "julia_proxy")
  jl_type <- tryCatch({
    JuliaConnectoR::juliaCall("typeof", proxy)
  }, error = function(e) {
    "MRIO"
  })
  cat("MRIO database wrapper (Julia object of type ", jl_type, ")\n", sep = "")
  cat("Fields: A, T, Z, VA, FD, Y, L, X, env\n")
  invisible(x)
}

#' Check if an object is a matrix entry type
#'
#' @param x An object to check.
#' @return A logical value.
#' @noRd
is_matrix_entry <- function(x) {
  inherits(x, "MatrixEntry") || inherits(x, "LeontiefFactorization") || inherits(x, "GroupedMatrixEntry")
}

#' Subset MatrixEntry
#'
#' @param x A MatrixEntry object.
#' @param i Rows to select (logical vector, named list, list of named lists, or missing).
#' @param j Columns to select (logical vector, named list, list of named lists, or missing).
#' @param ... Unused.
#' @return A single value, a SeriesEntry, or a MatrixEntry.
#' @export
`[.MatrixEntry` <- function(x, i, j, ...) {
  i_missing <- missing(i)
  j_missing <- missing(j)
  
  get_julia_connection()
  x_jl <- unwrap_julia_object(x)
  
  convert_index_arg <- function(arg, is_missing) {
    if (is_missing) {
      return(JuliaConnectoR::juliaEval(":"))
    }
    if (is.logical(arg)) {
      return(arg)
    }
    if (is.list(arg)) {
      if (!is.null(names(arg))) {
        return(to_named_tuple(arg))
      } else {
        return(to_named_tuple_vector(arg))
      }
    }
    stop("Subsetting index must be logical or a list representation of NamedTuple(s).", call. = FALSE)
  }
  
  i_jl <- convert_index_arg(i, i_missing)
  j_jl <- convert_index_arg(j, j_missing)
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Base.getindex", x_jl, i_jl, j_jl)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Subset LeontiefFactorization
#'
#' @param x A LeontiefFactorization object.
#' @param i Rows to select (logical vector, named list, list of named lists, or missing).
#' @param j Columns to select (logical vector, named list, list of named lists, or missing).
#' @param ... Unused.
#' @return A single value, a SeriesEntry, or a MatrixEntry.
#' @export
`[.LeontiefFactorization` <- `[.MatrixEntry`

#' Dimensions of MatrixEntry
#'
#' @param x A MatrixEntry object.
#' @return An integer vector of length 2.
#' @export
dim.MatrixEntry <- function(x) {
  c(nrow(x$row_indices), nrow(x$col_indices))
}

#' @export
`$.MatrixEntry` <- function(x, name) {
  if (name == "data") {
    return(JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data")))
  }
  x[[name]]
}

#' @export
`[[.MatrixEntry` <- function(x, i, ...) {
  if (identical(i, "data")) {
    return(JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data")))
  }
  NextMethod()
}

#' Get names of MatrixEntry fields
#'
#' @param x A MatrixEntry object.
#' @return A character vector of field names.
#' @export
names.MatrixEntry <- function(x) {
  c("data", "col_indices", "row_indices", "proxy")
}

#' Autocomplete names for MatrixEntry
#'
#' @param x A MatrixEntry object.
#' @param pattern A character string to match.
#' @return A character vector of matching field names.
#' @export
.DollarNames.MatrixEntry <- function(x, pattern = "") {
  fields <- c("data", "col_indices", "row_indices")
  grep(pattern, fields, value = TRUE)
}

#' Subset SeriesEntry
#'
#' @param x A SeriesEntry object.
#' @param i Column key (named list).
#' @param ... Unused.
#' @return A numeric value.
#' @export
`[.SeriesEntry` <- function(x, i, ...) {
  if (missing(i)) {
    return(JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data")))
  }
  
  get_julia_connection()
  x_jl <- unwrap_julia_object(x)
  
  if (is.logical(i) || is.numeric(i)) {
    res <- tryCatch({
      JuliaConnectoR::juliaCall("Base.getindex", x_jl, i)
    }, error = function(e) {
      stop("Julia Error: ", e$message, call. = FALSE)
    })
    return(wrap_julia_object(res))
  }
  
  if (!is.list(i) || is.null(names(i))) {
    stop("Subsetting index for SeriesEntry must be a logical vector, numeric indices, or a named list.", call. = FALSE)
  }
  
  i_jl <- to_named_tuple(i)
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Base.getindex", x_jl, i_jl)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  res
}

#' Length of SeriesEntry
#'
#' @param x A SeriesEntry object.
#' @return An integer.
#' @export
length.SeriesEntry <- function(x) {
  nrow(x$col_indices)
}

#' @export
`$.SeriesEntry` <- function(x, name) {
  if (name == "data") {
    return(JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data")))
  }
  x[[name]]
}

#' @export
`[[.SeriesEntry` <- function(x, i, ...) {
  if (identical(i, "data")) {
    return(JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data")))
  }
  NextMethod()
}

#' Get names of SeriesEntry fields
#'
#' @param x A SeriesEntry object.
#' @return A character vector of field names.
#' @export
names.SeriesEntry <- function(x) {
  c("data", "col_indices", "proxy")
}

#' Autocomplete names for SeriesEntry
#'
#' @param x A SeriesEntry object.
#' @param pattern A character string to match.
#' @return A character vector of matching field names.
#' @export
.DollarNames.SeriesEntry <- function(x, pattern = "") {
  fields <- c("data", "col_indices")
  grep(pattern, fields, value = TRUE)
}
