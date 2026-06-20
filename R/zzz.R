.juliora_env <- new.env(parent = emptyenv())

#' Find the directory containing Project.toml
#'
#' @return A character string representing the directory path.
find_julia_project <- function() {
  # 1. Check option
  opt <- getOption("juliora.julia_project")
  if (!is.null(opt) && file.exists(file.path(opt, "Project.toml"))) {
    return(opt)
  }
  
  # 2. Check environment variable
  env_val <- Sys.getenv("JULIORA_JULIA_PROJECT")
  if (nzchar(env_val) && file.exists(file.path(env_val, "Project.toml"))) {
    return(env_val)
  }
  
  # 3. Check system.file (dev mode)
  pkg_dir <- system.file(package = "Juliora")
  if (nzchar(pkg_dir) && file.exists(file.path(pkg_dir, "Project.toml"))) {
    return(pkg_dir)
  }
  
  # 4. Check working directory and parents
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (file.exists(file.path(dir, "Project.toml"))) {
      return(dir)
    }
    dir <- dirname(dir)
  }
  
  # 5. Default fallback
  return(".")
}

#' Get or initialize the Julia connection and load the Juliora package
#'
#' @return Invisible NULL.
get_julia_connection <- function() {
  if (is.null(.juliora_env$juliora)) {
    if (!JuliaConnectoR::juliaSetupOk()) {
      stop("Julia environment is not available or not properly configured.", call. = FALSE)
    }
    
    proj_dir <- find_julia_project()
    proj_dir <- normalizePath(proj_dir, winslash = "/", mustWork = FALSE)
    
    # Activate Julia environment and load Juliora
    JuliaConnectoR::juliaEval(sprintf('using Pkg; Pkg.activate("%s")', proj_dir))
    JuliaConnectoR::juliaEval("using Juliora")
    JuliaConnectoR::juliaEval("using Statistics")
    
    .juliora_env$juliora <- TRUE
  }
  invisible(NULL)
}

.onLoad <- function(libname, pkgname) {
  # Establish the connection dynamically on package load
  tryCatch({
    get_julia_connection()
  }, error = function(e) {
    packageStartupMessage("Warning: Could not initialize Julia connection on package load. ", e$message)
  })
}

# --- Type Conversion Helpers ---

#' Wrap Julia proxies in R S3 classes
#'
#' @param proxy A JuliaProxy object.
#' @return A wrapped object or the proxy itself.
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
  
  if (grepl("GroupedMatrixEntry", jl_type)) {
    return(structure(list(proxy = proxy), class = "GroupedMatrixEntry"))
  } else if (grepl("GroupedSeriesEntry", jl_type)) {
    return(structure(list(proxy = proxy), class = "GroupedSeriesEntry"))
  } else if (grepl("MatrixEntry", jl_type)) {
    return(new_matrix_entry(proxy))
  } else if (grepl("SeriesEntry", jl_type)) {
    return(new_series_entry(proxy))
  } else if (grepl("EnvironmentalExtension", jl_type)) {
    return(new_environmental_extension(proxy))
  } else if (grepl("LeontiefFactorization", jl_type)) {
    return(new_leontief_factorization(proxy))
  } else if (grepl("MRIO", jl_type)) {
    return(new_mrio(proxy))
  }
  
  return(proxy)
}

#' Unwrap wrapped R S3 classes back to Julia proxies
#'
#' @param x An object.
#' @return The underlying JuliaProxy or the object itself.
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
  } else if (inherits(x, "GroupedMatrixEntry")) {
    return(x$proxy)
  }
  return(x)
}

#' Convert R named list to Julia NamedTuple proxy
#'
#' @param x A named list.
#' @return A Julia proxy object representing a NamedTuple.
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
to_named_tuple_vector <- function(x) {
  if (!is.list(x)) {
    stop("Array of NamedTuples must be represented as a list of named lists.", call. = FALSE)
  }
  if (!is.null(names(x))) {
    return(to_named_tuple(x))
  }
  
  keys_list <- lapply(x, names)
  vals_list <- lapply(x, function(item) as.list(unname(item)))
  
  JuliaConnectoR::juliaCall("Juliora.make_named_tuple_vector", keys_list, vals_list)
}

# --- S3 Constructors and Methods ---

#' Create a MatrixEntry R object
#'
#' @param proxy A JuliaProxy to wrap.
#' @return A MatrixEntry S3 object.
new_matrix_entry <- function(proxy) {
  col_indices <- as.data.frame(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":col_indices")))
  row_indices <- as.data.frame(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":row_indices")))
  
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
  data_proxy <- JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data"))
  sub_data <- JuliaConnectoR::juliaCall("Base.getindex", data_proxy, 1:n_r, 1:n_c)
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
new_series_entry <- function(proxy) {
  col_indices <- as.data.frame(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":col_indices")))
  
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
  data_proxy <- JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data"))
  sub_data <- JuliaConnectoR::juliaCall("Base.getindex", data_proxy, 1:n_e)
  print(sub_data)
  invisible(x)
}

#' Create an EnvironmentalExtension R object
#'
#' @param proxy A JuliaProxy to wrap.
#' @return An EnvironmentalExtension S3 object.
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
new_leontief_factorization <- function(proxy) {
  col_indices <- as.data.frame(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":col_indices")))
  row_indices <- as.data.frame(JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(":row_indices")))
  
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
new_mrio <- function(proxy) {
  structure(
    list(),
    julia_proxy = proxy,
    class = "MRIO"
  )
}

#' @export
`$.MRIO` <- function(x, name) {
  proxy <- attr(x, "julia_proxy")
  # Map name to Julia property
  jl_name <- name
  if (name == "Z") {
    jl_name <- "Z"
  } else if (name == "Y") {
    jl_name <- "Y"
  }
  
  prop_proxy <- JuliaConnectoR::juliaCall("Base.getproperty", proxy, JuliaConnectoR::juliaEval(paste0(":", jl_name)))
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
    return(x$data)
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

