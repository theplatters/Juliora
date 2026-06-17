#' Check if Julia environment is available
#'
#' @title Check Julia availability
#' @description Verifies if a working Julia installation can be found by JuliaConnectoR.
#'
#' @return A logical value indicating whether Julia is available.
#' @export
#'
#' @examples
#' \dontrun{
#' is_julia_available()
#' }
is_julia_available <- function() {
  JuliaConnectoR::juliaSetupOk()
}

#' Load and construct complete Eora MRIO database
#'
#' @title Load Eora database
#' @description Load and construct complete Eora MRIO database from a file directory.
#'
#' @param path A character string representing the directory path containing Eora database files.
#'
#' @return An MRIO object wrapping the Julia MRIO database.
#' @export
#'
#' @examples
#' \dontrun{
#' eora_db <- Eora("data/2017/")
#' }
Eora <- function(path) {
  if (!is.character(path) || length(path) != 1) {
    stop("Argument 'path' must be a single character string.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.Eora", path)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Load and construct complete Gloria MRIO database
#'
#' @title Load Gloria database
#' @description Load and construct complete Gloria MRIO database from a file directory.
#'
#' @param path A character string representing the directory path containing Gloria database files.
#' @param version An integer specifying the Gloria version (e.g. 60).
#' @param year An integer specifying the database year.
#'
#' @return An MRIO object wrapping the Julia MRIO database.
#' @export
#'
#' @examples
#' \dontrun{
#' gloria_db <- Gloria("data/GLORIA/2019/", 60, 2019)
#' }
Gloria <- function(path, version, year) {
  if (!is.character(path) || length(path) != 1) {
    stop("Argument 'path' must be a single character string.", call. = FALSE)
  }
  if (!is.numeric(version) || length(version) != 1) {
    stop("Argument 'version' must be an integer.", call. = FALSE)
  }
  if (!is.numeric(year) || length(year) != 1) {
    stop("Argument 'year' must be an integer.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.Gloria", path, as.integer(version), as.integer(year))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Create a new MRIO object from components
#'
#' @title Construct MRIO object
#' @description Create a complete multi-region input-output (MRIO) database structure from transactions, final demand, and value added.
#'
#' @param Z A MatrixEntry representing intermediate transactions.
#' @param Y A MatrixEntry representing final demand.
#' @param VA A MatrixEntry representing value added.
#'
#' @return An MRIO object wrapping the Julia MRIO database.
#' @export
#'
#' @examples
#' \dontrun{
#' mrio_db <- MRIO(Z_matrix, Y_matrix, VA_matrix)
#' }
MRIO <- function(Z, Y, VA) {
  if (!inherits(Z, "MatrixEntry")) {
    stop("Argument 'Z' must be a MatrixEntry object.", call. = FALSE)
  }
  if (!inherits(Y, "MatrixEntry")) {
    stop("Argument 'Y' must be a MatrixEntry object.", call. = FALSE)
  }
  if (!inherits(VA, "MatrixEntry")) {
    stop("Argument 'VA' must be a MatrixEntry object.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.MRIO", Z = unwrap_julia_object(Z), Y = unwrap_julia_object(Y), VA = unwrap_julia_object(VA))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Parse Gloria database files
#'
#' @title Parse Gloria database
#' @description Parse Gloria database files for a given year.
#'
#' @param path A character string representing the directory path containing Gloria database files.
#' @param year An integer specifying the database year.
#' @param version An integer specifying the Gloria version (default: 60).
#'
#' @return An MRIO object wrapping the Julia MRIO database.
#' @export
#'
#' @examples
#' \dontrun{
#' db <- parse_gloria("data/GLORIA/", 2019, version = 60)
#' }
parse_gloria <- function(path, year, version = 60) {
  if (!is.character(path) || length(path) != 1) {
    stop("Argument 'path' must be a single character string.", call. = FALSE)
  }
  if (!is.numeric(year) || length(year) != 1) {
    stop("Argument 'year' must be an integer.", call. = FALSE)
  }
  if (!is.numeric(version) || length(version) != 1) {
    stop("Argument 'version' must be an integer.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.parse_gloria", path, as.integer(year), version = as.integer(version))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Parse Gloria Supply-Use Tables (SUT) files
#'
#' @title Parse Gloria SUT
#' @description Parse Gloria Supply-Use Tables (SUT) files for a given year.
#'
#' @param path A character string representing the directory path containing Gloria database files.
#' @param year An integer specifying the database year.
#' @param version An integer specifying the Gloria version (default: 60).
#'
#' @return An MRIO object wrapping the Julia MRIO database.
#' @export
#'
#' @examples
#' \dontrun{
#' sut_db <- parse_gloria_sut("data/GLORIA/", 2019, version = 60)
#' }
parse_gloria_sut <- function(path, year, version = 60) {
  if (!is.character(path) || length(path) != 1) {
    stop("Argument 'path' must be a single character string.", call. = FALSE)
  }
  if (!is.numeric(year) || length(year) != 1) {
    stop("Argument 'year' must be an integer.", call. = FALSE)
  }
  if (!is.numeric(version) || length(version) != 1) {
    stop("Argument 'version' must be an integer.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.parse_gloria_sut", path, as.integer(year), version = as.integer(version))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Create a MatrixEntry object
#'
#' @title Construct MatrixEntry object
#' @description Create a MatrixEntry with automatic validation and lookup table generation.
#'
#' @param data A numeric matrix.
#' @param col_indices A data.frame containing column labels and metadata.
#' @param row_indices A data.frame containing row labels and metadata.
#'
#' @return A MatrixEntry object wrapping the Julia MatrixEntry struct.
#' @export
#'
#' @examples
#' \dontrun{
#' col_df <- data.frame(Country = c("USA", "CHN"), Sector = c("Agr", "Man"))
#' row_df <- data.frame(Country = c("USA", "CHN", "DEU"), Sector = c("Agr", "Man", "Ser"))
#' me <- MatrixEntry(matrix(1:6, 3, 2), col_df, row_df)
#' }
MatrixEntry <- function(data, col_indices, row_indices) {
  if (!is.matrix(data)) {
    stop("Argument 'data' must be a matrix.", call. = FALSE)
  }
  if (!is.data.frame(col_indices)) {
    stop("Argument 'col_indices' must be a data.frame.", call. = FALSE)
  }
  if (!is.data.frame(row_indices)) {
    stop("Argument 'row_indices' must be a data.frame.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.MatrixEntry", data, col_indices, row_indices)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Create a SeriesEntry object
#'
#' @title Construct SeriesEntry object
#' @description Create a SeriesEntry with automatic validation and lookup table generation.
#'
#' @param data A numeric vector.
#' @param col_indices A data.frame containing labels and metadata.
#'
#' @return A SeriesEntry object wrapping the Julia SeriesEntry struct.
#' @export
#'
#' @examples
#' \dontrun{
#' col_df <- data.frame(Country = c("USA", "CHN"), Sector = c("Agr", "Man"))
#' se <- SeriesEntry(c(100.0, 200.0), col_df)
#' }
SeriesEntry <- function(data, col_indices) {
  if (!is.numeric(data)) {
    stop("Argument 'data' must be a numeric vector.", call. = FALSE)
  }
  if (!is.data.frame(col_indices)) {
    stop("Argument 'col_indices' must be a data.frame.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.SeriesEntry", data, col_indices)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Group a MatrixEntry by columns in indices
#'
#' @title Group MatrixEntry
#' @description Group row or column indices of a MatrixEntry by specified columns.
#'
#' @param m A MatrixEntry object.
#' @param cols A character vector of column names to group by.
#' @param dims An integer specifying the dimension: 1 for rows, 2 for columns (default: 1).
#'
#' @return A GroupedMatrixEntry proxy object.
#' @export
#'
#' @examples
#' \dontrun{
#' gm <- groupby(me, c("Country"))
#' }
groupby <- function(m, cols, dims = 1) {
  if (!inherits(m, "MatrixEntry")) {
    stop("Argument 'm' must be a MatrixEntry object.", call. = FALSE)
  }
  if (!is.character(cols)) {
    stop("Argument 'cols' must be a character vector of column names.", call. = FALSE)
  }
  if (!dims %in% c(1, 2)) {
    stop("Argument 'dims' must be 1 or 2.", call. = FALSE)
  }
  
  get_julia_connection()
  
  cols_jl <- if (length(cols) == 1) {
    JuliaConnectoR::juliaEval(sprintf("Symbol(\"%s\")", cols))
  } else {
    JuliaConnectoR::juliaCall("Vector{Symbol}", as.list(cols))
  }
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.groupby", unwrap_julia_object(m), cols_jl, dims = as.integer(dims))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Aggregate a GroupedMatrixEntry using a function
#'
#' @title Aggregate GroupedMatrixEntry
#' @description Aggregate matrix data grouped by groupby.
#'
#' @param gm A GroupedMatrixEntry object.
#' @param func An R function (e.g. sum, mean) or character string naming a Julia function (e.g. "sum", "mean").
#'
#' @return A MatrixEntry object.
#' @export
#'
#' @examples
#' \dontrun{
#' aggregated <- aggregate(gm, "sum")
#' }
aggregate <- function(gm, func) {
  if (!inherits(gm, "GroupedMatrixEntry")) {
    stop("Argument 'gm' must be a GroupedMatrixEntry object.", call. = FALSE)
  }
  
  get_julia_connection()
  
  func_jl <- if (is.character(func)) {
    JuliaConnectoR::juliaEval(func)
  } else if (is.function(func)) {
    JuliaConnectoR::juliaFun(func)
  } else {
    stop("Argument 'func' must be a function or a character string naming a Julia function.", call. = FALSE)
  }
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.aggregate", unwrap_julia_object(gm), func_jl)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Filter rows of a MatrixEntry based on a condition function
#'
#' @title Filter rows of MatrixEntry
#' @description Filter rows based on a condition function applied to row indices.
#'
#' @param m A MatrixEntry object.
#' @param condition_func A function that takes a list (row of index data.frame) and returns a logical value.
#'
#' @return A MatrixEntry object.
#' @export
#'
#' @examples
#' \dontrun{
#' filtered <- filter_rows(me, function(row) row$Country == "USA")
#' }
filter_rows <- function(m, condition_func) {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  if (!is.function(condition_func)) {
    stop("Argument 'condition_func' must be a function.", call. = FALSE)
  }
  
  get_julia_connection()
  
  func_jl <- JuliaConnectoR::juliaFun(condition_func)
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.filter_rows", unwrap_julia_object(m), func_jl)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Filter columns of a MatrixEntry based on a condition function
#'
#' @title Filter columns of MatrixEntry
#' @description Filter columns based on a condition function applied to column indices.
#'
#' @param m A MatrixEntry object.
#' @param condition_func A function that takes a list (column of index data.frame) and returns a logical value.
#'
#' @return A MatrixEntry object.
#' @export
#'
#' @examples
#' \dontrun{
#' filtered <- filter_cols(me, function(col) col$Country == "CHN")
#' }
filter_cols <- function(m, condition_func) {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  if (!is.function(condition_func)) {
    stop("Argument 'condition_func' must be a function.", call. = FALSE)
  }
  
  get_julia_connection()
  
  func_jl <- JuliaConnectoR::juliaFun(condition_func)
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.filter_cols", unwrap_julia_object(m), func_jl)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Drop rows or columns from a MatrixEntry
#'
#' @title Drop rows or columns
#' @description Drop rows or columns from a MatrixEntry using a named list (NamedTuple) or list of named lists.
#'
#' @param m A MatrixEntry object.
#' @param indices A named list (representing a NamedTuple) or a list of named lists (representing a vector of NamedTuples).
#' @param dims An integer specifying the dimension: 1 for rows, 2 for columns (default: 1).
#'
#' @return A MatrixEntry object.
#' @export
#'
#' @examples
#' \dontrun{
#' dropped <- drop(me, list(Country = "USA"), dims = 1)
#' }
drop <- function(m, indices, dims = 1) {
  if (!inherits(m, "MatrixEntry")) {
    stop("Argument 'm' must be a MatrixEntry object.", call. = FALSE)
  }
  if (!dims %in% c(1, 2)) {
    stop("Argument 'dims' must be 1 or 2.", call. = FALSE)
  }
  
  get_julia_connection()
  
  indices_jl <- if (!is.null(names(indices))) {
    to_named_tuple(indices)
  } else {
    to_named_tuple_vector(indices)
  }
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.drop", unwrap_julia_object(m), indices_jl, dims = as.integer(dims))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Drop rows or columns from a MatrixEntry in place (modifies the object)
#'
#' @title Drop rows or columns in-place
#' @description Modifies a MatrixEntry object in-place by dropping rows or columns.
#'
#' @param m A MatrixEntry object.
#' @param indices A named list (representing a NamedTuple).
#' @param dims An integer specifying the dimension: 1 for rows, 2 for columns (default: 1).
#'
#' @return The modified MatrixEntry object.
#' @export
#'
#' @examples
#' \dontrun{
#' drop_mut(me, list(Country = "USA"), dims = 1)
#' }
drop_mut <- function(m, indices, dims = 1) {
  if (!inherits(m, "MatrixEntry")) {
    stop("Argument 'm' must be a MatrixEntry object.", call. = FALSE)
  }
  if (!dims %in% c(1, 2)) {
    stop("Argument 'dims' must be 1 or 2.", call. = FALSE)
  }
  
  get_julia_connection()
  
  indices_jl <- to_named_tuple(indices)
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.drop!", unwrap_julia_object(m), indices_jl, dims = as.integer(dims))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  # Update R S3 object's native fields
  m$col_indices <- as.data.frame(JuliaConnectoR::juliaCall("Base.getproperty", m$proxy, JuliaConnectoR::juliaEval(":col_indices")))
  m$row_indices <- as.data.frame(JuliaConnectoR::juliaCall("Base.getproperty", m$proxy, JuliaConnectoR::juliaEval(":row_indices")))
  
  return(m)
}

#' Subset a MatrixEntry by condition functions
#'
#' @title Filter both dimensions of MatrixEntry
#' @description Filter both rows and columns using separate condition functions.
#'
#' @param m A MatrixEntry object.
#' @param row_condition A function that takes a list (row of row_indices) and returns a logical.
#' @param col_condition A function that takes a list (column of col_indices) and returns a logical.
#'
#' @return A MatrixEntry object.
#' @export
#'
#' @examples
#' \dontrun{
#' filtered <- filter_matrix(me, function(r) r$Country == "USA", function(c) c$Sector == "Agr")
#' }
filter_matrix <- function(m, row_condition, col_condition) {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  if (!is.function(row_condition) || !is.function(col_condition)) {
    stop("Row and column conditions must be functions.", call. = FALSE)
  }
  
  get_julia_connection()
  
  row_cond_jl <- JuliaConnectoR::juliaFun(row_condition)
  col_cond_jl <- JuliaConnectoR::juliaFun(col_condition)
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.filter_matrix", unwrap_julia_object(m), row_cond_jl, col_cond_jl)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Convert a MatrixEntry to long-form DataFrame
#'
#' @title MatrixEntry to long DataFrame
#' @description Convert a MatrixEntry to long-form DataFrame suitable for R database/tidyverse operations.
#'
#' @param m A MatrixEntry object.
#' @param value_name A character string specifying the name of the value column (default: "value").
#'
#' @return A data.frame.
#' @export
#'
#' @examples
#' \dontrun{
#' df <- to_long_dataframe(me)
#' }
to_long_dataframe <- function(m, value_name = "value") {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  if (!is.character(value_name) || length(value_name) != 1) {
    stop("Argument 'value_name' must be a single character string.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.to_long_dataframe", unwrap_julia_object(m), value_name = value_name)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  as.data.frame(res)
}

#' Convert a long-form DataFrame back to MatrixEntry format
#'
#' @title Long DataFrame to MatrixEntry
#' @description Convert a long-form DataFrame back to MatrixEntry format.
#'
#' @param df A data.frame.
#' @param value_col A character string specifying the name of the value column (default: "value").
#' @param row_prefix A character string specifying the row columns prefix (default: "row_").
#' @param col_prefix A character string specifying the column columns prefix (default: "col_").
#'
#' @return A MatrixEntry object.
#' @export
#'
#' @examples
#' \dontrun{
#' me <- from_long_dataframe(df)
#' }
from_long_dataframe <- function(df, value_col = "value", row_prefix = "row_", col_prefix = "col_") {
  if (!is.data.frame(df)) {
    stop("Argument 'df' must be a data.frame.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.from_long_dataframe", df, value_col = value_col, row_prefix = row_prefix, col_prefix = col_prefix)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Group and aggregate matrix data by specified index columns
#'
#' @title Group and aggregate MatrixEntry
#' @description Group and aggregate matrix data by specified index columns.
#'
#' @param m A MatrixEntry object.
#' @param ... Column names (symbols or character strings) to group by.
#' @param agg_func An R function or Julia function name (default: "sum" or sum).
#' @param rows A logical value indicating whether to group rows (TRUE) or columns (FALSE) (default: TRUE).
#' @param value_name A character string specifying the name of the value column (default: "value").
#'
#' @return A data.frame.
#' @export
#'
#' @examples
#' \dontrun{
#' df <- groupby_matrix(me, "CountryCode", agg_func = "sum")
#' }
groupby_matrix <- function(m, ..., agg_func = "sum", rows = TRUE, value_name = "value") {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  
  grouping_cols <- list(...)
  grouping_cols <- lapply(grouping_cols, function(col) {
    if (is.name(col) || is.symbol(col)) {
      as.character(col)
    } else {
      as.character(col)
    }
  })
  
  get_julia_connection()
  
  agg_func_jl <- if (is.character(agg_func)) {
    JuliaConnectoR::juliaEval(agg_func)
  } else if (is.function(agg_func)) {
    JuliaConnectoR::juliaFun(agg_func)
  } else {
    stop("Argument 'agg_func' must be a function or a character string naming a Julia function.", call. = FALSE)
  }
  
  grouping_cols_jl <- lapply(grouping_cols, function(col) JuliaConnectoR::juliaEval(paste0(":", col)))
  
  res <- tryCatch({
    do.call(JuliaConnectoR::juliaCall, c(list("Juliora.groupby_matrix", unwrap_julia_object(m)), grouping_cols_jl, list(agg_func = agg_func_jl, rows = rows, value_name = value_name)))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  as.data.frame(res)
}

#' Sum matrix values by country codes
#'
#' @title Sum MatrixEntry by country
#' @description Sum matrix values by country codes.
#'
#' @param m A MatrixEntry object
#' @param dimension A character string indicating the dimension to sum over: "both", "rows", or "cols" (default: "both")
#'
#' @return A data.frame containing the summed values by country
#' @export
#'
#' @examples
#' \dontrun{
#' df <- sum_by_country(me, dimension = "rows")
#' }
sum_by_country <- function(m, dimension = "both") {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  if (!dimension %in% c("both", "rows", "cols")) {
    stop("Argument 'dimension' must be one of 'both', 'rows', or 'cols'.", call. = FALSE)
  }
  
  get_julia_connection()
  dim_jl <- JuliaConnectoR::juliaEval(paste0(":", dimension))
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.sum_by_country", unwrap_julia_object(m), dimension = dim_jl)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  as.data.frame(res)
}

#' Sum matrix values by sector codes
#'
#' @title Sum MatrixEntry by sector
#' @description Sum matrix values by sector codes.
#'
#' @param m A MatrixEntry object
#' @param dimension A character string indicating the dimension to sum over: "both", "rows", or "cols" (default: "both")
#'
#' @return A data.frame containing the summed values by sector
#' @export
#'
#' @examples
#' \dontrun{
#' df <- sum_by_sector(me, dimension = "cols")
#' }
sum_by_sector <- function(m, dimension = "both") {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  if (!dimension %in% c("both", "rows", "cols")) {
    stop("Argument 'dimension' must be one of 'both', 'rows', or 'cols'.", call. = FALSE)
  }
  
  get_julia_connection()
  dim_jl <- JuliaConnectoR::juliaEval(paste0(":", dimension))
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.sum_by_sector", unwrap_julia_object(m), dimension = dim_jl)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  as.data.frame(res)
}

#' Add a calculated column to row or column indices
#'
#' @title Add calculated column
#' @description Add a calculated column to row or column indices based on existing index values.
#'
#' @param m A MatrixEntry object.
#' @param col_name A character string specifying the name of the new column.
#' @param calculation_func A function that takes a list (row of index data.frame) and returns a value.
#' @param to_rows A logical value indicating whether to add the column to row indices (TRUE) or column indices (FALSE) (default: TRUE).
#'
#' @return A MatrixEntry object.
#' @export
#'
#' @examples
#' \dontrun{
#' me_new <- add_calculated_column(me, "is_developed", function(row) row$Country == "USA")
#' }
add_calculated_column <- function(m, col_name, calculation_func, to_rows = TRUE) {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  if (!is.character(col_name) || length(col_name) != 1) {
    stop("Argument 'col_name' must be a single character string.", call. = FALSE)
  }
  if (!is.function(calculation_func)) {
    stop("Argument 'calculation_func' must be a function.", call. = FALSE)
  }
  
  get_julia_connection()
  
  col_name_jl <- JuliaConnectoR::juliaEval(paste0(":", col_name))
  calc_func_jl <- JuliaConnectoR::juliaFun(calculation_func)
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.add_calculated_column", unwrap_julia_object(m), col_name_jl, calc_func_jl, to_rows = to_rows)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  wrap_julia_object(res)
}

#' Pivot the matrix data to wide format
#'
#' @title Pivot MatrixEntry to wide
#' @description Pivot the matrix data to wide format for analysis or visualization.
#'
#' @param m A MatrixEntry object.
#' @param row_vars A character vector of row variables to keep.
#' @param col_var A character string specifying the column variable to pivot.
#' @param value_var A character string specifying the value column (default: "value").
#'
#' @return A data.frame.
#' @export
#'
#' @examples
#' \dontrun{
#' df_wide <- pivot_matrix_to_wide(me, c("Country"), "Sector")
#' }
pivot_matrix_to_wide <- function(m, row_vars, col_var, value_var = "value") {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  if (!is.character(row_vars)) {
    stop("Argument 'row_vars' must be a character vector.", call. = FALSE)
  }
  if (!is.character(col_var) || length(col_var) != 1) {
    stop("Argument 'col_var' must be a single character string.", call. = FALSE)
  }
  
  get_julia_connection()
  
  row_vars_jl <- if (length(row_vars) == 1) {
    JuliaConnectoR::juliaEval(sprintf("Symbol(\"%s\")", row_vars))
  } else {
    JuliaConnectoR::juliaCall("Vector{Symbol}", as.list(row_vars))
  }
  col_var_jl <- JuliaConnectoR::juliaEval(sprintf("Symbol(\"%s\")", col_var))
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.pivot_matrix_to_wide", unwrap_julia_object(m), row_vars_jl, col_var_jl, value_var)
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  as.data.frame(res)
}

#' Generate comprehensive summary statistics for the matrix data
#'
#' @title MatrixEntry summary
#' @description Generate comprehensive summary statistics for the matrix data.
#'
#' @param m A MatrixEntry object.
#'
#' @return A data.frame.
#' @export
#'
#' @examples
#' \dontrun{
#' summary_df <- matrix_summary(me)
#' }
matrix_summary <- function(m) {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.matrix_summary", unwrap_julia_object(m))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  as.data.frame(res)
}

#' Generate country-by-country flow summary
#'
#' @title Country summary of MatrixEntry
#' @description Generate country-by-country flow summary for bilateral analysis.
#'
#' @param m A MatrixEntry object.
#'
#' @return A data.frame.
#' @export
#'
#' @examples
#' \dontrun{
#' bilateral_flows <- country_summary(me)
#' }
country_summary <- function(m) {
  if (!is_matrix_entry(m)) {
    stop("Argument 'm' must be a MatrixEntry or LeontiefFactorization object.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.country_summary", unwrap_julia_object(m))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  as.data.frame(res)
}

#' Calculate induced production using the Leontief Inverse
#'
#' @title Leontief induced production
#' @description Calculate the production induced by the final demand of specified consumer countries on specified producer countries.
#'
#' @param mrio An MRIO database object.
#' @param consumer_countries A character vector of consumer country codes.
#' @param producer_countries A character vector of producer country codes.
#'
#' @return A data.frame containing columns CountryCode, Sector, and InducedProduction.
#' @export
#'
#' @examples
#' \dontrun{
#' res <- induced_production(mrio, c("DEU", "FRA"), c("IND", "BRA"))
#' }
induced_production <- function(mrio, consumer_countries, producer_countries) {
  if (!inherits(mrio, "MRIO")) {
    stop("Argument 'mrio' must be an MRIO object.", call. = FALSE)
  }
  if (!is.character(consumer_countries)) {
    stop("Argument 'consumer_countries' must be a character vector.", call. = FALSE)
  }
  if (!is.character(producer_countries)) {
    stop("Argument 'producer_countries' must be a character vector.", call. = FALSE)
  }
  
  get_julia_connection()
  
  res <- tryCatch({
    JuliaConnectoR::juliaCall("Juliora.induced_production", unwrap_julia_object(mrio), as.character(consumer_countries), as.character(producer_countries))
  }, error = function(e) {
    stop("Julia Error: ", e$message, call. = FALSE)
  })
  
  as.data.frame(res)
}
