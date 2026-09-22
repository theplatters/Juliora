#' Convert MatrixEntry to long DataFrame
#'
#' @title Convert MatrixEntry to long DataFrame
#' @description Convert a MatrixEntry to a long-form DataFrame.
#'
#' @param x A MatrixEntry object.
#' @param row.names Unused.
#' @param optional Unused.
#' @param ... Unused.
#' @param value_name A character string specifying the name of the value column (default: "value").
#'
#' @return A data.frame.
#' @export
as.data.frame.MatrixEntry <- function(x, row.names = NULL, optional = FALSE, ..., value_name = "value") {
  to_long_dataframe(x, value_name = value_name)
}

#' Convert SeriesEntry to DataFrame
#'
#' @title Convert SeriesEntry to DataFrame
#' @description Convert a SeriesEntry to a DataFrame.
#'
#' @param x A SeriesEntry object.
#' @param row.names Unused.
#' @param optional Unused.
#' @param ... Unused.
#' @param value_name A character string specifying the name of the value column (default: "value").
#'
#' @return A data.frame.
#' @export
as.data.frame.SeriesEntry <- function(x, row.names = NULL, optional = FALSE, ..., value_name = "value") {
  df <- x$col_indices
  n_e <- nrow(df)
  data_proxy <- JuliaConnectoR::juliaCall("Base.getproperty", x$proxy, JuliaConnectoR::juliaEval(":data"))
  data_vec <- JuliaConnectoR::juliaCall("Base.getindex", data_proxy, seq_len(n_e))
  df[[value_name]] <- data_vec
  df
}

#' Validate an explicit `.dims` argument
#'
#' @param .dims Value passed as `.dims` (must not be NULL).
#' @return The validated `.dims` value.
#' @noRd
.validate_dims <- function(.dims) {
  if (length(.dims) != 1L || is.na(.dims) || !(.dims %in% c(1, 2))) {
    stop(".dims must be 1 (rows) or 2 (columns)", call. = FALSE)
  }
  .dims
}

#' Auto-detect the target dimension from expression variable names
#'
#' Inspects which index table(s) contain the variable names used in the
#' expression. Unambiguous matches resolve silently; ambiguous or
#' undetectable cases warn and default to rows (.dims = 1).
#'
#' @param vars Character vector of variable names (e.g. from `all.vars()`).
#' @param x A MatrixEntry object with `$row_indices` and `$col_indices`.
#' @return 1 (rows) or 2 (columns).
#' @noRd
.detect_dims <- function(vars, x) {
  in_rows <- any(vars %in% names(x$row_indices))
  in_cols <- any(vars %in% names(x$col_indices))
  if (in_rows && in_cols) {
    ambiguous <- unique(vars[vars %in% names(x$row_indices) & vars %in% names(x$col_indices)])
    warning(
      "Column name(s) ",
      paste0("'", ambiguous, "'", collapse = ", "),
      " found in both row and column indices; defaulting to rows (.dims = 1). ",
      "Pass .dims explicitly to disambiguate.",
      call. = FALSE
    )
    return(1)
  }
  if (in_cols) {
    return(2)
  }
  if (in_rows) {
    return(1)
  }
  warning(
    "Could not auto-detect dimension from variable names in the expression; ",
    "defaulting to rows (.dims = 1). Pass .dims explicitly to disambiguate.",
    call. = FALSE
  )
  1
}

#' Filter rows or columns of a MatrixEntry using dplyr syntax
#'
#' @title Filter MatrixEntry
#' @description Filter rows or columns of a MatrixEntry using dplyr syntax.
#'   Dimension auto-detection uses variable NAMES in the expression, so
#'   expressions without detectable variable names (e.g. tidy-select helpers
#'   such as all_of() or where(), or positional selection) cannot be
#'   auto-detected — pass `.dims` explicitly in those cases.
#'
#' @param .data A MatrixEntry object.
#' @param ... Logical expressions evaluated on row_indices or col_indices.
#' @param .dims An optional integer specifying the dimension to filter: 1 for rows, 2 for columns. If NULL (default), auto-detects based on column names.
#'
#' @return A MatrixEntry object.
#' @export
filter.MatrixEntry <- function(.data, ..., .dims = NULL) {
  if (is.null(.dims)) {
    # Auto-detect dimension
    vars <- all.vars(substitute(list(...)))
    .dims <- .detect_dims(vars, .data)
  } else {
    .dims <- .validate_dims(.dims)
  }
  
  if (.dims == 1) {
    row_df <- .data$row_indices
    row_df$.row_id <- seq_len(nrow(row_df))
    filtered_row_df <- dplyr::filter(row_df, ...)
    kept_rows <- filtered_row_df$.row_id
    logical_mask <- rep(FALSE, nrow(row_df))
    logical_mask[kept_rows] <- TRUE
    return(.data[logical_mask, ])
  } else if (.dims == 2) {
    col_df <- .data$col_indices
    col_df$.col_id <- seq_len(nrow(col_df))
    filtered_col_df <- dplyr::filter(col_df, ...)
    kept_cols <- filtered_col_df$.col_id
    logical_mask <- rep(FALSE, nrow(col_df))
    logical_mask[kept_cols] <- TRUE
    return(.data[, logical_mask])
  } else {
    stop(".dims must be 1 (rows) or 2 (columns)", call. = FALSE)
  }
}

#' Filter elements of a SeriesEntry using dplyr syntax
#'
#' @title Filter SeriesEntry
#' @description Filter elements of a SeriesEntry using dplyr syntax.
#'
#' @param .data A SeriesEntry object.
#' @param ... Logical expressions evaluated on col_indices.
#'
#' @return A SeriesEntry object.
#' @export
filter.SeriesEntry <- function(.data, ...) {
  col_df <- .data$col_indices
  col_df$.col_id <- seq_len(nrow(col_df))
  filtered_col_df <- dplyr::filter(col_df, ...)
  kept_cols <- filtered_col_df$.col_id
  logical_mask <- rep(FALSE, nrow(col_df))
  logical_mask[kept_cols] <- TRUE
  return(.data[logical_mask])
}

#' Select columns from MatrixEntry index metadata using dplyr syntax
#'
#' @title Select MatrixEntry metadata columns
#' @description Select columns from MatrixEntry index metadata using dplyr syntax.
#'   Dimension auto-detection uses variable NAMES in the expression, so
#'   tidy-select helpers (all_of(), where()) or positional selection cannot be
#'   auto-detected — pass `.dims` explicitly in those cases.
#'
#' @param .data A MatrixEntry object.
#' @param ... Columns to select.
#' @param .dims An optional integer: 1 for row_indices, 2 for col_indices. If NULL (default), auto-detected from variable names.
#'
#' @return A MatrixEntry object.
#' @export
select.MatrixEntry <- function(.data, ..., .dims = NULL) {
  if (is.null(.dims)) {
    vars <- all.vars(substitute(list(...)))
    .dims <- .detect_dims(vars, .data)
  } else {
    .dims <- .validate_dims(.dims)
  }
  
  if (.dims == 1) {
    new_row_indices <- dplyr::select(.data$row_indices, ...)
    new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_row_indices", unwrap_julia_object(.data), new_row_indices)
    return(wrap_julia_object(new_proxy))
  } else if (.dims == 2) {
    new_col_indices <- dplyr::select(.data$col_indices, ...)
    new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_col_indices", unwrap_julia_object(.data), new_col_indices)
    return(wrap_julia_object(new_proxy))
  } else {
    stop(".dims must be 1 or 2", call. = FALSE)
  }
}

#' Select columns from SeriesEntry metadata using dplyr syntax
#'
#' @title Select SeriesEntry metadata columns
#' @description Select columns from SeriesEntry metadata using dplyr syntax.
#'
#' @param .data A SeriesEntry object.
#' @param ... Columns to select.
#'
#' @return A SeriesEntry object.
#' @export
select.SeriesEntry <- function(.data, ...) {
  new_col_indices <- dplyr::select(.data$col_indices, ...)
  new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_col_indices", unwrap_julia_object(.data), new_col_indices)
  return(wrap_julia_object(new_proxy))
}

#' Mutate MatrixEntry index metadata using dplyr syntax
#'
#' @title Mutate MatrixEntry metadata columns
#' @description Mutate MatrixEntry index metadata using dplyr syntax.
#'   Dimension auto-detection uses variable NAMES in the expression, so
#'   expressions without detectable variable names cannot be auto-detected —
#'   pass `.dims` explicitly in those cases.
#'
#' @param .data A MatrixEntry object.
#' @param ... Mutate expressions.
#' @param .dims An optional integer: 1 for row_indices, 2 for col_indices. If NULL (default), auto-detected from variable names.
#'
#' @return A MatrixEntry object.
#' @export
mutate.MatrixEntry <- function(.data, ..., .dims = NULL) {
  if (is.null(.dims)) {
    vars <- all.vars(substitute(list(...)))
    .dims <- .detect_dims(vars, .data)
  } else {
    .dims <- .validate_dims(.dims)
  }
  
  if (.dims == 1) {
    new_row_indices <- dplyr::mutate(.data$row_indices, ...)
    new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_row_indices", unwrap_julia_object(.data), new_row_indices)
    return(wrap_julia_object(new_proxy))
  } else if (.dims == 2) {
    new_col_indices <- dplyr::mutate(.data$col_indices, ...)
    new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_col_indices", unwrap_julia_object(.data), new_col_indices)
    return(wrap_julia_object(new_proxy))
  } else {
    stop(".dims must be 1 or 2", call. = FALSE)
  }
}

#' Mutate SeriesEntry metadata using dplyr syntax
#'
#' @title Mutate SeriesEntry metadata columns
#' @description Mutate SeriesEntry metadata using dplyr syntax.
#'
#' @param .data A SeriesEntry object.
#' @param ... Mutate expressions.
#'
#' @return A SeriesEntry object.
#' @export
mutate.SeriesEntry <- function(.data, ...) {
  new_col_indices <- dplyr::mutate(.data$col_indices, ...)
  new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_col_indices", unwrap_julia_object(.data), new_col_indices)
  return(wrap_julia_object(new_proxy))
}

#' Rename MatrixEntry index metadata using dplyr syntax
#'
#' @title Rename MatrixEntry metadata columns
#' @description Rename MatrixEntry index metadata using dplyr syntax.
#'   Dimension auto-detection uses variable NAMES in the expression, so
#'   expressions without detectable variable names cannot be auto-detected —
#'   pass `.dims` explicitly in those cases.
#'
#' @param .data A MatrixEntry object.
#' @param ... Rename expressions.
#' @param .dims An optional integer: 1 for row_indices, 2 for col_indices. If NULL (default), auto-detected from variable names.
#'
#' @return A MatrixEntry object.
#' @export
rename.MatrixEntry <- function(.data, ..., .dims = NULL) {
  if (is.null(.dims)) {
    vars <- all.vars(substitute(list(...)))
    .dims <- .detect_dims(vars, .data)
  } else {
    .dims <- .validate_dims(.dims)
  }
  
  if (.dims == 1) {
    new_row_indices <- dplyr::rename(.data$row_indices, ...)
    new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_row_indices", unwrap_julia_object(.data), new_row_indices)
    return(wrap_julia_object(new_proxy))
  } else if (.dims == 2) {
    new_col_indices <- dplyr::rename(.data$col_indices, ...)
    new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_col_indices", unwrap_julia_object(.data), new_col_indices)
    return(wrap_julia_object(new_proxy))
  } else {
    stop(".dims must be 1 or 2", call. = FALSE)
  }
}

#' Rename SeriesEntry metadata using dplyr syntax
#'
#' @title Rename SeriesEntry metadata columns
#' @description Rename SeriesEntry metadata using dplyr syntax.
#'
#' @param .data A SeriesEntry object.
#' @param ... Rename expressions.
#'
#' @return A SeriesEntry object.
#' @export
rename.SeriesEntry <- function(.data, ...) {
  new_col_indices <- dplyr::rename(.data$col_indices, ...)
  new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_col_indices", unwrap_julia_object(.data), new_col_indices)
  return(wrap_julia_object(new_proxy))
}

#' Relocate MatrixEntry index metadata using dplyr syntax
#'
#' @title Relocate MatrixEntry metadata columns
#' @description Relocate MatrixEntry index metadata using dplyr syntax.
#'   Dimension auto-detection uses variable NAMES in the expression, so
#'   expressions without detectable variable names (e.g. positional selection)
#'   cannot be auto-detected — pass `.dims` explicitly in those cases.
#'
#' @param .data A MatrixEntry object.
#' @param ... Relocate expressions.
#' @param .dims An optional integer: 1 for row_indices, 2 for col_indices. If NULL (default), auto-detected from variable names.
#'
#' @return A MatrixEntry object.
#' @export
relocate.MatrixEntry <- function(.data, ..., .dims = NULL) {
  if (is.null(.dims)) {
    vars <- all.vars(substitute(list(...)))
    .dims <- .detect_dims(vars, .data)
  } else {
    .dims <- .validate_dims(.dims)
  }
  
  if (.dims == 1) {
    new_row_indices <- dplyr::relocate(.data$row_indices, ...)
    new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_row_indices", unwrap_julia_object(.data), new_row_indices)
    return(wrap_julia_object(new_proxy))
  } else if (.dims == 2) {
    new_col_indices <- dplyr::relocate(.data$col_indices, ...)
    new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_col_indices", unwrap_julia_object(.data), new_col_indices)
    return(wrap_julia_object(new_proxy))
  } else {
    stop(".dims must be 1 or 2", call. = FALSE)
  }
}

#' Relocate SeriesEntry metadata using dplyr syntax
#'
#' @title Relocate SeriesEntry metadata columns
#' @description Relocate SeriesEntry metadata using dplyr syntax.
#'
#' @param .data A SeriesEntry object.
#' @param ... Relocate expressions.
#'
#' @return A SeriesEntry object.
#' @export
relocate.SeriesEntry <- function(.data, ...) {
  new_col_indices <- dplyr::relocate(.data$col_indices, ...)
  new_proxy <- JuliaConnectoR::juliaCall("Juliora.update_col_indices", unwrap_julia_object(.data), new_col_indices)
  return(wrap_julia_object(new_proxy))
}

#' Slice rows or columns of a MatrixEntry using dplyr syntax
#'
#' @title Slice MatrixEntry
#' @description Slice rows or columns of a MatrixEntry using dplyr syntax.
#'
#' @param .data A MatrixEntry object.
#' @param ... Slicing expressions.
#' @param .dims An optional integer specifying the dimension: 1 for rows, 2 for columns.
#'
#' @return A MatrixEntry object.
#' @export
slice.MatrixEntry <- function(.data, ..., .dims = 1) {
  .dims <- .validate_dims(.dims)
  if (.dims == 1) {
    row_df <- .data$row_indices
    row_df$.row_id <- seq_len(nrow(row_df))
    sliced_row_df <- dplyr::slice(row_df, ...)
    kept_rows <- sliced_row_df$.row_id
    if (anyDuplicated(kept_rows) > 0L) {
      stop("slice() with duplicate indices is not supported for MatrixEntry (row/column masks cannot represent repetition)", call. = FALSE)
    }
    logical_mask <- rep(FALSE, nrow(row_df))
    logical_mask[kept_rows] <- TRUE
    return(.data[logical_mask, ])
  } else if (.dims == 2) {
    col_df <- .data$col_indices
    col_df$.col_id <- seq_len(nrow(col_df))
    sliced_col_df <- dplyr::slice(col_df, ...)
    kept_cols <- sliced_col_df$.col_id
    if (anyDuplicated(kept_cols) > 0L) {
      stop("slice() with duplicate indices is not supported for MatrixEntry (row/column masks cannot represent repetition)", call. = FALSE)
    }
    logical_mask <- rep(FALSE, nrow(col_df))
    logical_mask[kept_cols] <- TRUE
    return(.data[, logical_mask])
  } else {
    stop(".dims must be 1 or 2", call. = FALSE)
  }
}

#' Slice elements of a SeriesEntry using dplyr syntax
#'
#' @title Slice SeriesEntry
#' @description Slice elements of a SeriesEntry using dplyr syntax.
#'
#' @param .data A SeriesEntry object.
#' @param ... Slicing expressions.
#'
#' @return A SeriesEntry object.
#' @export
slice.SeriesEntry <- function(.data, ...) {
  col_df <- .data$col_indices
  col_df$.col_id <- seq_len(nrow(col_df))
  sliced_col_df <- dplyr::slice(col_df, ...)
  kept_cols <- sliced_col_df$.col_id
  if (anyDuplicated(kept_cols) > 0L) {
    stop("slice() with duplicate indices is not supported for SeriesEntry (masks cannot represent repetition)", call. = FALSE)
  }
  logical_mask <- rep(FALSE, nrow(col_df))
  logical_mask[kept_cols] <- TRUE
  return(.data[logical_mask])
}

#' Group MatrixEntry index metadata using dplyr syntax
#'
#' @title Group MatrixEntry
#' @description Group MatrixEntry index metadata using dplyr syntax.
#'   Dimension auto-detection uses variable NAMES in the expression, so
#'   expressions without detectable variable names cannot be auto-detected —
#'   pass `.dims` explicitly in those cases.
#'
#' @param .data A MatrixEntry object.
#' @param ... Columns to group by.
#' @param .add Unused.
#' @param .drop Unused.
#' @param .dims An optional integer: 1 for row_indices, 2 for col_indices. If NULL (default), auto-detected from variable names.
#'
#' @return A GroupedMatrixEntry object.
#' @export
group_by.MatrixEntry <- function(.data, ..., .add = FALSE, .drop = dplyr::group_by_drop_default(.data), .dims = NULL) {
  if (is.null(.dims)) {
    vars <- all.vars(substitute(list(...)))
    .dims <- .detect_dims(vars, .data)
  } else {
    .dims <- .validate_dims(.dims)
  }

  vars <- all.vars(substitute(list(...)))

  dims <- .dims

  groupby(.data, vars, dims = dims)
}

#' Summarize and aggregate a GroupedMatrixEntry using dplyr syntax
#'
#' @title Summarize GroupedMatrixEntry
#' @description Summarize and aggregate a GroupedMatrixEntry using dplyr syntax.
#'   Supports exactly one aggregation expression per call, using one of the
#'   supported functions: sum, mean, median, std, min, max, var.
#'   The output name (e.g. `total` in `total = sum(value)`) is ignored because
#'   the result is a MatrixEntry, not a data.frame.
#'
#' @param .data A GroupedMatrixEntry object.
#' @param ... A single aggregation call (e.g. sum(value)).
#' @param .groups Unused.
#'
#' @return A MatrixEntry object.
#' @export
summarise.GroupedMatrixEntry <- function(.data, ..., .groups = NULL) {
  exprs <- as.list(substitute(list(...))[-1])
  if (length(exprs) != 1L) {
    stop(
      "summarise() on a GroupedMatrixEntry supports exactly one aggregation expression, got ",
      length(exprs),
      call. = FALSE
    )
  }
  expr <- exprs[[1]]
  func_name <- NULL
  if (is.call(expr)) {
    fn <- expr[[1]]
    if (is.name(fn)) {
      func_name <- as.character(fn)
    } else if (is.call(fn) && length(fn) == 3L && identical(fn[[1]], quote(`::`))) {
      func_name <- as.character(fn[[3]])
    }
  }

  supported <- c("sum", "mean", "median", "std", "min", "max", "var")
  if (is.null(func_name) || !(func_name %in% supported)) {
    stop(
      "Unsupported aggregation function ",
      if (is.null(func_name)) "(expression is not a function call)" else paste0("'", func_name, "'"),
      "; supported functions are: ",
      paste(supported, collapse = ", "),
      call. = FALSE
    )
  }

  jl_func <- if (func_name == "sum") {
    "Base.sum"
  } else if (func_name == "mean") {
    "Statistics.mean"
  } else if (func_name == "median") {
    "Statistics.median"
  } else if (func_name == "std") {
    "Statistics.std"
  } else if (func_name == "var") {
    "Statistics.var"
  } else if (func_name == "min") {
    "Base.minimum"
  } else if (func_name == "max") {
    "Base.maximum"
  }

  aggregate(.data, jl_func)
}

#' @export
summarize.GroupedMatrixEntry <- summarise.GroupedMatrixEntry
