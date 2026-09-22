library(testthat)
library(Juliora)

# Core R-binding coverage: closures passed to Julia, S3 generics, Symbol
# construction, validation, and wrapper edge cases. Fixture style mirrors
# tests/testthat/test-dplyr_integrations.R.

make_core_entry <- function() {
  col_df <- data.frame(
    CountryCode = c("USA", "CHN", "DEU"),
    Sector = c("Agr", "Man", "Ser"),
    stringsAsFactors = FALSE
  )
  row_df <- data.frame(
    CountryCode = c("USA", "CHN", "DEU"),
    Sector = c("Agr", "Man", "Ser"),
    stringsAsFactors = FALSE
  )
  data_mat <- matrix(c(
    10.0, 20.0, 30.0,
    40.0, 50.0, 60.0,
    70.0, 80.0, 90.0
  ), nrow = 3, ncol = 3, byrow = TRUE)
  MatrixEntry(data_mat, col_df, row_df)
}

make_core_series <- function() {
  col_df <- data.frame(
    CountryCode = c("USA", "CHN", "DEU"),
    Sector = c("Agr", "Man", "Ser"),
    stringsAsFactors = FALSE
  )
  SeriesEntry(c(1.5, 2.5, 3.5), col_df)
}

make_core_mrio <- function() {
  indices <- data.frame(
    CountryCode = c("USA", "CHN"),
    Sector = c("Agr", "Man"),
    stringsAsFactors = FALSE
  )
  fd_indices <- data.frame(Category = "Households", stringsAsFactors = FALSE)
  va_indices <- data.frame(Category = "Value added", stringsAsFactors = FALSE)
  MRIO(
    MatrixEntry(matrix(c(10, 2, 3, 8), nrow = 2, byrow = TRUE), indices, indices),
    MatrixEntry(matrix(c(5, 7), nrow = 2), fd_indices, indices),
    MatrixEntry(matrix(c(4, 6), nrow = 1), indices, va_indices)
  )
}

test_that("filter_rows works with a real R closure and filters values", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()

  filtered <- filter_rows(me, function(row) row$CountryCode == "USA")

  expect_s3_class(filtered, "MatrixEntry")
  expect_equal(dim(filtered), c(1, 3))
  expect_equal(as.character(filtered$row_indices$CountryCode), "USA")
  expect_equal(as.vector(filtered$data), c(10, 20, 30))
})

test_that("filter_cols works with a real R closure and filters values", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()

  filtered <- filter_cols(me, function(col) col$Sector == "Man")

  expect_s3_class(filtered, "MatrixEntry")
  expect_equal(dim(filtered), c(3, 1))
  expect_equal(as.character(filtered$col_indices$Sector), "Man")
  expect_equal(as.vector(filtered$data), c(20, 50, 80))
})

test_that("filter_matrix works with two R closures", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()

  filtered <- filter_matrix(
    me,
    function(row) row$CountryCode %in% c("USA", "CHN"),
    function(col) col$Sector %in% c("Agr", "Ser")
  )

  expect_s3_class(filtered, "MatrixEntry")
  expect_equal(dim(filtered), c(2, 2))
  expect_equal(as.vector(filtered$data), c(10, 40, 30, 60))
})

test_that("filter helpers reject non-function conditions", {
  expect_error(filter_rows(list(), "not-a-function"), "must be a MatrixEntry")
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()

  expect_error(filter_rows(me, "USA"), "must be a function")
  expect_error(filter_cols(me, 42), "must be a function")
  expect_error(
    filter_matrix(me, function(row) TRUE, "not-a-function"),
    "must be functions"
  )
})

test_that("add_calculated_column works with an R closure", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()

  out <- add_calculated_column(me, "Big", function(row) row$CountryCode == "USA")

  expect_s3_class(out, "MatrixEntry")
  expect_true("Big" %in% names(out$row_indices))
  expect_equal(out$row_indices$Big, c(TRUE, FALSE, FALSE))
})

test_that("aggregate dispatches on functions and strings with equal values", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()
  grouped <- groupby(me, "CountryCode")

  # Julia calls the aggregation function as func(block; dims = 1) and needs
  # a row-matrix back, so R closures must follow that convention (base sum
  # returns a scalar and cannot be used here).
  colsum_closure <- function(m, dims = 1) matrix(colSums(m), nrow = 1)
  from_fn <- aggregate(grouped, colsum_closure)
  from_str <- aggregate(grouped, "sum")

  expect_s3_class(from_fn, "MatrixEntry")
  expect_s3_class(from_str, "MatrixEntry")
  # One row per group; every group holds a single original row here.
  expect_equal(dim(from_fn), c(3, 3))
  expect_equal(sort(as.vector(from_fn$data)), sort(as.vector(me$data)))
  expect_equal(sort(as.vector(from_str$data)), sort(as.vector(me$data)))
  expect_error(aggregate(grouped, 42), "must be a function or a single character string")
})

test_that("groupby_matrix accepts an agg_func string", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()

  df <- groupby_matrix(me, "CountryCode", agg_func = "sum", rows = TRUE)

  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 3)
  expect_equal(sum(df$value), sum(me$data))
})

test_that("groupby works with non-syntactic column names", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  col_df <- data.frame(
    CountryCode = c("USA", "CHN", "DEU"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  col_df[["GDP per capita"]] <- c(10, 20, 30)
  row_df <- col_df
  me <- MatrixEntry(diag(3), col_df, row_df)

  grouped <- groupby(me, "GDP per capita")
  expect_s3_class(grouped, "GroupedMatrixEntry")

  agg <- aggregate(grouped, "sum")
  expect_s3_class(agg, "MatrixEntry")
  expect_equal(dim(agg), c(3, 3))

  df <- groupby_matrix(me, "GDP per capita", agg_func = "sum", rows = TRUE)
  expect_equal(nrow(df), 3)
  expect_equal(sum(df$value), 3)

  out <- add_calculated_column(me, "VA share (%)", function(row) row$CountryCode == "USA")
  expect_true("VA share (%)" %in% names(out$row_indices))
})

test_that("stats::aggregate and base::drop are not masked", {
  expect_equal(
    stats::aggregate(mtcars["mpg"], list(cyl = mtcars$cyl), mean)$mpg,
    as.vector(tapply(mtcars$mpg, mtcars$cyl, mean))
  )
  expect_identical(base::drop(matrix(1:4, 2)), matrix(1:4, 2))

  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()
  grouped <- groupby(me, "CountryCode")

  # Unqualified calls dispatch to the Juliora S3 methods.
  agg <- aggregate(grouped, "sum")
  expect_s3_class(agg, "MatrixEntry")

  # NOTE: base::drop is not an S3 generic (it is `.Internal(drop(x))`), so
  # unqualified `drop(entry, ...)` cannot dispatch; the method is called
  # explicitly. See the H12 report note.
  dropped <- drop.MatrixEntry(me, list(CountryCode = "USA"))
  expect_s3_class(dropped, "MatrixEntry")
  expect_equal(dim(dropped), c(2, 3))
  expect_equal(dropped$row_indices$CountryCode, c("CHN", "DEU"))
})

test_that("unqualified drop() cannot dispatch (base::drop is not generic)", {
  skip("base::drop is `.Internal(drop(x))`, not an S3 generic: `drop(entry, ...)` cannot reach drop.MatrixEntry. Orchestrator decision pending (see H12 report note).")
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()
  dropped <- drop(me, list(CountryCode = "USA"))
  expect_equal(dim(dropped), c(2, 3))
})

test_that("drop validates vectorized and NA dims", {
  mock <- structure(list(), class = "MatrixEntry")

  # base::drop is not generic, so the methods are invoked explicitly here.
  expect_error(drop.MatrixEntry(mock, list(a = 1), dims = c(1, 2)), "must be 1 or 2")
  expect_error(drop.MatrixEntry(mock, list(a = 1), dims = NA), "must be 1 or 2")

  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()
  expect_error(drop.MatrixEntry(me, list(CountryCode = "USA"), dims = c(1, 2)), "must be 1 or 2")
  expect_error(drop.MatrixEntry(me, list(CountryCode = "USA"), dims = NA), "must be 1 or 2")
  expect_error(drop_mut(me, list(CountryCode = "USA"), dims = c(1, 2)), "must be 1 or 2")
  expect_error(
    groupby(me, "CountryCode", dims = c(1, 2)),
    "must be 1 or 2"
  )
})

test_that("se[] returns the data vector", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  se <- make_core_series()

  expect_identical(as.vector(se[]), as.vector(se$data))
  expect_equal(as.vector(se[]), c(1.5, 2.5, 3.5))
})

test_that("drop_mut accepts a list of named lists and mutates in place", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()

  me <- drop_mut(me, list(list(CountryCode = "USA"), list(CountryCode = "CHN")))
  expect_s3_class(me, "MatrixEntry")
  expect_equal(dim(me), c(1, 3))
  expect_equal(as.character(me$row_indices$CountryCode), "DEU")

  me <- drop_mut(me, list(CountryCode = "DEU"))
  expect_equal(dim(me), c(0, 3))

  expect_error(drop_mut(make_core_entry(), list("USA")), "must be a named list")
})

test_that("printing empty entries does not error", {
  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  me <- make_core_entry()
  se <- make_core_series()

  empty_me <- filter_rows(me, function(row) FALSE)
  expect_equal(dim(empty_me), c(0, 3))
  expect_no_error(invisible(capture.output(print(empty_me))))

  empty_se <- se[c(FALSE, FALSE, FALSE)]
  expect_equal(length(empty_se), 0)
  expect_no_error(invisible(capture.output(print(empty_se))))
})

test_that("mrio field access validates names and reads fields", {
  mock <- structure(list(), julia_proxy = NULL, class = "MRIO")
  expect_error(mock$nonexistent, "Valid fields")

  skip_if_not(is_julia_available(), "Julia environment not available for testing")
  mrio <- make_core_mrio()
  expect_s3_class(mrio$Z, "MatrixEntry")
  expect_error(mrio$nonexistent, "Valid fields")
  # `$env` is an EnvironmentalExtension at HEAD but becomes NULL once MRIO
  # objects built without environmental data carry `env = nothing`.
  env <- mrio$env
  expect_true(is.null(env) || inherits(env, "EnvironmentalExtension"))
})
