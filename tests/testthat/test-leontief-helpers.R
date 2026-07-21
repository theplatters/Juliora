library(testthat)
library(Juliora)

if (!Juliora::is_julia_available()) {
  skip("Julia environment not available for testing")
}

make_test_factorization <- function() {
  industry_indices <- data.frame(
    CountryCode = c("AUT", "DEU"),
    Sector = c("A", "B")
  )
  final_demand_indices <- data.frame(Category = "Households")
  value_added_indices <- data.frame(Category = "Value added")

  transactions <- matrix(c(10, 2, 3, 8), nrow = 2, byrow = TRUE)
  final_demand <- matrix(c(5, 7), nrow = 2)
  value_added <- matrix(c(4, 6), nrow = 1)

  mrio <- MRIO(
    MatrixEntry(transactions, industry_indices, industry_indices),
    MatrixEntry(final_demand, final_demand_indices, industry_indices),
    MatrixEntry(value_added, industry_indices, value_added_indices)
  )

  list(
    factorization = mrio$L,
    coefficients = transactions / rep(c(17, 18), each = 2)
  )
}

test_that("solve_leontief handles vector and matrix final demand", {
  fixture <- make_test_factorization()

  vector_demand <- c(10, 20)
  vector_result <- solve_leontief(fixture$factorization, vector_demand)
  expected_vector <- solve(diag(2) - fixture$coefficients, vector_demand)

  expect_type(vector_result, "double")
  expect_null(dim(vector_result))
  expect_equal(vector_result, as.vector(expected_vector), tolerance = 1e-12)

  matrix_demand <- matrix(c(10L, 20L, 2L, 4L), nrow = 2)
  matrix_result <- solve_leontief(fixture$factorization, matrix_demand)
  expected_matrix <- solve(diag(2) - fixture$coefficients, matrix_demand)

  expect_type(matrix_result, "double")
  expect_equal(dim(matrix_result), dim(matrix_demand))
  expect_equal(matrix_result, expected_matrix, tolerance = 1e-12)
})

test_that("solve_leontief validates inputs and translates Julia errors", {
  fixture <- make_test_factorization()

  expect_error(
    solve_leontief(list(), c(1, 2)),
    "must be a LeontiefFactorization"
  )
  expect_error(
    solve_leontief(fixture$factorization, c("1", "2")),
    "must be a numeric vector or matrix"
  )
  expect_error(
    solve_leontief(fixture$factorization, array(1, dim = c(2, 1, 1))),
    "must be a numeric vector or matrix"
  )
  expect_error(
    solve_leontief(fixture$factorization, numeric(3)),
    "Julia Error"
  )
})

test_that("sum_rows and sum_cols preserve matrix orientation", {
  values <- matrix(1:6, nrow = 2, byrow = TRUE)

  expect_equal(sum_rows(values), c(6, 15))
  expect_equal(sum_cols(values), c(5, 7, 9))
  expect_null(dim(sum_rows(values)))
  expect_null(dim(sum_cols(values)))

  decimal_values <- matrix(c(0.5, 1.5, 2.5, 3.5), nrow = 2)
  expect_equal(sum_rows(decimal_values), rowSums(decimal_values))
  expect_equal(sum_cols(decimal_values), colSums(decimal_values))
})

test_that("sum_rows and sum_cols accept MatrixEntry objects", {
  values <- matrix(as.numeric(1:6), nrow = 2, byrow = TRUE)
  rows <- data.frame(Row = c("A", "B"))
  cols <- data.frame(Column = c("X", "Y", "Z"))
  entry <- MatrixEntry(values, cols, rows)

  expect_equal(sum_rows(entry), c(6, 15))
  expect_equal(sum_cols(entry), c(5, 7, 9))
})

test_that("sum helpers validate inputs", {
  expect_error(sum_rows(c(1, 2)), "must be a numeric matrix")
  expect_error(sum_cols(data.frame(x = c(1, 2))), "must be a numeric matrix")
  expect_error(sum_rows(matrix(c("a", "b"))), "must be a numeric matrix")
  expect_error(sum_cols(matrix(c(TRUE, FALSE))), "must be a numeric matrix")
})

test_that("sum helpers handle large matrices across the Julia boundary", {
  values <- matrix(seq_len(100000), ncol = 1)

  expect_equal(sum_rows(values), as.vector(values))
  expect_equal(sum_cols(values), sum(values))
})
