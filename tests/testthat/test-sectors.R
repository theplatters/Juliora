library(testthat)
library(Juliora)

if (!Juliora::is_julia_available()) {
  skip("Julia environment not available for testing")
}

make_sector_test_mrio <- function(indices) {
  n <- nrow(indices)
  transactions <- MatrixEntry(matrix(0, nrow = n, ncol = n), indices, indices)
  final_demand <- MatrixEntry(
    matrix(1, nrow = n, ncol = 1),
    data.frame(Category = "Households"),
    indices
  )
  value_added <- MatrixEntry(
    matrix(1, nrow = 1, ncol = n),
    indices,
    data.frame(Category = "Value added")
  )
  MRIO(transactions, final_demand, value_added)
}

test_that("sectors returns the same character representation for Gloria and Eora", {
  gloria_indices <- data.frame(
    CountryCode = rep(c("AUT", "DEU"), each = 2),
    Sector = rep(c("Agriculture", "Manufacturing"), 2)
  )
  eora_indices <- data.frame(
    CountryCode = rep(c("AUT", "DEU"), each = 2),
    Industry = rep(c("01", "02"), 2),
    Sector = rep(c("Agriculture", "Manufacturing"), 2)
  )

  gloria <- make_sector_test_mrio(gloria_indices)
  eora <- make_sector_test_mrio(eora_indices)
  expected <- c("Agriculture", "Manufacturing")

  expect_type(sectors(gloria), "character")
  expect_identical(sectors(gloria), expected)
  expect_identical(sectors(eora), expected)
  expect_identical(sectors(gloria), sectors(eora))
  expect_identical(sector(gloria), expected)
})

test_that("sectors reports invalid inputs as native R errors", {
  expect_error(sectors(42), "Julia Error")
})
