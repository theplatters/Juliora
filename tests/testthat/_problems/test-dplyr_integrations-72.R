# Extracted from test-dplyr_integrations.R:72

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "Juliora", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
library(testthat)
library(Juliora)
library(dplyr)
if (!Juliora::is_julia_available()) {
  skip("Julia environment not available for testing")
}
context("Testing dplyr integrations for Juliora R package")
col_df <- data.frame(
  Sector = c("Agr", "Man", "Ser"),
  Type = c("Primary", "Secondary", "Tertiary"),
  stringsAsFactors = FALSE
)
row_df <- data.frame(
  Country = c("USA", "CHN", "DEU"),
  Region = c("NA", "Asia", "EU"),
  GDP = c(20000, 14000, 4000),
  Developed = c(TRUE, FALSE, TRUE),
  stringsAsFactors = FALSE
)
data_mat <- matrix(c(
  10.0, 20.0, 30.0,
  40.0, 50.0, 60.0,
  70.0, 80.0, 90.0
), nrow = 3, ncol = 3, byrow = TRUE)
me <- MatrixEntry(data_mat, col_df, row_df)
se <- SeriesEntry(c(1.5, 2.5, 3.5), col_df)

# test -------------------------------------------------------------------------
filtered_row <- me %>% dplyr::filter(GDP > 5000)
expect_s3_class(filtered_row, "MatrixEntry")
expect_equal(dim(filtered_row), c(2, 3))
expect_equal(filtered_row$row_indices$Country, c("USA", "CHN"))
filtered_col <- me %>% dplyr::filter(Type %in% c("Primary", "Tertiary"))
expect_s3_class(filtered_col, "MatrixEntry")
expect_equal(dim(filtered_col), c(3, 2))
expect_equal(filtered_col$col_indices$Sector, c("Agr", "Ser"))
filtered_explicit <- me %>% dplyr::filter(GDP > 5000, .dims = 1)
expect_equal(dim(filtered_explicit), c(2, 3))
filtered_se <- se %>% dplyr::filter(Type == "Secondary")
expect_s3_class(filtered_se, "SeriesEntry")
expect_equal(length(filtered_se), 1)
expect_equal(filtered_se$col_indices$Sector, "Man")
