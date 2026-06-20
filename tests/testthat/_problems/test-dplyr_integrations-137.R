# Extracted from test-dplyr_integrations.R:137

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
grouped <- me %>% dplyr::group_by(Region)
expect_s3_class(grouped, "GroupedMatrixEntry")
summed <- grouped %>% dplyr::summarise(total = sum(value))
