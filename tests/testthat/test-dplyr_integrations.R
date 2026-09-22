library(testthat)
library(Juliora)
library(dplyr)

if (!Juliora::is_julia_available()) {
  skip("Julia environment not available for testing")
}

context("Testing dplyr integrations for Juliora R package")

# Setup dummy data for testing
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

test_that("as.data.frame conversion works for MatrixEntry and SeriesEntry", {
  # MatrixEntry conversion
  df_me <- as.data.frame(me)
  expect_s3_class(df_me, "data.frame")
  expect_equal(nrow(df_me), 9)
  expect_true("row_Country" %in% names(df_me))
  expect_true("col_Sector" %in% names(df_me))
  expect_true("value" %in% names(df_me))
  
  # SeriesEntry conversion
  df_se <- as.data.frame(se)
  expect_s3_class(df_se, "data.frame")
  expect_equal(nrow(df_se), 3)
  expect_true("Sector" %in% names(df_se))
  expect_true("value" %in% names(df_se))
  expect_equal(df_se$value, c(1.5, 2.5, 3.5))
})

test_that("filter S3 method works for MatrixEntry and SeriesEntry", {
  # MatrixEntry row filtering (auto-detected)
  filtered_row <- me %>% dplyr::filter(GDP > 5000)
  expect_s3_class(filtered_row, "MatrixEntry")
  expect_equal(dim(filtered_row), c(2, 3))
  expect_equal(filtered_row$row_indices$Country, c("USA", "CHN"))
  
  # MatrixEntry column filtering (auto-detected)
  filtered_col <- me %>% dplyr::filter(Type %in% c("Primary", "Tertiary"))
  expect_s3_class(filtered_col, "MatrixEntry")
  expect_equal(dim(filtered_col), c(3, 2))
  expect_equal(filtered_col$col_indices$Sector, c("Agr", "Ser"))
  
  # Explicit .dims argument
  filtered_explicit <- me %>% dplyr::filter(GDP > 5000, .dims = 1)
  expect_equal(dim(filtered_explicit), c(2, 3))
  
  # SeriesEntry filtering
  filtered_se <- se %>% dplyr::filter(Type == "Secondary")
  expect_s3_class(filtered_se, "SeriesEntry")
  expect_equal(length(filtered_se), 1)
  expect_equal(as.character(filtered_se$col_indices$Sector), "Man")
  expect_equal(as.vector(filtered_se$data), 2.5)
  
  # Filtering resulting in empty matrix
  filtered_empty <- me %>% dplyr::filter(GDP > 100000)
  expect_equal(dim(filtered_empty), c(0, 3))
  expect_equal(nrow(filtered_empty$row_indices), 0)
})

test_that("select S3 method works for MatrixEntry and SeriesEntry", {
  # Select rows of MatrixEntry
  selected_row <- me %>% dplyr::select(Country, GDP)
  expect_s3_class(selected_row, "MatrixEntry")
  expect_equal(names(selected_row$row_indices), c("Country", "GDP"))
  expect_equal(dim(selected_row), c(3, 3))
  
  # Select cols of SeriesEntry
  selected_se <- se %>% dplyr::select(Sector)
  expect_s3_class(selected_se, "SeriesEntry")
  expect_equal(names(selected_se$col_indices), "Sector")
  expect_equal(length(selected_se), 3)
})

test_that("mutate S3 method works for MatrixEntry and SeriesEntry", {
  # Mutate rows of MatrixEntry
  mutated_row <- me %>% dplyr::mutate(GDP_Double = GDP * 2)
  expect_s3_class(mutated_row, "MatrixEntry")
  expect_true("GDP_Double" %in% names(mutated_row$row_indices))
  expect_equal(mutated_row$row_indices$GDP_Double, c(40000, 28000, 8000))
  
  # Mutate cols of SeriesEntry
  mutated_se <- se %>% dplyr::mutate(New_Col = paste0(Sector, "_type"))
  expect_s3_class(mutated_se, "SeriesEntry")
  expect_true("New_Col" %in% names(mutated_se$col_indices))
  expect_equal(mutated_se$col_indices$New_Col, c("Agr_type", "Man_type", "Ser_type"))
})

test_that("rename and relocate S3 methods work", {
  # Rename MatrixEntry rows
  renamed_row <- me %>% dplyr::rename(Nation = Country)
  expect_true("Nation" %in% names(renamed_row$row_indices))
  expect_false("Country" %in% names(renamed_row$row_indices))
  
  # Relocate MatrixEntry rows
  relocated_row <- me %>% dplyr::relocate(GDP, .before = Country)
  expect_equal(names(relocated_row$row_indices)[1], "GDP")
})

test_that("slice S3 method works for MatrixEntry and SeriesEntry", {
  # Slice rows of MatrixEntry
  sliced_row <- me %>% dplyr::slice(1:2)
  expect_equal(dim(sliced_row), c(2, 3))
  expect_equal(sliced_row$row_indices$Country, c("USA", "CHN"))
  
  # Slice elements of SeriesEntry
  sliced_se <- se %>% dplyr::slice(c(1, 3))
  expect_equal(length(sliced_se), 2)
  expect_equal(sliced_se$col_indices$Sector, c("Agr", "Ser"))
})

test_that("group_by and summarize/summarise work for aggregation", {
  # Group by and sum aggregate
  grouped <- me %>% dplyr::group_by(Region)
  expect_s3_class(grouped, "GroupedMatrixEntry")

  summed <- grouped %>% dplyr::summarise(total = sum(value))
  expect_s3_class(summed, "MatrixEntry")
  expect_equal(dim(summed), c(3, 3)) # 3 groups of Region

  # Test with alternate spelling summarize
  summed2 <- grouped %>% dplyr::summarize(total = sum(value))
  expect_equal(dim(summed2), c(3, 3))
})

test_that("summarise rejects multiple, zero, or unsupported expressions", {
  grouped <- me %>% dplyr::group_by(Region)

  # Multiple expressions -> error mentioning "exactly one"
  expect_error(
    grouped %>% dplyr::summarise(a = sum(value), b = mean(value)),
    "exactly one"
  )

  # Zero expressions -> error mentioning "exactly one"
  expect_error(
    grouped %>% dplyr::summarise(),
    "exactly one"
  )

  # Unsupported function -> error listing supported functions
  expect_error(
    grouped %>% dplyr::summarise(total = first(value)),
    "sum.*mean.*median|supported"
  )

  # Non-call expression -> error listing supported functions
  expect_error(
    grouped %>% dplyr::summarise(total),
    "supported"
  )

  # Single supported expression still works (mean + output name ignored)
  averaged <- grouped %>% dplyr::summarise(whatever = mean(value))
  expect_s3_class(averaged, "MatrixEntry")
  expect_equal(dim(averaged), c(3, 3))
})

test_that("explicit .dims is honored even when names exist in both indices", {
  col_both <- data.frame(
    Sector = c("Agr", "Man", "Ser"),
    Region = c("X", "Y", "Z"),
    stringsAsFactors = FALSE
  )
  row_both <- data.frame(
    Country = c("USA", "CHN", "DEU"),
    Region = c("NA", "Asia", "EU"),
    stringsAsFactors = FALSE
  )
  me_both <- MatrixEntry(data_mat, col_both, row_both)

  # select: explicit .dims wins over auto-detection
  sel_cols <- me_both %>% dplyr::select(Region, .dims = 2)
  expect_equal(names(sel_cols$col_indices), "Region")
  expect_equal(names(sel_cols$row_indices), c("Country", "Region"))
  sel_rows <- me_both %>% dplyr::select(Region, .dims = 1)
  expect_equal(names(sel_rows$row_indices), "Region")
  expect_equal(names(sel_rows$col_indices), c("Sector", "Region"))

  # mutate: explicit .dims wins
  mut_cols <- me_both %>% dplyr::mutate(Region2 = paste0(Region, "!"), .dims = 2)
  expect_true("Region2" %in% names(mut_cols$col_indices))
  expect_false("Region2" %in% names(mut_cols$row_indices))

  # rename: explicit .dims wins
  ren_cols <- me_both %>% dplyr::rename(Area = Region, .dims = 2)
  expect_true("Area" %in% names(ren_cols$col_indices))
  expect_true("Region" %in% names(ren_cols$row_indices))

  # relocate: explicit .dims wins (row table has 2 cols so order change is visible)
  rel_cols <- me_both %>% dplyr::relocate(Region, .dims = 2)
  expect_equal(names(rel_cols$col_indices)[1], "Region")

  # filter: explicit .dims honored on the overlapping fixture
  filt_rows <- me_both %>% dplyr::filter(Region == "NA", .dims = 1)
  expect_equal(dim(filt_rows), c(1, 3))
  filt_cols <- me_both %>% dplyr::filter(Region == "X", .dims = 2)
  expect_equal(dim(filt_cols), c(3, 1))

  # invalid .dims values give clear errors
  expect_error(me %>% dplyr::select(Country, .dims = NA), ".dims must be")
  expect_error(me %>% dplyr::select(Country, .dims = c(1, 2)), ".dims must be")
  expect_error(me %>% dplyr::select(Country, .dims = 3), ".dims must be")
  expect_error(me %>% dplyr::mutate(x = 1, .dims = 3), ".dims must be")
  expect_error(me %>% dplyr::filter(GDP > 1, .dims = 0), ".dims must be")
})

test_that("ambiguous or undetectable dimensions warn and default to rows", {
  col_both <- data.frame(
    Sector = c("Agr", "Man", "Ser"),
    Region = c("X", "Y", "Z"),
    stringsAsFactors = FALSE
  )
  row_both <- data.frame(
    Country = c("USA", "CHN", "DEU"),
    Region = c("NA", "Asia", "EU"),
    stringsAsFactors = FALSE
  )
  me_both <- MatrixEntry(data_mat, col_both, row_both)

  # Name in both index tables -> warning, defaults to rows
  expect_warning(
    res <- me_both %>% dplyr::select(Region),
    "both row and column"
  )
  expect_equal(names(res$row_indices), "Region")

  expect_warning(me_both %>% dplyr::filter(Region == "NA"), "both row and column")

  # No detectable variable names (positional select) -> warning, defaults to rows
  expect_warning(
    res_pos <- me %>% dplyr::select(1),
    "auto-detect"
  )
  expect_equal(names(res_pos$row_indices), "Country")
})

test_that("group_by supports .dims = 2 for column grouping", {
  col_both <- data.frame(
    Sector = c("Agr", "Man", "Ser"),
    Region = c("X", "Y", "Z"),
    stringsAsFactors = FALSE
  )
  row_both <- data.frame(
    Country = c("USA", "CHN", "DEU"),
    Region = c("NA", "Asia", "EU"),
    stringsAsFactors = FALSE
  )
  me_both <- MatrixEntry(data_mat, col_both, row_both)

  grouped_cols <- me_both %>% dplyr::group_by(Sector, .dims = 2)
  expect_s3_class(grouped_cols, "GroupedMatrixEntry")
  summed_cols <- grouped_cols %>% dplyr::summarise(total = sum(value))
  expect_s3_class(summed_cols, "MatrixEntry")

  # Ambiguous grouping variable warns without explicit .dims
  expect_warning(me_both %>% dplyr::group_by(Region), "both row and column")
})

test_that("slice with duplicate indices errors", {
  expect_error(me %>% dplyr::slice(c(1, 1)), "duplicate")
  expect_error(se %>% dplyr::slice(c(1, 1)), "duplicate")

  # Existing non-duplicate behavior still works, incl. negative indices
  expect_equal(dim(me %>% dplyr::slice(1:2)), c(2, 3))
  expect_equal(dim(me %>% dplyr::slice(-1)), c(2, 3))
  expect_equal(length(se %>% dplyr::slice(c(1, 3))), 2)
})

test_that("as.data.frame works on an empty SeriesEntry", {
  se_empty <- se %>% dplyr::filter(Type == "NOPE")
  expect_equal(length(se_empty), 0)
  df_empty <- as.data.frame(se_empty)
  expect_s3_class(df_empty, "data.frame")
  expect_equal(nrow(df_empty), 0)
  expect_true("Sector" %in% names(df_empty))
  expect_true("value" %in% names(df_empty))
})
