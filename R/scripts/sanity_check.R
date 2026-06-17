# Set local library path to load Juliora
.libPaths(c("R_libs", .libPaths()))
library(Juliora)

# Configure database location and parameters
gloria_path <- "data/GLORIA/2019/"
gloria_year <- 2019
gloria_version <- 60

cat("1. Loading GLORIA database...\n")
db <- parse_gloria(gloria_path, gloria_year, version = gloria_version)

# 2. Define Country Code list for Europe (EU + UK)
EU_codes <- c(
  "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", 
  "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", 
  "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "GBR"
)

cat("2. Extracting and summing European Final Demand...\n")
# We pull Y's metadata and Y's data (Y is very small, so this is extremely fast)
eu_cols_mask <- db$Y$col_indices$CountryCode %in% EU_codes
total_eu_demand <- sum(db$Y$data[, eu_cols_mask])

cat("3. Solving Globally Induced Production (processed entirely in Julia)...\n")
# Call general induced_production with no producer filter (returns all producing sectors)
global_induced_df <- induced_production(db, consumer_countries = EU_codes)
total_global_induced <- sum(global_induced_df$InducedProduction)

cat(sprintf("Total European Final Demand: %f\n", total_eu_demand))
cat(sprintf("Total Globally Induced Production: %f\n", total_global_induced))

# 4. Calculate Ratio
multiplier_ratio <- total_global_induced / total_eu_demand
cat("\n--- Sanity Check Results ---\n")
cat(sprintf("Ratio (Globally Induced Output / European Demand): %f\n", multiplier_ratio))

if (multiplier_ratio > 1) {
  cat("SUCCESS: The ratio is greater than 1, confirming a valid economic multiplier effect.\n")
} else {
  cat("WARNING: The ratio is less than or equal to 1, indicating a potential calculation issue.\n")
}

# 5. Units analysis
cat("\n--- Units Analysis ---\n")
cat("Note on units:\n")
cat("GLORIA monetary values are reported in thousands of USD (1000 USD).\n")
cat(sprintf("If the raw final demand is %f, this corresponds to:\n", total_eu_demand))
cat(sprintf("  - In raw database units: %f thousands\n", total_eu_demand))
cat(sprintf("  - In billions: %.3f Billion USD\n", total_eu_demand / 1e6))
cat(sprintf("  - In trillions: %.3f Trillion USD\n", total_eu_demand / 1e9))
