# Set local library path to load Juliora
.libPaths(c("R_libs", .libPaths()))
library(Juliora)
library(ggplot2)

# Create plots directory if it doesn't exist
dir.create("R/plots", showWarnings = FALSE, recursive = TRUE)

# Configure database location and parameters
gloria_path <- "data/GLORIA/2019/"
gloria_year <- 2019
gloria_version <- 60

cat("1. Loading GLORIA database...\n")
db <- parse_gloria(gloria_path, gloria_year, version = gloria_version)

# 2. Define Country Code lists (ISO 3-letter codes)
# European Union + United Kingdom
EU_codes <- c(
  "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", 
  "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", 
  "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "GBR"
)

# Region lists
South_America_codes <- c(
  "ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PRY", "PER", "SUR", "URY", "VEN"
)

Africa_codes <- c(
  "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", 
  "COM", "COG", "COD", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH", "GAB", 
  "GMB", "GHA", "GIN", "GNB", "CIV", "KEN", "LSO", "LBR", "LBY", "MDG", 
  "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA", 
  "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", 
  "TUN", "UGA", "ZMB", "ZWE"
)

# Asia (including Middle East, Central Asia, South Asia, East Asia, and Southeast Asia)
Asia_codes <- c(
  "AFG", "ARM", "AZE", "BHR", "BGD", "BTN", "BRN", "KHM", "CHN", "CYP", 
  "GEO", "IND", "IDN", "IRN", "IRQ", "ISR", "JPN", "JOR", "KAZ", "KWT", 
  "KGZ", "LAO", "LBN", "MYS", "MDV", "MNG", "MMR", "NPL", "OMN", "PAK", 
  "PSE", "PHL", "QAT", "SAU", "SGP", "KOR", "LKA", "SYR", "TWN", "TJK", 
  "THA", "TLS", "TUR", "TKM", "ARE", "UZB", "VNM", "YEM"
)

Asia_no_china_codes <- setdiff(Asia_codes, "CHN")

# Function to run calculations and print summaries
run_regional_analysis <- function(region_name, producer_codes) {
  cat(sprintf("\n--- Analyzing Induced Production: %s ---\n", region_name))
  
  # Perform the calculation in Julia using the general function with named parameters
  df <- induced_production(db, consumer_countries = EU_codes, producer_countries = producer_codes)
  
  total_induced <- sum(df$InducedProduction)
  cat(sprintf("Total production induced: %f\n", total_induced))
  
  # Sort and print the top 5 sectors
  df_sorted <- df[order(-df$InducedProduction), ]
  cat("Top 5 induced country-sectors:\n")
  print(head(df_sorted, 5))
  
  return(list(total = total_induced, df = df))
}

# Run the calculations
sa_res <- run_regional_analysis("South America", South_America_codes)
af_res <- run_regional_analysis("Africa", Africa_codes)
asia_res <- run_regional_analysis("Asia (including China)", Asia_codes)
asia_no_chn_res <- run_regional_analysis("Asia (excluding China)", Asia_no_china_codes)

# Extract results
sa_total <- sa_res$total
sa_df <- sa_res$df

af_total <- af_res$total
af_df <- af_res$df

asia_total <- asia_res$total
asia_df <- asia_res$df

asia_no_chn_total <- asia_no_chn_res$total
asia_no_chn_df <- asia_no_chn_res$df

# Summary Comparison Data Frame
summary_df <- data.frame(
  Region = c("South America", "Africa", "Asia (with China)", "Asia (excluding China)"),
  InducedProduction = c(sa_total, af_total, asia_total, asia_no_chn_total)
)

cat("\n================ Summary Comparison ================\n")
print(summary_df)

# ==================== ggplot2 Visualization ====================
cat("\nGenerating ggplot2 visualizations...\n")

summary_df$Region <- factor(summary_df$Region, levels = summary_df$Region)

# Plot 1: Total Induced Production by Region
p1 <- ggplot(summary_df, aes(x = Region, y = InducedProduction / 1e6, fill = Region)) +
  geom_bar(stat = "identity", width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = sprintf("$%.1f B", InducedProduction / 1e6)), vjust = -0.5, fontface = "bold", size = 4) +
  scale_fill_manual(values = c("#4A90E2", "#E28B4A", "#4AE290", "#8D4AE2")) +
  labs(
    title = "Production Induced by European Demand (2019)",
    subtitle = "Computed entirely in Julia via General Leontief Solver",
    x = "Region",
    y = "Induced Production (Billions USD)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16, margin = margin(b = 8)),
    plot.subtitle = element_text(color = "grey40", size = 11, margin = margin(b = 15)),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10)),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

ggsave("R/plots/induced_production_by_region.png", plot = p1, width = 8.5, height = 5.5, dpi = 300)
cat("- Saved R/plots/induced_production_by_region.png\n")

# Plot 2: Top Induced Country-Sectors
sa_df$Region <- "South America"
af_df$Region <- "Africa"
asia_df$Region <- "Asia (with China)"
asia_no_chn_df$Region <- "Asia (excluding China)"

top5_sa <- head(sa_df[order(-sa_df$InducedProduction), ], 5)
top5_af <- head(af_df[order(-af_df$InducedProduction), ], 5)
top5_asia <- head(asia_df[order(-asia_df$InducedProduction), ], 5)
top5_asia_no_chn <- head(asia_no_chn_df[order(-asia_no_chn_df$InducedProduction), ], 5)

top5_all <- rbind(top5_sa, top5_af, top5_asia, top5_asia_no_chn)
top5_all$Label <- paste0(top5_all$CountryCode, " - ", top5_all$Sector)

# Clean up long labels
top5_all$Label <- ifelse(nchar(top5_all$Label) > 45, paste0(substr(top5_all$Label, 1, 42), "..."), top5_all$Label)
top5_all$Region <- factor(top5_all$Region, levels = c("South America", "Africa", "Asia (with China)", "Asia (excluding China)"))

# Creating multi-faceted vertical comparison plot
p2 <- ggplot(top5_all, aes(x = reorder(Label, InducedProduction), y = InducedProduction / 1e6, fill = Region)) +
  geom_bar(stat = "identity", width = 0.7, show.legend = FALSE) +
  coord_flip() +
  facet_wrap(~Region, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = c("#4A90E2", "#E28B4A", "#4AE290", "#8D4AE2")) +
  labs(
    title = "Top 5 Induced Country-Sectors per Region",
    subtitle = "Induced by European Final Demand (Billions USD)",
    x = "Sector",
    y = "Induced Production (Billions USD)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15, margin = margin(b = 5)),
    plot.subtitle = element_text(color = "grey40", size = 10, margin = margin(b = 15)),
    strip.text = element_text(face = "bold", size = 11, hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    strip.background = element_rect(fill = "grey95", color = NA)
  )

ggsave("R/plots/top_induced_sectors_by_region.png", plot = p2, width = 9.5, height = 11, dpi = 300)
cat("- Saved R/plots/top_induced_sectors_by_region.png\n")

cat("\nAll plots generated successfully!\n")
