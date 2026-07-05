# Juliora

Juliora is a Julia package, with R bindings, for parsing and analyzing
multi-regional input-output (MRIO) databases. It provides labeled matrix and
series containers, MRIO database constructors, environmental extensions,
aggregation tools, and helpers for tidy analysis workflows.

The package is designed for working with large economic input-output systems
where matrix values need to stay attached to country, sector, final demand,
value added, or environmental stressor metadata.

## Capabilities

- Load complete Eora and Gloria MRIO databases from local data files.
- Parse Gloria input-output and supply-use table files.
- Store labeled matrices with `MatrixEntry` and labeled vectors with
  `SeriesEntry`.
- Construct complete `MRIO` objects from transaction, final demand, and value
  added matrices.
- Compute technical coefficients, total output, and Leontief factorizations.
- Attach environmental extensions with direct impacts and impact intensities.
- Calculate environmental impacts from production vectors or production
  scenario matrices.
- Estimate induced production using the Leontief inverse for selected consumer
  and producer countries.
- Filter, subset, drop, group, and aggregate matrices by country, sector, or
  other index metadata.
- Convert matrices to and from long-form data frames for tidy analysis.
- Produce country, sector, bilateral flow, and matrix summary tables.
- Use Julia `Tidier` macros and R `dplyr` methods for metadata-oriented
  workflows.

## Core Data Model

Juliora centers on a few typed containers:

- `MatrixEntry`: a numeric matrix plus row and column metadata tables.
- `SeriesEntry`: a numeric vector plus element metadata.
- `EnvironmentalExtension`: direct environmental impacts `F` and
  environmental intensity matrix `A`.
- `MRIO`: a complete input-output database containing technical coefficients
  `A`, transactions `T`/`Z`, value added `VA`, final demand `FD`/`Y`, Leontief
  factorization `L`, total output `X`, and environmental data `env`.

Because matrix dimensions are tied to metadata dimensions, constructors validate
that data sizes match the supplied row and column indices.

## Julia Usage

Instantiate dependencies:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Load the package from this repository:

```julia
using Juliora
using DataFrames
```

Create a labeled matrix:

```julia
data = [1.0 2.0; 3.0 4.0]

indices = DataFrame(
    CountryCode = ["AUT", "DEU"],
    Sector = ["Agriculture", "Manufacturing"],
)

z = MatrixEntry(data, indices, indices)
```

Use named metadata to access values:

```julia
z[(CountryCode = "AUT", Sector = "Agriculture"),
  (CountryCode = "DEU", Sector = "Manufacturing")]
```

Load external MRIO data:

```julia
eora = Eora("data/2017/")
gloria = Gloria("data/GLORIA/", 60, 2019)

# Lower-level Gloria parsers are also available.
gloria_io = parse_gloria("data/GLORIA/", 2019; version = 60)
gloria_sut = parse_gloria_sut("data/GLORIA/", 2019; version = 60)
```

Query metadata and summarize flows:

```julia
countries(gloria)
sectors(gloria)
stressors(gloria)

sum_by_country(gloria.Z; dimension = :both)
sum_by_sector(gloria.Z; dimension = :rows)
country_summary(gloria.Z)
matrix_summary(gloria.Z)
```

Filter and aggregate matrices:

```julia
aut_rows = filter_rows(gloria.Z, row -> row.CountryCode == "AUT")
manufacturing_cols = filter_cols(gloria.Z, col -> col.Sector == "Manufacturing")

sector_totals = aggregate(groupby(gloria.Z, :Sector; dims = 1), sum)
aggregated_mrio = aggregate(gloria, [:CountryCode]; dims = 2)
```

Run environmental and Leontief-based analysis:

```julia
impact = environmental_impact(gloria, gloria.X.data)

production = induced_production(
    gloria;
    consumer_countries = ["AUT", "DEU"],
    producer_countries = ["CHN", "IND"],
)
```

Convert between matrix and tabular forms:

```julia
long = to_long_dataframe(gloria.Z; value_name = "flow")
wide = pivot_matrix_to_wide(gloria.Z, [:CountryCode], :Sector, "flow")
rebuilt = from_long_dataframe(long; value_col = "flow")
```

## R Usage

The repository also exposes R bindings through `JuliaConnectoR`. The R package
wraps Julia objects as S3 objects and provides R functions with names aligned to
the Julia API.

Run R tests:

```sh
Rscript -e 'devtools::test()'
```

Check that Julia is available to R:

```r
library(Juliora)

is_julia_available()
```

Create and inspect labeled data:

```r
idx <- data.frame(
  CountryCode = c("AUT", "DEU"),
  Sector = c("Agriculture", "Manufacturing")
)

z <- MatrixEntry(matrix(c(1, 3, 2, 4), nrow = 2), idx, idx)

matrix_summary(z)
as.data.frame(z)
```

Load and analyze MRIO data:

```r
gloria <- Gloria("data/GLORIA/", version = 60, year = 2019)

countries(gloria)
sectors(gloria)
stressors(gloria)

sum_by_country(gloria$Z, dimension = "both")
country_summary(gloria$Z)
```

Use `dplyr` verbs on metadata:

```r
library(dplyr)

filtered <- gloria$Z |>
  filter(CountryCode == "AUT", .dims = 1) |>
  mutate(region_group = "Austria", .dims = 1)
```

## Development Commands

Run the Julia test suite:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run R package tests:

```sh
Rscript -e 'devtools::test()'
```

Regenerate R documentation after roxygen changes:

```sh
Rscript -e 'devtools::document()'
```

Run the full R package check:

```sh
R CMD check .
```

Format Julia source:

```sh
runic --inplace src/*
```

## Data Requirements

Juliora expects MRIO source data to be available locally. Eora loaders look for
files such as `T.txt`, `VA.txt`, `FD.txt`, `Q.txt`, and their label files in the
provided directory. Gloria loaders expect the corresponding Gloria data release
files for the requested version and year.

Large MRIO datasets are not bundled with this repository.

## License

Juliora is licensed under the MIT license. See `LICENSE` for details.
