# Repository Guidelines

## Project Structure & Module Organization

Juliora is a Julia package with R bindings. Julia source lives in `src/`, with the main module in `src/Juliora.jl` and domain files such as `matrixentry.jl`, `seriesentry.jl`, `mrio.jl`, `analysis.jl`, and `aggregation.jl`. Parser code is under `src/parsers/`. Julia tests are in `test/`, with `test/runtests.jl` including the individual `test_*.jl` files.

R bindings live in `R/`, integration scripts in `R/scripts/`, testthat tests in `tests/testthat/`, and generated Rd documentation in `man/`. Keep generated documentation consistent with roxygen comments in `R/*.R`.

## Build, Test, and Development Commands

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'`: install Julia dependencies from `Project.toml` and `Manifest.toml`.
- `julia --project=. -e 'using Pkg; Pkg.test()'`: run the Julia test suite via `test/runtests.jl`.
- `Rscript -e 'devtools::test()'`: run R `testthat` tests.
- `Rscript -e 'devtools::document()'`: regenerate `man/` files after roxygen changes.
- `R CMD check .`: run the full R package check, including documentation and examples.

## Coding Style & Naming Conventions

Use four-space indentation for Julia and two-space indentation for R. Julia functions and variables generally use `snake_case`; types and structs use `PascalCase` such as `MatrixEntry`, `SeriesEntry`, and `MRIO`. R wrappers should keep user-facing names aligned with the Julia API where practical, and exported R functions need roxygen blocks with `@export`.

Use `runic --inplace src/*` for formatting Julia code.

Prefer small, typed Julia methods and clear R validation before calling Julia through `JuliaConnectoR`. Avoid broad refactors when touching parser or binding code; keep changes close to the feature or bug being addressed.

## Testing Guidelines

Add Julia tests beside related coverage in `test/test_*.jl`, then include new files from `test/runtests.jl` if needed. Add R binding tests in `tests/testthat/test-*.R`. Cover both successful calls and validation or error paths, especially for wrappers that cross the R-Julia boundary.

## Commit & Pull Request Guidelines

Git history uses short, direct commit subjects such as `Aggregation fixes`, `R bindings`, and `performance improvements and simplifications`. Keep commit messages imperative or descriptive, and mention the affected area when helpful.

Pull requests should include a concise summary, the tests run, and any data or Julia/R environment assumptions. Link related issues when available. Include screenshots only for visual output or plots.

## Agent-Specific Instructions

Do not overwrite `AGENTS.md` if it already exists. Before changing generated files in `man/` or test artifacts under `tests/testthat/_problems/`, verify they are expected outputs of the current change.
