# Skills & Standards: R-Julia Binding Developer

## 1. Objective & Mindset

You are an autonomous systems engineer tasked with building performant, idiomatic R bindings for the `Juliora` Julia package. Your goal is to make the underlying Julia logic feel like a native, seamless R experience. You prioritize type safety, efficient memory transfer across the language boundary, and clear user-facing documentation.

---

## 2. Cross-Language Bridge Architecture

- **Bridge Dependency:** All communication with the Julia backend must be routed through the `[ e.g., JuliaConnectoR / reticulate / Custom C API ]` package.
- **Initialization:** The Julia environment must be initialized dynamically when the package is loaded via `.onLoad()`. Never hardcode absolute system paths to the Julia executable.
- **Session Persistence:** Reuse a single persistent Julia session across the R lifetime rather than spinning up processes on every function call.

---

## 3. Style & Design Guidelines

### Code Formatting

- Use the runic formater to format julia code. To use it call `runic -i src/**.jl`
- Use the air formater to format R code

### Memory & Data Transfer

- **Copying vs. Referencing:** When passing large datasets across the boundary, prefer `zero-copy pointers` to prevent excessive memory overhead.
- **Type Constraints:** Explicitly enforce R input types _before_ sending them across the bridge. If an R function expects a matrix, validate it using `is.matrix()` early to avoid opaque errors downstream in Julia.

---

## 4. Type Mapping & Conversion Matrix

When moving data between environments, adhere to the following strict mapping conventions:

| R Object Type        | Target Julia Type   | Handling Notes                                         |
| :------------------- | :------------------ | :----------------------------------------------------- |
| `numeric` (Vector)   | `Vector{Float64}`   |                                                        |
| `integer` (Vector)   | `Vector{Int64}`     | Watch for R 32-bit vs Julia 64-bit defaults            |
| `logical` (Vector)   | `Vector{Bool}`      |                                                        |
| `character` (Vector) | `Vector{String}`    |                                                        |
| `matrix`             | `Matrix` or `Array` | Account for column-major alignment in both languages   |
| `data.frame`         | `DataFrames`        | Ensure column names and types map with 100% fidelity   |
| `NA` / `NaN`         | `NaN`               | Prevent conversions from dropping or skipping elements |
| `NULL` / `list()`    | `nothing`           |                                                        |

---

## 5. Documentation Standards

Every user-facing function must be completely documented using **roxygen2** syntax directly above the function definition.

- **Required Tags:** Every export must include `@title`, `@description`, `@param` (for all arguments), `@return` (explaining the structure of the output), and `@export`.
- **Examples:** Include a executable `@examples` block showing standard usage. Wrap examples in `\dontrun{}` if they require an external system dependency or a highly specific Julia environment setup to prevent CRAN check failures.

---

## 6. Implementation Template

When creating a new R binding, use the following code structure:

```r
#' [Short, Clear Title of the Function]
#'
#' [Detailed description of what the Julia backend function achieves.]
#'
#' @param x A numeric vector representing [description].
#' @param options A list of configuration parameters, matching Julia's NamedTuple keys.
#'
#' @return A native R [type, e.g., data.frame] containing the processed results.
#' @export
#'
#' @examples
#' \dontrun{
#' result <- juliora_feature_function(c(1.1, 2.2, 3.3))
#' }
juliora_feature_function <- function(x, options = list()) {
  # 1. Input Validation
  if (!is.numeric(x)) {
    stop("Argument 'x' must be a numeric vector.", call. = FALSE)
  }

  # 2. Type/Structure Preparation
  # [ Insert any custom conversion steps needed for options/lists here ]

  # 3. Cross-Language Bridge Call
  # Safely evaluate inside the designated Julia bridge evaluation context
  result <- tryCatch({
    # [ Syntax placeholder for executing the Julia function, e.g., juliaCall("Juliora.core_func", x) ]
```
