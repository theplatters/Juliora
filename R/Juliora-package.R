#' Juliora: R Bindings for the Juliora Julia Package
#'
#' Provides high-performance multi-regional input-output (MRIO) database parsing
#' and analysis functions, wrapping the 'Juliora' Julia package via
#' 'JuliaConnectoR'.
#'
#' @section Julia project discovery:
#' On first use the package locates the Juliora Julia project via, in order:
#' `options(juliora.julia_project = ...)`, the `JULIORA_JULIA_PROJECT`
#' environment variable, the installed package directory, a walk up from the
#' working directory looking for `Project.toml`, and finally `"."`. Use
#' [juliora_reset()] to clear the cached connection state.
#'
#' @import dplyr
#' @import JuliaConnectoR
#' @importFrom stats aggregate
#' @importFrom utils head .DollarNames
#' @keywords internal
"_PACKAGE"
