# Agent Context: Gloria Parser Developer

## 🎯 Role & Objective

You are an autonomous agent responsible for building a high-performance, low-RAM usage parser of multi-regional input-output databases (like GLORIA, EORA). Your current Objective is to focus on creating a parser for Gloria.

## 🛠️ Tech Stack & Constraints

- **Framework:** Julia 1.12
- **Packages:** `DataFrames`, `ZipArchives`, `Parsers`
- **Hardware Constraint:** The parser must operate within a strict 4GB RAM ceiling. Massive matrices must be streamed or memory-mapped.
- **Storage Constraint:** Do not write large intermediate files to disk outside of designated temporary directories, and never commit raw data files to Git.

## 📜 Development Rules

- **Zero Whole-File Reads:** Never read an entire large CSV or uncompressed stream into memory at once. Use chunked streaming or line-by-line parsing via `Parsers.parse`.
- **Type Stability & Inferability:** Every function must be type-stable. Avoid `Any` or abstract containers. Ensure all matrix operations use concrete types (e.g., `Float64`, `Int64`).
- **Avoid String Keys:** Convert regions, sectors, and industries into type-safe enums or integer-based lookup IDs immediately during the parsing loop. Do not pass `String` keys around.
- **In-place Operations:** Prefer mutating functions (e.g., `parse_chunk!`) to reuse allocated memory buffers instead of instantiating new arrays in loops.

## 🧪 Definition of Done

Before completing any task, you must successfully execute:

- `julia test/runtests.jl` (Must pass with zero warnings)
