# Tidier.jl Integrations for Juliora

using Tidier
using DataFrames
using Juliora: AbstractMatrixEntry, MatrixEntry, SeriesEntry, LeontiefFactorization

# Helper functions for updating metadata dataframes in-place/zero-copy
function update_row_indices(m::MatrixEntry, new_row_indices)
    return MatrixEntry(m.data, m.col_indices, safe_dataframe(new_row_indices))
end

function update_col_indices(m::MatrixEntry, new_col_indices)
    return MatrixEntry(m.data, safe_dataframe(new_col_indices), m.row_indices)
end

function update_col_indices(se::SeriesEntry, new_col_indices)
    return SeriesEntry(se.data, safe_dataframe(new_col_indices))
end

function update_row_indices(m::LeontiefFactorization, new_row_indices)
    return LeontiefFactorization(m.factorization, m.col_indices, safe_dataframe(new_row_indices))
end

function update_col_indices(m::LeontiefFactorization, new_col_indices)
    return LeontiefFactorization(m.factorization, safe_dataframe(new_col_indices), m.row_indices)
end

# 2D-style indexing for SeriesEntry to support column filtering and slicing uniformly
function Base.getindex(m::SeriesEntry, ::Colon, idxs::AbstractVector{<:Integer})
    return m[idxs]
end

function Base.getindex(m::SeriesEntry, ::Colon, mask::AbstractVector{Bool})
    return m[mask]
end

# Integer indexing for AbstractMatrixEntry
function Base.getindex(m::AbstractMatrixEntry, row_idxs::AbstractVector{<:Integer}, ::Colon)
    @assert all(1 <= idx <= size(m.data, 1) for idx in row_idxs) "Row index out of bounds"
    new_data = m.data[row_idxs, :]
    new_row_indices = m.row_indices[row_idxs, :]
    return MatrixEntry(new_data, m.col_indices, new_row_indices)
end

function Base.getindex(m::AbstractMatrixEntry, ::Colon, col_idxs::AbstractVector{<:Integer})
    @assert all(1 <= idx <= size(m.data, 2) for idx in col_idxs) "Column index out of bounds"
    new_data = m.data[:, col_idxs]
    new_col_indices = m.col_indices[col_idxs, :]
    return MatrixEntry(new_data, new_col_indices, m.row_indices)
end

function Base.getindex(m::AbstractMatrixEntry, row_idxs::AbstractVector{<:Integer}, col_idxs::AbstractVector{<:Integer})
    @assert all(1 <= idx <= size(m.data, 1) for idx in row_idxs) "Row index out of bounds"
    @assert all(1 <= idx <= size(m.data, 2) for idx in col_idxs) "Column index out of bounds"
    new_data = m.data[row_idxs, col_idxs]
    new_row_indices = m.row_indices[row_idxs, :]
    new_col_indices = m.col_indices[col_idxs, :]
    return MatrixEntry(new_data, new_col_indices, new_row_indices)
end

# Macros using TidierData internally (entire block escaped to preserve AST clean for TidierData)
macro filter_rows(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.row_indices)
        df_temp.__row_id__ = 1:nrow(df_temp)
        local filtered_df = TidierData.@filter(df_temp, $(exprs...))
        local kept_rows = filtered_df.__row_id__
        m_val[kept_rows, :]
    end)
end

macro filter_cols(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.col_indices)
        df_temp.__col_id__ = 1:nrow(df_temp)
        local filtered_df = TidierData.@filter(df_temp, $(exprs...))
        local kept_cols = filtered_df.__col_id__
        m_val[:, kept_cols]
    end)
end

macro mutate_rows(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.row_indices)
        local mutated_df = TidierData.@mutate(df_temp, $(exprs...))
        update_row_indices(m_val, mutated_df)
    end)
end

macro mutate_cols(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.col_indices)
        local mutated_df = TidierData.@mutate(df_temp, $(exprs...))
        update_col_indices(m_val, mutated_df)
    end)
end

macro select_rows(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.row_indices)
        local selected_df = TidierData.@select(df_temp, $(exprs...))
        update_row_indices(m_val, selected_df)
    end)
end

macro select_cols(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.col_indices)
        local selected_df = TidierData.@select(df_temp, $(exprs...))
        update_col_indices(m_val, selected_df)
    end)
end

macro rename_rows(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.row_indices)
        local renamed_df = TidierData.@rename(df_temp, $(exprs...))
        update_row_indices(m_val, renamed_df)
    end)
end

macro rename_cols(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.col_indices)
        local renamed_df = TidierData.@rename(df_temp, $(exprs...))
        update_col_indices(m_val, renamed_df)
    end)
end

macro slice_rows(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.row_indices)
        df_temp.__row_id__ = 1:nrow(df_temp)
        local sliced_df = TidierData.@slice(df_temp, $(exprs...))
        local kept_rows = sliced_df.__row_id__
        m_val[kept_rows, :]
    end)
end

macro slice_cols(m, exprs...)
    return esc(quote
        local m_val = $m
        local df_temp = copy(m_val.col_indices)
        df_temp.__col_id__ = 1:nrow(df_temp)
        local sliced_df = TidierData.@slice(df_temp, $(exprs...))
        local kept_cols = sliced_df.__col_id__
        m_val[:, kept_cols]
    end)
end
