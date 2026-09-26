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

"""
    _check_sentinel_absent(df::DataFrame, name::String)

Internal helper used by the Tidier macros: refuse to inject a bookkeeping
`__row_id__`/`__col_id__` column when the user's index frame already
contains one, instead of silently clobbering user data.
"""
function _check_sentinel_absent(df::DataFrame, name::String)
    if name in names(df)
        throw(ArgumentError("index frame already contains a column named \"$name\"; rename it before using the Tidier macros"))
    end
    return true
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
    if !all(1 <= idx <= size(m.data, 1) for idx in row_idxs)
        throw(BoundsError(m, row_idxs))
    end
    new_data = m.data[row_idxs, :]
    new_row_indices = m.row_indices[row_idxs, :]
    return MatrixEntry(new_data, m.col_indices, new_row_indices)
end

function Base.getindex(m::AbstractMatrixEntry, ::Colon, col_idxs::AbstractVector{<:Integer})
    if !all(1 <= idx <= size(m.data, 2) for idx in col_idxs)
        throw(BoundsError(m, col_idxs))
    end
    new_data = m.data[:, col_idxs]
    new_col_indices = m.col_indices[col_idxs, :]
    return MatrixEntry(new_data, new_col_indices, m.row_indices)
end

function Base.getindex(m::AbstractMatrixEntry, row_idxs::AbstractVector{<:Integer}, col_idxs::AbstractVector{<:Integer})
    if !all(1 <= idx <= size(m.data, 1) for idx in row_idxs)
        throw(BoundsError(m, row_idxs))
    end
    if !all(1 <= idx <= size(m.data, 2) for idx in col_idxs)
        throw(BoundsError(m, col_idxs))
    end
    new_data = m.data[row_idxs, col_idxs]
    new_row_indices = m.row_indices[row_idxs, :]
    new_col_indices = m.col_indices[col_idxs, :]
    return MatrixEntry(new_data, new_col_indices, new_row_indices)
end

# Macros using TidierData internally.
#
# Hygiene: the expanded block is escaped (as before) so that the entry
# expression and the TidierData column expressions evaluate in caller scope,
# but every helper is referenced through a GlobalRef to its defining module
# (Base.copy, DataFrames.nrow, Juliora.update_row_indices/update_col_indices,
# Juliora._check_sentinel_absent) and the TidierData macros are invoked
# through GlobalRef heads. Callers therefore do NOT need `using Tidier`,
# `TidierData`, `DataFrames`, `copy`, or `nrow` in scope.
macro filter_rows(m, exprs...)
    td_filter = Expr(:macrocall, GlobalRef(TidierData, Symbol("@filter")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.row_indices)
            $(GlobalRef(Juliora, :_check_sentinel_absent))(df_temp, "__row_id__")
            df_temp.__row_id__ = 1:$(GlobalRef(DataFrames, :nrow))(df_temp)
            local filtered_df = $(td_filter)
            local kept_rows = filtered_df.__row_id__
            m_val[kept_rows, :]
        end
    )
end

macro filter_cols(m, exprs...)
    td_filter = Expr(:macrocall, GlobalRef(TidierData, Symbol("@filter")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.col_indices)
            $(GlobalRef(Juliora, :_check_sentinel_absent))(df_temp, "__col_id__")
            df_temp.__col_id__ = 1:$(GlobalRef(DataFrames, :nrow))(df_temp)
            local filtered_df = $(td_filter)
            local kept_cols = filtered_df.__col_id__
            m_val[:, kept_cols]
        end
    )
end

macro mutate_rows(m, exprs...)
    td_mutate = Expr(:macrocall, GlobalRef(TidierData, Symbol("@mutate")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.row_indices)
            local mutated_df = $(td_mutate)
            $(GlobalRef(Juliora, :update_row_indices))(m_val, mutated_df)
        end
    )
end

macro mutate_cols(m, exprs...)
    td_mutate = Expr(:macrocall, GlobalRef(TidierData, Symbol("@mutate")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.col_indices)
            local mutated_df = $(td_mutate)
            $(GlobalRef(Juliora, :update_col_indices))(m_val, mutated_df)
        end
    )
end

macro select_rows(m, exprs...)
    td_select = Expr(:macrocall, GlobalRef(TidierData, Symbol("@select")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.row_indices)
            local selected_df = $(td_select)
            $(GlobalRef(Juliora, :update_row_indices))(m_val, selected_df)
        end
    )
end

macro select_cols(m, exprs...)
    td_select = Expr(:macrocall, GlobalRef(TidierData, Symbol("@select")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.col_indices)
            local selected_df = $(td_select)
            $(GlobalRef(Juliora, :update_col_indices))(m_val, selected_df)
        end
    )
end

macro rename_rows(m, exprs...)
    td_rename = Expr(:macrocall, GlobalRef(TidierData, Symbol("@rename")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.row_indices)
            local renamed_df = $(td_rename)
            $(GlobalRef(Juliora, :update_row_indices))(m_val, renamed_df)
        end
    )
end

macro rename_cols(m, exprs...)
    td_rename = Expr(:macrocall, GlobalRef(TidierData, Symbol("@rename")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.col_indices)
            local renamed_df = $(td_rename)
            $(GlobalRef(Juliora, :update_col_indices))(m_val, renamed_df)
        end
    )
end

macro slice_rows(m, exprs...)
    td_slice = Expr(:macrocall, GlobalRef(TidierData, Symbol("@slice")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.row_indices)
            $(GlobalRef(Juliora, :_check_sentinel_absent))(df_temp, "__row_id__")
            df_temp.__row_id__ = 1:$(GlobalRef(DataFrames, :nrow))(df_temp)
            local sliced_df = $(td_slice)
            local kept_rows = sliced_df.__row_id__
            m_val[kept_rows, :]
        end
    )
end

macro slice_cols(m, exprs...)
    td_slice = Expr(:macrocall, GlobalRef(TidierData, Symbol("@slice")), __source__, :df_temp, exprs...)
    return esc(
        quote
            local m_val = $m
            local df_temp = $(GlobalRef(Base, :copy))(m_val.col_indices)
            $(GlobalRef(Juliora, :_check_sentinel_absent))(df_temp, "__col_id__")
            df_temp.__col_id__ = 1:$(GlobalRef(DataFrames, :nrow))(df_temp)
            local sliced_df = $(td_slice)
            local kept_cols = sliced_df.__col_id__
            m_val[:, kept_cols]
        end
    )
end
