from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df=df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_constant_columns():
    """Тест для проверки эвристики has_constant_columns."""
    df = pd.DataFrame(
        {
            "constant_col": [1, 1, 1, 1],
            "normal_col": [10, 20, 30, 40],
            "category": ["A", "B", "A", "B"],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df=df)

    assert flags["has_constant_columns"] is True
    assert "constant_col" in flags["constant_column_names"]
    assert "normal_col" not in flags["constant_column_names"]


def test_quality_flags_high_cardinality():
    """Тест для проверки эвристики has_high_cardinality_categoricals."""
    # Создаём DataFrame с категориальной колонкой с большим числом уникальных значений
    categories = [f"cat_{i}" for i in range(150)]  # 150 уникальных значений
    df = pd.DataFrame(
        {
            "high_cardinality_col": categories,
            "normal_col": [1, 2, 3] * 50,
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df=df, high_cardinality_threshold=100)

    assert flags["has_high_cardinality_categoricals"] is True
    assert "high_cardinality_col" in flags["high_cardinality_column_names"]


def test_quality_flags_many_zeros():
    """Тест для проверки эвристики has_many_zero_values."""
    df = pd.DataFrame(
        {
            "zero_rich_col": [0, 0, 0, 0, 0, 1, 2, 3],  # 5 из 8 = 62.5% нулей
            "normal_col": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df=df, zero_share_threshold=0.5)

    assert flags["has_many_zero_values"] is True
    assert "zero_rich_col" in flags["zero_rich_column_names"]
    assert "normal_col" not in flags["zero_rich_column_names"]
