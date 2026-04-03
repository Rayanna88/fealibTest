"""
Integration tests for pyfealib.Fealib end-to-end pipeline.

Tests the full feature processing pipeline:
  1. Metadata   : sparse column count, dense width
  2. User-only  : zero-batch mode (items=[])
  3. Item feats : dense (star/score) + sparse (packagename, category, tokens…)
  4. Cross feats: concat_ws cross features
  5. Batch      : multiple items in one call
  6. Format feat: composite-key context features
  7. Idempotence: repeated calls are deterministic

Requires:
  - pyfealib installed
  - fealib.yaml reachable (via REPO_ROOT/test/testdata/fealib.yaml)
  - collection data directory reachable (see conftest.py COLLECTION_CANDIDATES)
  - pip install mmh3 pytest pytest-xdist

All tests auto-skip if pyfealib or the collection is not available.
"""

import pytest
import numpy as np

pytestmark = pytest.mark.integration

try:
    import pyfealib
    PYFEALIB_AVAILABLE = True
except ImportError:
    pyfealib = None
    PYFEALIB_AVAILABLE = False

from tests.conftest import (
    SparseCol, DenseCol,
    TOTAL_SPARSE_COLS, TOTAL_DENSE_COLS,
    str_hash, int32_hash, int64_hash,
    make_user_features, make_context_features,
)


# ============================================================
# Ground-truth constants for test items
# (must match the actual collection data)
# ============================================================
ITEM1 = "s_0001068bce52203a4ce39a57297a2258"
ITEM2 = "s_0003426311904e7a89be2cd4ec980dd0"

ITEM1_STAR          = 3.5
ITEM1_SCORE         = 0.0
ITEM1_DOWNLOAD      = 28493
ITEM1_EXPOSE7D      = 49
ITEM1_CLICK7D       = 0
ITEM1_TOKEN_COUNT   = 3          # "Bubble", "Fall", "3D"
ITEM1_PACKAGE       = "com.tarboosh.bubblefall3d"
ITEM1_CATEGORY      = "Game"
ITEM1_CATEGORYNAME  = "Puzzle"
ITEM1_LAN           = "en"

ITEM2_STAR          = 4.9
ITEM2_SCORE         = 4.7
ITEM2_DOWNLOAD      = 1005304
ITEM2_EXPOSE7D      = 48
ITEM2_CLICK7D       = 3
ITEM2_CATEGORYNAME  = "Role Playing"

TOKEN_SLOT_LEN      = 8          # item_tokens_hash export len
DEFAULT_MASK        = 0x1FFFFFFFFFFFFF


# ============================================================
# Helper: sparse accessor for zero-batch vs normal batch
# ============================================================
def get_sparse(sparse, col: int, batch_idx: int, zero_batch: bool):
    """
    Return the list[int64] values for (col, batch_idx).
      zero_batch=True  → sparse[col]          (no nesting)
      zero_batch=False → sparse[col][batch_idx]
    """
    if zero_batch:
        return sparse[col]
    return sparse[col][batch_idx]


# ============================================================
# Fixture aliases (re-exported from conftest for convenience)
# ============================================================
@pytest.fixture
def uf():
    return make_user_features()


@pytest.fixture
def cf():
    return make_context_features()


# ============================================================
# 1. Metadata
# ============================================================
class TestMetadata:
    """Verify output shape metadata: column counts."""

    def test_sparse_column_count(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [])
        assert len(result["sparse"]) == TOTAL_SPARSE_COLS, (
            f"Expected {TOTAL_SPARSE_COLS} sparse cols, "
            f"got {len(result['sparse'])}"
        )

    def test_dense_column_count(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [])
        assert result["dense"].shape[1] == TOTAL_DENSE_COLS

    def test_result_has_sparse_key(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [])
        assert "sparse" in result

    def test_result_has_dense_key(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [])
        assert "dense" in result


# ============================================================
# 2. Zero-batch mode (items=[])
# ============================================================
class TestZeroBatch:
    """Tests for user-only run: items=[] → zero-batch."""

    def test_dense_shape_is_1xN(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [])
        assert result["dense"].shape == (1, TOTAL_DENSE_COLS)

    def test_user_id_sparse_nonempty(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [])
        assert len(result["sparse"][SparseCol.USER_ID]) > 0

    def test_user_country_sparse_nonempty(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [])
        assert len(result["sparse"][SparseCol.USER_COUNTRY]) > 0

    def test_item_cols_empty_in_zero_batch(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [])
        sparse = result["sparse"]
        for col in range(SparseCol.PACKAGENAME, TOTAL_SPARSE_COLS):
            assert len(sparse[col]) == 0, (
                f"col {col} should be empty when items=[]"
            )

    def test_user_install_count_dense(self, fealib_handler):
        result = fealib_handler.run(
            make_user_features(install_cnt=7.0), {}, []
        )
        assert abs(result["dense"][0, DenseCol.INSTALL_CNT] - 7.0) < 1e-5

    def test_user_id_hash_value(self, fealib_handler):
        result = fealib_handler.run(make_user_features(uid=123456), {}, [])
        got = result["sparse"][SparseCol.USER_ID][0]
        expected = int64_hash("uid_", 123456, 0xFFFFF)
        assert got == expected

    def test_user_country_hash_value(self, fealib_handler):
        result = fealib_handler.run(make_user_features(country="us"), {}, [])
        got = result["sparse"][SparseCol.USER_COUNTRY][0]
        expected = str_hash("co_", "us", 0xFFFF)
        assert got == expected


# ============================================================
# 3. Single item — dense features
# ============================================================
class TestItemDenseFeatures:
    """Tests for per-item dense output (star, score)."""

    def test_item1_star(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        assert abs(result["dense"][0, DenseCol.STAR] - ITEM1_STAR) < 1e-4

    def test_item1_score(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        assert abs(result["dense"][0, DenseCol.SCORE] - ITEM1_SCORE) < 1e-4

    def test_item2_star(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM2])
        assert abs(result["dense"][0, DenseCol.STAR] - ITEM2_STAR) < 0.01

    def test_item2_score(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM2])
        assert abs(result["dense"][0, DenseCol.SCORE] - ITEM2_SCORE) < 1e-4


# ============================================================
# 4. Single item — sparse features
# ============================================================
class TestItemSparseFeatures:
    """Tests for per-item sparse (hashed) output."""

    def test_item1_download_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        got = result["sparse"][SparseCol.DOWNLOAD][0][0]
        expected = int32_hash("i_dlc_", ITEM1_DOWNLOAD, DEFAULT_MASK)
        assert got == expected

    def test_item1_expose7d_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        got = result["sparse"][SparseCol.EXPOSE7D][0][0]
        expected = int32_hash("i_euv7d_", ITEM1_EXPOSE7D, DEFAULT_MASK)
        assert got == expected

    def test_item1_click7d_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        got = result["sparse"][SparseCol.CLICK7D][0][0]
        expected = int32_hash("i_cuv7d_", ITEM1_CLICK7D, DEFAULT_MASK)
        assert got == expected

    def test_item1_packagename_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        got = result["sparse"][SparseCol.PACKAGENAME][0][0]
        expected = str_hash("pkg_", ITEM1_PACKAGE, 0xFFFFF)
        assert got == expected

    def test_item1_category_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        got = result["sparse"][SparseCol.CATEGORY][0][0]
        expected = str_hash("cat_", ITEM1_CATEGORY, 0xFFFF)
        assert got == expected

    def test_item1_lan_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        got = result["sparse"][SparseCol.LAN][0][0]
        expected = str_hash("lan_", ITEM1_LAN, 0xFFFF)
        assert got == expected

    def test_item1_tokens_length_is_8(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        tokens = result["sparse"][SparseCol.TOKENS][0]
        assert len(tokens) == TOKEN_SLOT_LEN

    def test_item1_tokens_padding_slots_are_zero(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        tokens = result["sparse"][SparseCol.TOKENS][0]
        for i in range(ITEM1_TOKEN_COUNT, TOKEN_SLOT_LEN):
            assert tokens[i] == 0, f"padding slot [{i}] must be 0"

    def test_item1_tokens_nonpadding_slots_nonzero(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        tokens = result["sparse"][SparseCol.TOKENS][0]
        nonzero = sum(1 for t in tokens[:ITEM1_TOKEN_COUNT] if t != 0)
        assert nonzero == ITEM1_TOKEN_COUNT

    def test_item1_first_token_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        tokens = result["sparse"][SparseCol.TOKENS][0]
        expected = str_hash("tok_", "Bubble", 0xFFFFF)
        assert tokens[0] == expected

    def test_item2_download_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM2])
        got = result["sparse"][SparseCol.DOWNLOAD][0][0]
        expected = int32_hash("i_dlc_", ITEM2_DOWNLOAD, DEFAULT_MASK)
        assert got == expected

    def test_item2_click7d_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM2])
        got = result["sparse"][SparseCol.CLICK7D][0][0]
        expected = int32_hash("i_cuv7d_", ITEM2_CLICK7D, DEFAULT_MASK)
        assert got == expected

    def test_item2_categoryname_hash(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM2])
        got = result["sparse"][SparseCol.CATEGORYNAME][0][0]
        expected = str_hash("catn_", ITEM2_CATEGORYNAME, 0xFFFF)
        assert got == expected


# ============================================================
# 5. Cross features (concat_ws)
# ============================================================
class TestCrossFeatures:
    """Tests for user × item cross features via concat_ws op."""

    def test_cross_country_cat_item1(self, fealib_handler):
        result = fealib_handler.run(make_user_features(country="us"), {}, [ITEM1])
        got = result["sparse"][SparseCol.CROSS_COUNTRY_CAT][0][0]
        # concat_ws("@", country="us", categoryname="Puzzle") → "us@Puzzle"
        expected = str_hash("co_cat_", "us@Puzzle", 0x1FFFF)
        assert got == expected

    def test_cross_country_lan_item1(self, fealib_handler):
        result = fealib_handler.run(make_user_features(country="us"), {}, [ITEM1])
        got = result["sparse"][SparseCol.CROSS_COUNTRY_LAN][0][0]
        # concat_ws("@", country="us", lan="en") → "us@en"
        expected = str_hash("co_lan_", "us@en", 0xFFFF)
        assert got == expected

    def test_cross_country_cat_item2(self, fealib_handler):
        result = fealib_handler.run(make_user_features(country="us"), {}, [ITEM2])
        got = result["sparse"][SparseCol.CROSS_COUNTRY_CAT][0][0]
        expected = str_hash("co_cat_", f"us@{ITEM2_CATEGORYNAME}", 0x1FFFF)
        assert got == expected

    def test_cross_values_differ_between_items(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1, ITEM2])
        sparse = result["sparse"]
        val1 = sparse[SparseCol.CROSS_COUNTRY_CAT][0][0]
        val2 = sparse[SparseCol.CROSS_COUNTRY_CAT][1][0]
        assert val1 != val2, "Different items must produce different cross feature values"

    def test_cross_features_nonempty_on_item_row(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        assert len(result["sparse"][SparseCol.CROSS_COUNTRY_CAT][0]) > 0
        assert len(result["sparse"][SparseCol.CROSS_COUNTRY_LAN][0]) > 0


# ============================================================
# 6. Batch of multiple items
# ============================================================
class TestBatchProcessing:
    """Tests for batch inference with multiple items."""

    def test_batch2_dense_shape(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1, ITEM2])
        assert result["dense"].shape == (2, TOTAL_DENSE_COLS)

    def test_batch2_item1_star(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1, ITEM2])
        assert abs(result["dense"][0, DenseCol.STAR] - ITEM1_STAR) < 1e-4

    def test_batch2_item2_star(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1, ITEM2])
        assert abs(result["dense"][1, DenseCol.STAR] - ITEM2_STAR) < 0.01

    def test_user_features_broadcast_across_items(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1, ITEM2])
        sparse = result["sparse"]
        # user_id same hash for all items in batch
        assert sparse[SparseCol.USER_ID][0] == sparse[SparseCol.USER_ID][1]
        # user_country same hash for all items in batch
        assert sparse[SparseCol.USER_COUNTRY][0] == sparse[SparseCol.USER_COUNTRY][1]

    def test_sparse_col_count_unchanged_in_batch(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1, ITEM2])
        assert len(result["sparse"]) == TOTAL_SPARSE_COLS


# ============================================================
# 7. Format (context-composite) features
# ============================================================
class TestFormatFeatures:
    """Tests for context.country composite-key features (slot 50)."""

    def test_format_feature_absent_without_context(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        assert len(result["sparse"][SparseCol.EXPOSE7D_BY_COUNTRY][0]) == 0, (
            "format feature must be absent when context is not provided"
        )

    def test_format_feature_with_context_does_not_crash(self, fealib_handler):
        result = fealib_handler.run(
            make_user_features(), make_context_features("us"), [ITEM1]
        )
        assert result is not None
        assert "sparse" in result
        assert "dense" in result

    def test_nonempty_item_cols_count_without_context(self, fealib_handler):
        result = fealib_handler.run(make_user_features(), {}, [ITEM1])
        sparse = result["sparse"]
        # col 11 (format) absent → expect 11 non-empty item cols out of [2..13]
        nonempty = sum(
            1 for col in range(SparseCol.PACKAGENAME, TOTAL_SPARSE_COLS)
            if len(sparse[col][0]) > 0
        )
        assert nonempty == 11

    def test_nonempty_item_cols_count_with_context(self, fealib_handler):
        result = fealib_handler.run(
            make_user_features(), make_context_features("us"), [ITEM1]
        )
        sparse = result["sparse"]
        nonempty = sum(
            1 for col in range(SparseCol.PACKAGENAME, TOTAL_SPARSE_COLS)
            if len(sparse[col][0]) > 0
        )
        # 12 if collection has the composite key, 11 if absent
        assert nonempty in (11, 12)


# ============================================================
# 8. Idempotence / stability
# ============================================================
class TestIdempotence:
    """Tests for determinism and stability of repeated calls."""

    def test_repeated_calls_same_sparse(self, fealib_handler):
        r1 = fealib_handler.run(make_user_features(), {}, [ITEM1])
        r2 = fealib_handler.run(make_user_features(), {}, [ITEM1])
        assert (
            r1["sparse"][SparseCol.DOWNLOAD][0][0]
            == r2["sparse"][SparseCol.DOWNLOAD][0][0]
        )

    def test_repeated_calls_same_dense(self, fealib_handler):
        r1 = fealib_handler.run(make_user_features(), {}, [ITEM1])
        r2 = fealib_handler.run(make_user_features(), {}, [ITEM1])
        np.testing.assert_array_equal(r1["dense"], r2["dense"])

    def test_many_calls_do_not_crash(self, fealib_handler):
        for _ in range(20):
            result = fealib_handler.run(make_user_features(), {}, [ITEM1, ITEM2])
            assert result is not None

    def test_empty_items_call_stable(self, fealib_handler):
        for _ in range(5):
            result = fealib_handler.run(make_user_features(), {}, [])
            assert result["dense"].shape == (1, TOTAL_DENSE_COLS)
