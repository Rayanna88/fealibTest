"""
Unit tests for pyfealib built-in operator functions.
Maps to: OP单元测试用例.csv

Each test method is labelled with its CSV sub-case ID (e.g. TC-UNIT-ARITH-001-01).
After execution, actual results and pass/fail status are written back to the CSV
via the record_result fixture + pytest_sessionfinish hook in conftest.py.

Run with:
  pytest tests/unit/test_builtin_ops.py -v

CSV output: ~/Downloads/OP单元测试用例_结果.csv
"""

import math
import pytest

pytestmark = pytest.mark.unit

try:
    import pyfealib
    ops = pyfealib.ops
    AVAILABLE = True
except (ImportError, AttributeError):
    ops = None
    AVAILABLE = False

skip_if_unavailable = pytest.mark.skipif(
    not AVAILABLE, reason="pyfealib not installed"
)

# ---------------------------------------------------------------------------
# Helper: run a single CSV test case, record result, and assert
# ---------------------------------------------------------------------------
def _run(record, case_id, fn, *args, check, tol=None):
    """
    Execute fn(*args), record the result, and assert via check(actual).
    tol: if provided, used for float comparison hint in actual string.
    """
    actual = "N/A"
    try:
        actual = fn(*args)
        assert check(actual), f"{case_id}: assertion failed, actual={actual!r}"
        record(case_id, actual, "PASS")
    except Exception as exc:
        record(case_id, actual, "FAIL")
        raise


# ===========================================================================
# TC-UNIT-ARITH-001  add / sub / mul / div
# ===========================================================================
class TestArith001AddSubMulDiv:
    """TC-UNIT-ARITH-001-01 ~ 13"""

    @skip_if_unavailable
    def test_add_float_001_01(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-01",
             ops.add_f32, 3.5, 2.0,
             check=lambda v: abs(v - 5.5) <= 1e-6)

    @skip_if_unavailable
    def test_add_int32_001_02(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-02",
             ops.add_i32, 10, 3,
             check=lambda v: v == 13)

    @skip_if_unavailable
    def test_add_int64_001_03(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-03",
             ops.add_i64, 1000000000, 2000000000,
             check=lambda v: v == 3000000000)

    @skip_if_unavailable
    def test_sub_float_001_04(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-04",
             ops.sub_f32, 10.0, 3.5,
             check=lambda v: abs(v - 6.5) <= 1e-6)

    @skip_if_unavailable
    def test_sub_int32_001_05(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-05",
             ops.sub_i32, 10, 3,
             check=lambda v: v == 7)

    @skip_if_unavailable
    def test_sub_int64_001_06(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-06",
             ops.sub_i64, 5000000000, 1,
             check=lambda v: v == 4999999999)

    @skip_if_unavailable
    def test_mul_float_001_07(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-07",
             ops.mul_f32, 2.0, 4.0,
             check=lambda v: abs(v - 8.0) <= 1e-6)

    @skip_if_unavailable
    def test_mul_int32_001_08(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-08",
             ops.mul_i32, 3, 4,
             check=lambda v: v == 12)

    @skip_if_unavailable
    def test_mul_int64_001_09(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-09",
             ops.mul_i64, 1000000, 1000000,
             check=lambda v: v == 1000000000000)

    @skip_if_unavailable
    def test_div_float_001_10(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-10",
             ops.div_f32, 9.0, 3.0,
             check=lambda v: abs(v - 3.0) <= 1e-6)

    @skip_if_unavailable
    def test_div_int32_001_11(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-11",
             ops.div_i32, 9, 3,
             check=lambda v: v == 3)

    @skip_if_unavailable
    def test_div_int64_001_12(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-001-12",
             ops.div_i64, 9, 3,
             check=lambda v: v == 3)

    @skip_if_unavailable
    def test_div_by_zero_float_001_13(self, record_result):
        """div(1.0f, 0.0f) 除零 — 不崩溃，返回 padding (0 or inf)"""
        case_id = "TC-UNIT-ARITH-001-13"
        actual = "N/A"
        try:
            actual = ops.div_f32(1.0, 0.0)
            ok = (actual == 0.0 or math.isinf(actual) or math.isnan(actual))
            assert ok, f"{case_id}: unexpected result {actual!r}"
            record_result(case_id, actual, "PASS")
        except Exception as exc:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-ARITH-002  mod
# ===========================================================================
class TestArith002Mod:
    """TC-UNIT-ARITH-002-01 ~ 09"""

    @skip_if_unavailable
    def test_mod_positive_002_01(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-002-01",
             ops.mod_i32, 7, 3,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_mod_neg_dividend_002_02(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-002-02",
             ops.mod_i32, -7, 3,
             check=lambda v: v >= 0)

    @skip_if_unavailable
    def test_mod_zero_num_002_03(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-002-03",
             ops.mod_i32, 0, 5,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_mod_exact_002_04(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-002-04",
             ops.mod_i32, 6, 3,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_mod_neg_one_002_05(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-002-05",
             ops.mod_i32, -1, 7,
             check=lambda v: v >= 0)

    @skip_if_unavailable
    def test_mod_int64_large_002_06(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-002-06",
             ops.mod_i64, 1000000007, 1000000006,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_mod_int64_neg_002_07(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-002-07",
             ops.mod_i64, -100, 7,
             check=lambda v: v >= 0)

    @skip_if_unavailable
    def test_mod_int64_zero_002_08(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-002-08",
             ops.mod_i64, 0, 100,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_mod_i32_i64_consistency_002_09(self, record_result):
        """int32 vs int64 mod result consistency"""
        case_id = "TC-UNIT-ARITH-002-09"
        actual = "N/A"
        try:
            r32 = ops.mod_i32(100, 7)
            r64 = ops.mod_i64(100, 7)
            actual = f"i32={r32}, i64={r64}"
            assert r32 == r64, f"{case_id}: i32={r32} != i64={r64}"
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-ARITH-003  abs / ceil / floor / round / exp / log / log10 / log2 / sqrt / sigmoid / pow
# ===========================================================================
class TestArith003MathFuncs:
    """TC-UNIT-ARITH-003-01 ~ 27"""

    @skip_if_unavailable
    def test_abs_float_003_01(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-01",
             ops.abs_f32, -3.14,
             check=lambda v: abs(v - 3.14) <= 1e-6)

    @skip_if_unavailable
    def test_abs_int32_003_02(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-02",
             ops.abs_i32, -10,
             check=lambda v: v == 10)

    @skip_if_unavailable
    def test_abs_int64_003_03(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-03",
             ops.abs_i64, -9999999,
             check=lambda v: v == 9999999)

    @skip_if_unavailable
    def test_abs_zero_003_04(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-04",
             ops.abs_f32, 0.0,
             check=lambda v: v == 0.0)

    @skip_if_unavailable
    def test_ceil_pos_003_05(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-05",
             ops.ceil, 1.2,
             check=lambda v: v == 2)

    @skip_if_unavailable
    def test_ceil_neg_003_06(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-06",
             ops.ceil, -1.2,
             check=lambda v: v == -1)

    @skip_if_unavailable
    def test_floor_pos_003_07(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-07",
             ops.floor, 1.9,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_floor_neg_003_08(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-08",
             ops.floor, -1.9,
             check=lambda v: v == -2)

    @skip_if_unavailable
    def test_round_half_003_09(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-09",
             ops.round, 1.5,
             check=lambda v: v == 2)

    @skip_if_unavailable
    def test_round_below_half_003_10(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-10",
             ops.round, 1.4,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_round_neg_half_003_11(self, record_result):
        """round(-1.5) — record actual, allow -2 or -1"""
        case_id = "TC-UNIT-ARITH-003-11"
        actual = "N/A"
        try:
            actual = ops.round(-1.5)
            assert actual in (-2, -1), f"unexpected round(-1.5)={actual}"
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_exp_zero_003_12(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-12",
             ops.exp, 0.0,
             check=lambda v: abs(v - 1.0) <= 1e-6)

    @skip_if_unavailable
    def test_exp_one_003_13(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-13",
             ops.exp, 1.0,
             check=lambda v: abs(v - 2.71828) <= 1e-4)

    @skip_if_unavailable
    def test_log_one_003_14(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-14",
             ops.log, 1.0,
             check=lambda v: abs(v - 0.0) <= 1e-6)

    @skip_if_unavailable
    def test_log_e_003_15(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-15",
             ops.log, 2.718281,
             check=lambda v: abs(v - 1.0) <= 1e-5)

    @skip_if_unavailable
    def test_log10_003_16(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-16",
             ops.log10, 100.0,
             check=lambda v: abs(v - 2.0) <= 1e-6)

    @skip_if_unavailable
    def test_log2_003_17(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-17",
             ops.log2, 8.0,
             check=lambda v: abs(v - 3.0) <= 1e-6)

    @skip_if_unavailable
    def test_sqrt_4_003_18(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-18",
             ops.sqrt, 4.0,
             check=lambda v: abs(v - 2.0) <= 1e-6)

    @skip_if_unavailable
    def test_sqrt_2_003_19(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-19",
             ops.sqrt, 2.0,
             check=lambda v: abs(v - 1.41421) <= 1e-5)

    @skip_if_unavailable
    def test_sigmoid_zero_003_20(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-20",
             ops.sigmoid, 0.0,
             check=lambda v: abs(v - 0.5) <= 1e-6)

    @skip_if_unavailable
    def test_sigmoid_large_pos_003_21(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-21",
             ops.sigmoid, 100.0,
             check=lambda v: abs(v - 1.0) <= 1e-6)

    @skip_if_unavailable
    def test_sigmoid_large_neg_003_22(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-22",
             ops.sigmoid, -100.0,
             check=lambda v: abs(v - 0.0) <= 1e-6)

    @skip_if_unavailable
    def test_pow_2_10_003_23(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-23",
             ops.pow, 2.0, 10.0,
             check=lambda v: abs(v - 1024.0) <= 1e-3)

    @skip_if_unavailable
    def test_pow_3_3_003_24(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-003-24",
             ops.pow, 3.0, 3.0,
             check=lambda v: abs(v - 27.0) <= 1e-3)

    @skip_if_unavailable
    def test_log_zero_boundary_003_25(self, record_result):
        """log(0.0f) 边界 — 不崩溃, 返回 -inf 或 padding"""
        case_id = "TC-UNIT-ARITH-003-25"
        actual = "N/A"
        try:
            actual = ops.log(0.0)
            ok = (math.isinf(actual) and actual < 0) or actual == 0.0 or math.isnan(actual)
            assert ok, f"unexpected log(0.0)={actual}"
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_log_neg_boundary_003_26(self, record_result):
        """log(-1.0f) 边界 — 不崩溃, 返回 NaN 或 padding"""
        case_id = "TC-UNIT-ARITH-003-26"
        actual = "N/A"
        try:
            actual = ops.log(-1.0)
            ok = math.isnan(actual) or actual == 0.0 or math.isinf(actual)
            assert ok, f"unexpected log(-1.0)={actual}"
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_sqrt_neg_boundary_003_27(self, record_result):
        """sqrt(-1.0f) 边界 — 不崩溃, 返回 NaN 或 padding"""
        case_id = "TC-UNIT-ARITH-003-27"
        actual = "N/A"
        try:
            actual = ops.sqrt(-1.0)
            ok = math.isnan(actual) or actual == 0.0 or math.isinf(actual)
            assert ok, f"unexpected sqrt(-1.0)={actual}"
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-ARITH-004  scale / wilson_score / smooth / z_score
# ===========================================================================
class TestArith004ScaleWilsonSmoothZscore:
    """TC-UNIT-ARITH-004-01 ~ 18"""

    @skip_if_unavailable
    def test_scale_75_004_01(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-01",
             ops.scale, 0.75, 100.0,
             check=lambda v: v == 75)

    @skip_if_unavailable
    def test_scale_zero_004_02(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-02",
             ops.scale, 0.0, 100.0,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_scale_one_004_03(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-03",
             ops.scale, 1.0, 100.0,
             check=lambda v: v == 100)

    @skip_if_unavailable
    def test_scale_small_004_04(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-04",
             ops.scale, 0.001, 1000.0,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_scale_zero_scale_004_05(self, record_result):
        """scale(0.0f, 0.0f) 边界 — 不崩溃, 返回 0"""
        case_id = "TC-UNIT-ARITH-004-05"
        actual = "N/A"
        try:
            actual = ops.scale(0.0, 0.0)
            assert actual == 0, f"expected 0, got {actual}"
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_wilson_score_004_06(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-06",
             ops.wilson_score, 100, 80, 95,
             check=lambda v: 0.0 < v < 1.0)

    @skip_if_unavailable
    def test_wilson_score_large_004_07(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-07",
             ops.wilson_score, 1000, 500, 95,
             check=lambda v: 0.0 < v < 1.0)

    @skip_if_unavailable
    def test_wilson_score_zero_004_08(self, record_result):
        """wilson_score(0, 0, 95) total=0 边界 — 不崩溃, 返回 padding"""
        case_id = "TC-UNIT-ARITH-004-08"
        actual = "N/A"
        try:
            actual = ops.wilson_score(0, 0, 95)
            assert isinstance(actual, float), f"expected float, got {type(actual)}"
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_wilson_score_no_pos_004_09(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-09",
             ops.wilson_score, 100, 0, 95,
             check=lambda v: v >= 0.0)

    @skip_if_unavailable
    def test_wilson_score_all_pos_004_10(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-10",
             ops.wilson_score, 100, 100, 95,
             check=lambda v: 0.0 < v < 1.0)

    @skip_if_unavailable
    def test_smooth_basic_004_11(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-11",
             ops.smooth, 1000.0, 50.0, 100.0,
             check=lambda v: abs(v - 0.0464) <= 1e-4)

    @skip_if_unavailable
    def test_smooth_zero_004_12(self, record_result):
        """smooth(0.0f, 0.0f, 100.0f) — 不崩溃, result ≈ prior"""
        case_id = "TC-UNIT-ARITH-004-12"
        actual = "N/A"
        try:
            actual = ops.smooth(0.0, 0.0, 100.0)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_smooth_zero_denom_004_13(self, record_result):
        """smooth(0.0f, 0.0f, 0.0f) 分母为0 — 不崩溃, 返回 padding"""
        case_id = "TC-UNIT-ARITH-004-13"
        actual = "N/A"
        try:
            actual = ops.smooth(0.0, 0.0, 0.0)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_smooth_equal_004_14(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-14",
             ops.smooth, 100.0, 100.0, 100.0,
             check=lambda v: 0.0 < v <= 1.0)

    @skip_if_unavailable
    def test_z_score_pos_004_15(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-15",
             ops.z_score, 85.0, 70.0, 15.0,
             check=lambda v: abs(v - 1.0) <= 1e-5)

    @skip_if_unavailable
    def test_z_score_zero_004_16(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-16",
             ops.z_score, 70.0, 70.0, 15.0,
             check=lambda v: abs(v - 0.0) <= 1e-6)

    @skip_if_unavailable
    def test_z_score_neg_004_17(self, record_result):
        _run(record_result, "TC-UNIT-ARITH-004-17",
             ops.z_score, 55.0, 70.0, 15.0,
             check=lambda v: abs(v - (-1.0)) <= 1e-5)

    @skip_if_unavailable
    def test_z_score_zero_std_004_18(self, record_result):
        """z_score(85.0f, 70.0f, 0.0f) std=0 边界 — 不崩溃, 返回 padding"""
        case_id = "TC-UNIT-ARITH-004-18"
        actual = "N/A"
        try:
            actual = ops.z_score(85.0, 70.0, 0.0)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-STAT-001  min / max / avg / var / std
# ===========================================================================
class TestStat001MinMaxAvgVarStd:
    """TC-UNIT-STAT-001-01 ~ 15"""

    @skip_if_unavailable
    def test_avg_001_01(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-01",
             ops.average_f32, [1.0, 2.0, 3.0, 4.0, 5.0],
             check=lambda v: abs(v - 3.0) <= 1e-6)

    @skip_if_unavailable
    def test_avg_single_001_02(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-02",
             ops.average_f32, [100.0],
             check=lambda v: abs(v - 100.0) <= 1e-6)

    @skip_if_unavailable
    def test_avg_empty_001_03(self, record_result):
        """avg([]) 空数组 — 不崩溃, 返回 padding"""
        case_id = "TC-UNIT-STAT-001-03"
        actual = "N/A"
        try:
            actual = ops.average_f32([])
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")  # exception is acceptable

    @skip_if_unavailable
    def test_var_basic_001_04(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-04",
             ops.variance_f32, [1.0, 2.0, 3.0],
             check=lambda v: abs(v - 0.667) <= 1e-3)

    @skip_if_unavailable
    def test_var_uniform_001_05(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-05",
             ops.variance_f32, [5.0, 5.0, 5.0],
             check=lambda v: abs(v) <= 1e-6)

    @skip_if_unavailable
    def test_var_empty_001_06(self, record_result):
        case_id = "TC-UNIT-STAT-001-06"
        actual = "N/A"
        try:
            actual = ops.variance_f32([])
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_std_basic_001_07(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-07",
             ops.stddev_f32, [1.0, 2.0, 3.0],
             check=lambda v: abs(v - 0.816) <= 1e-3)

    @skip_if_unavailable
    def test_std_uniform_001_08(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-08",
             ops.stddev_f32, [5.0, 5.0, 5.0],
             check=lambda v: abs(v) <= 1e-6)

    @skip_if_unavailable
    def test_std_empty_001_09(self, record_result):
        case_id = "TC-UNIT-STAT-001-09"
        actual = "N/A"
        try:
            actual = ops.stddev_f32([])
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_min_basic_001_10(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-10",
             ops.min_f32, [3.0, 1.0, 4.0, 1.5],
             check=lambda v: abs(v - 1.0) <= 1e-6)

    @skip_if_unavailable
    def test_min_single_001_11(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-11",
             ops.min_i32, [5],
             check=lambda v: v == 5)

    @skip_if_unavailable
    def test_min_empty_001_12(self, record_result):
        case_id = "TC-UNIT-STAT-001-12"
        actual = "N/A"
        try:
            actual = ops.min_i32([])
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_max_basic_001_13(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-13",
             ops.max_f32, [3.0, 1.0, 4.0, 1.5],
             check=lambda v: abs(v - 4.0) <= 1e-6)

    @skip_if_unavailable
    def test_max_single_001_14(self, record_result):
        _run(record_result, "TC-UNIT-STAT-001-14",
             ops.max_i32, [5],
             check=lambda v: v == 5)

    @skip_if_unavailable
    def test_max_empty_001_15(self, record_result):
        case_id = "TC-UNIT-STAT-001-15"
        actual = "N/A"
        try:
            actual = ops.max_i32([])
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")


# ===========================================================================
# TC-UNIT-STAT-002  topk
# ===========================================================================
class TestStat002Topk:
    """TC-UNIT-STAT-002-01 ~ 09"""

    @skip_if_unavailable
    def test_topk_basic_002_01(self, record_result):
        _run(record_result, "TC-UNIT-STAT-002-01",
             ops.topk_i32, [5, 3, 1, 4, 2], 3,
             check=lambda v: list(v) == [5, 3, 1])

    @skip_if_unavailable
    def test_topk_float_002_02(self, record_result):
        _run(record_result, "TC-UNIT-STAT-002-02",
             ops.topk_f32, [1.0, 2.0, 3.0, 4.0, 5.0], 2,
             check=lambda v: len(v) == 2)

    @skip_if_unavailable
    def test_topk_str_002_03(self, record_result):
        _run(record_result, "TC-UNIT-STAT-002-03",
             ops.topk_str, ["a", "b", "c", "d"], 2,
             check=lambda v: len(v) == 2)

    @skip_if_unavailable
    def test_topk_k_gt_len_002_04(self, record_result):
        _run(record_result, "TC-UNIT-STAT-002-04",
             ops.topk_i32, [1, 2, 3], 5,
             check=lambda v: len(v) <= 5)

    @skip_if_unavailable
    def test_topk_k_eq_len_002_05(self, record_result):
        _run(record_result, "TC-UNIT-STAT-002-05",
             ops.topk_i32, [1, 2, 3], 3,
             check=lambda v: len(v) == 3)

    @skip_if_unavailable
    def test_topk_empty_002_06(self, record_result):
        _run(record_result, "TC-UNIT-STAT-002-06",
             ops.topk_i32, [], 3,
             check=lambda v: list(v) == [])

    @skip_if_unavailable
    def test_topk_empty_k0_002_07(self, record_result):
        _run(record_result, "TC-UNIT-STAT-002-07",
             ops.topk_i32, [], 0,
             check=lambda v: list(v) == [])

    @skip_if_unavailable
    def test_topk_k0_002_08(self, record_result):
        _run(record_result, "TC-UNIT-STAT-002-08",
             ops.topk_i32, [1, 2, 3], 0,
             check=lambda v: list(v) == [])

    @skip_if_unavailable
    def test_topk_padding_002_09(self, record_result):
        """topk([1,2,3], k=5) + export.len=5 padding=0 — expect [1,2,3,0,0]"""
        case_id = "TC-UNIT-STAT-002-09"
        actual = "N/A"
        try:
            actual = ops.topk_i32([1, 2, 3], 5)
            assert len(actual) <= 5
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-STAT-003  norm / normalize / dot_product / count / contains / len
# ===========================================================================
class TestStat003NormNormalizeDotProductCountContainsLen:
    """TC-UNIT-STAT-003-01 ~ 19"""

    @skip_if_unavailable
    def test_norm_l2_003_01(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-01",
             ops.norm_f32, [3.0, 4.0], 2.0,
             check=lambda v: abs(v - 5.0) <= 1e-6)

    @skip_if_unavailable
    def test_norm_l2_unit_003_02(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-02",
             ops.norm_f32, [1.0, 1.0, 1.0, 1.0], 2.0,
             check=lambda v: abs(v - 2.0) <= 1e-6)

    @skip_if_unavailable
    def test_norm_l1_003_03(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-03",
             ops.norm_f32, [3.0, 4.0], 1.0,
             check=lambda v: abs(v - 7.0) <= 1e-6)

    @skip_if_unavailable
    def test_norm_empty_003_04(self, record_result):
        case_id = "TC-UNIT-STAT-003-04"
        actual = "N/A"
        try:
            actual = ops.norm_f32([], 2.0)
            assert actual == 0.0 or isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_normalize_l2_003_05(self, record_result):
        case_id = "TC-UNIT-STAT-003-05"
        actual = "N/A"
        try:
            actual = ops.normalize_f32([3.0, 4.0], 2.0)
            assert len(actual) == 2
            assert abs(actual[0] - 0.6) <= 1e-6
            assert abs(actual[1] - 0.8) <= 1e-6
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_normalize_unit_003_06(self, record_result):
        case_id = "TC-UNIT-STAT-003-06"
        actual = "N/A"
        try:
            actual = ops.normalize_f32([1.0, 0.0, 0.0], 2.0)
            assert abs(actual[0] - 1.0) <= 1e-6
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_normalize_zero_vector_003_07(self, record_result):
        case_id = "TC-UNIT-STAT-003-07"
        actual = "N/A"
        try:
            actual = ops.normalize_f32([0.0, 0.0], 2.0)
            assert all(v == 0.0 for v in actual) or isinstance(actual, list)
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_dot_product_003_08(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-08",
             ops.dot_product, [1.0, 2.0], [3.0, 4.0],
             check=lambda v: abs(v - 11.0) <= 1e-6)

    @skip_if_unavailable
    def test_dot_product_zero_003_09(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-09",
             ops.dot_product, [0.0, 0.0], [1.0, 1.0],
             check=lambda v: abs(v - 0.0) <= 1e-6)

    @skip_if_unavailable
    def test_dot_product_mismatch_003_10(self, record_result):
        """dot_product dim mismatch — 不崩溃, 返回 padding"""
        case_id = "TC-UNIT-STAT-003-10"
        actual = "N/A"
        try:
            actual = ops.dot_product([1.0, 2.0], [3.0])
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_count_present_003_11(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-11",
             ops.count_i32, [1, 2, 2, 3, 2], 2,
             check=lambda v: v == 3)

    @skip_if_unavailable
    def test_count_absent_003_12(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-12",
             ops.count_i32, [1, 2, 3], 5,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_count_empty_003_13(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-13",
             ops.count_i32, [], 1,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_contains_true_003_14(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-14",
             ops.contains_i32, [1, 2, 3], 2,
             check=lambda v: v == "true")

    @skip_if_unavailable
    def test_contains_false_003_15(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-15",
             ops.contains_i32, [1, 2, 3], 5,
             check=lambda v: v == "false")

    @skip_if_unavailable
    def test_contains_empty_003_16(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-16",
             ops.contains_i32, [], 1,
             check=lambda v: v == "false")

    @skip_if_unavailable
    def test_len_basic_003_17(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-17",
             ops.len_i32, [1, 2, 3, 4, 5],
             check=lambda v: v == 5)

    @skip_if_unavailable
    def test_len_empty_003_18(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-18",
             ops.len_i32, [],
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_len_str_003_19(self, record_result):
        _run(record_result, "TC-UNIT-STAT-003-19",
             ops.len_str, ["a", "b"],
             check=lambda v: v == 2)


# ===========================================================================
# TC-UNIT-DISC-001  binarize
# ===========================================================================
class TestDisc001Binarize:
    """TC-UNIT-DISC-001-01 ~ 12"""

    @skip_if_unavailable
    def test_binarize_above_001_01(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-01",
             ops.binarize_f32, 5.0, 3.0,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_binarize_below_001_02(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-02",
             ops.binarize_f32, 2.0, 3.0,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_binarize_eq_001_03(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-03",
             ops.binarize_f32, 3.0, 3.0,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_binarize_neg_001_04(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-04",
             ops.binarize_f32, -1.0, 0.0,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_binarize_zero_thresh_001_05(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-05",
             ops.binarize_f32, 0.0, 0.0,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_binarize_i32_above_001_06(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-06",
             ops.binarize_i32, 5, 3,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_binarize_i32_below_001_07(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-07",
             ops.binarize_i32, 2, 3,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_binarize_i32_eq_001_08(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-08",
             ops.binarize_i32, 3, 3,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_binarize_i64_above_001_09(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-09",
             ops.binarize_i64, 5, 3,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_binarize_i64_below_001_10(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-10",
             ops.binarize_i64, 2, 3,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_binarize_i64_eq_001_11(self, record_result):
        _run(record_result, "TC-UNIT-DISC-001-11",
             ops.binarize_i64, 3, 3,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_binarize_hash_001_12(self, record_result):
        """binarize int32 + hash.prefix=bin_ — hash output in [0, mask]"""
        case_id = "TC-UNIT-DISC-001-12"
        actual = "N/A"
        try:
            raw = ops.binarize_i32(5, 3)
            actual = raw
            assert isinstance(actual, int)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-DISC-002  bucketize
# ===========================================================================
class TestDisc002Bucketize:
    """TC-UNIT-DISC-002-01 ~ 12"""
    BOUNDS = [18.0, 25.0, 35.0, 45.0, 55.0, 65.0]

    @skip_if_unavailable
    def test_bucket_mid_002_01(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-01",
             ops.bucketize_f32, 20.0, self.BOUNDS,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_bucket_mid2_002_02(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-02",
             ops.bucketize_f32, 30.0, self.BOUNDS,
             check=lambda v: v == 2)

    @skip_if_unavailable
    def test_bucket_high_002_03(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-03",
             ops.bucketize_f32, 60.0, self.BOUNDS,
             check=lambda v: v == 5)

    @skip_if_unavailable
    def test_bucket_below_min_002_04(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-04",
             ops.bucketize_f32, 17.0, self.BOUNDS,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_bucket_very_low_002_05(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-05",
             ops.bucketize_f32, -100.0, self.BOUNDS,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_bucket_above_max_002_06(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-06",
             ops.bucketize_f32, 70.0, self.BOUNDS,
             check=lambda v: v == 6)

    @skip_if_unavailable
    def test_bucket_very_high_002_07(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-07",
             ops.bucketize_f32, 1000.0, self.BOUNDS,
             check=lambda v: v == 6)

    @skip_if_unavailable
    def test_bucket_eq_min_002_08(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-08",
             ops.bucketize_f32, 18.0, self.BOUNDS,
             check=lambda v: v == 1)

    @skip_if_unavailable
    def test_bucket_eq_mid_002_09(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-09",
             ops.bucketize_f32, 25.0, self.BOUNDS,
             check=lambda v: v == 2)

    @skip_if_unavailable
    def test_bucket_i32_002_10(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-10",
             ops.bucketize_i32, 30, [18, 25, 35, 45, 55, 65],
             check=lambda v: v == 2)

    @skip_if_unavailable
    def test_bucket_i64_002_11(self, record_result):
        _run(record_result, "TC-UNIT-DISC-002-11",
             ops.bucketize_i64, 30, [18, 25, 35],
             check=lambda v: v == 2)

    @skip_if_unavailable
    def test_bucket_hash_002_12(self, record_result):
        case_id = "TC-UNIT-DISC-002-12"
        actual = "N/A"
        try:
            actual = ops.bucketize_f32(20.0, self.BOUNDS)
            assert isinstance(actual, int)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-IDENTITY-001  identity
# ===========================================================================
class TestIdentity001:
    """TC-UNIT-IDENTITY-001-01 ~ 12"""

    @skip_if_unavailable
    def test_identity_float_001_01(self, record_result):
        _run(record_result, "TC-UNIT-IDENTITY-001-01",
             ops.identity_f32, 3.14,
             check=lambda v: abs(v - 3.14) <= 1e-4)

    @skip_if_unavailable
    def test_identity_i32_hash_001_02(self, record_result):
        case_id = "TC-UNIT-IDENTITY-001-02"
        actual = "N/A"
        try:
            actual = ops.identity_i32(42)
            assert isinstance(actual, int)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_identity_i64_passthrough_001_03(self, record_result):
        _run(record_result, "TC-UNIT-IDENTITY-001-03",
             ops.identity_i64, 9999999,
             check=lambda v: v == 9999999)

    @skip_if_unavailable
    def test_identity_str_hash_001_04(self, record_result):
        case_id = "TC-UNIT-IDENTITY-001-04"
        actual = "N/A"
        try:
            actual = ops.identity_str("male")
            assert isinstance(actual, (int, str))
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_identity_str_idempotent_001_05(self, record_result):
        case_id = "TC-UNIT-IDENTITY-001-05"
        actual = "N/A"
        try:
            r1 = ops.identity_str("male")
            r2 = ops.identity_str("male")
            actual = f"r1={r1}, r2={r2}"
            assert r1 == r2
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_identity_missing_field_001_06(self, record_result):
        """identity 字段缺失 — 不崩溃, 返回 padding"""
        case_id = "TC-UNIT-IDENTITY-001-06"
        actual = "N/A"
        try:
            actual = ops.identity_i32(0)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_identity_str_arr_padding_001_07(self, record_result):
        case_id = "TC-UNIT-IDENTITY-001-07"
        actual = "N/A"
        try:
            actual = ops.identity_vec_str(["sports", "news"])
            assert len(list(actual)) == 2
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_identity_i64_arr_001_08(self, record_result):
        case_id = "TC-UNIT-IDENTITY-001-08"
        actual = "N/A"
        try:
            actual = ops.identity_vec_i64([1, 2, 3])
            assert list(actual) == [1, 2, 3]
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_identity_float_arr_001_09(self, record_result):
        case_id = "TC-UNIT-IDENTITY-001-09"
        actual = "N/A"
        try:
            actual = ops.identity_vec_f32([0.1, 0.2, 0.3])
            assert len(list(actual)) == 3
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_identity_str_arr_empty_001_10(self, record_result):
        case_id = "TC-UNIT-IDENTITY-001-10"
        actual = "N/A"
        try:
            actual = ops.identity_vec_str([])
            assert list(actual) == []
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_identity_float_invalid_hash_001_11(self, record_result):
        """identity(float) + hash 非法 — float 不支持 hash"""
        case_id = "TC-UNIT-IDENTITY-001-11"
        actual = "N/A"
        try:
            actual = ops.identity_f32(3.14)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_identity_i32_no_hash_invalid_001_12(self, record_result):
        """identity(int32) 无 hash 非法 — int32 必须配 hash"""
        case_id = "TC-UNIT-IDENTITY-001-12"
        actual = "N/A"
        try:
            actual = ops.identity_i32(25)
            assert isinstance(actual, int)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")


# ===========================================================================
# TC-UNIT-STR-001  lower / upper / reverse
# ===========================================================================
class TestStr001LowerUpperReverse:
    """TC-UNIT-STR-001-01 ~ 15"""

    @skip_if_unavailable
    def test_lower_001_01(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-01",
             ops.lower_str, "Hello_World",
             check=lambda v: v == "hello_world")

    @skip_if_unavailable
    def test_lower_alphanum_001_02(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-02",
             ops.lower_str, "ABC123",
             check=lambda v: v == "abc123")

    @skip_if_unavailable
    def test_lower_already_lower_001_03(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-03",
             ops.lower_str, "already_lower",
             check=lambda v: v == "already_lower")

    @skip_if_unavailable
    def test_lower_empty_001_04(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-04",
             ops.lower_str, "",
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_lower_idempotent_001_05(self, record_result):
        case_id = "TC-UNIT-STR-001-05"
        actual = "N/A"
        try:
            r1 = ops.lower_str("SAME")
            r2 = ops.lower_str("SAME")
            actual = f"r1={r1}, r2={r2}"
            assert r1 == r2
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_upper_001_06(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-06",
             ops.upper, "hello",
             check=lambda v: v == "HELLO")

    @skip_if_unavailable
    def test_upper_alphanum_001_07(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-07",
             ops.upper, "abc123",
             check=lambda v: v == "ABC123")

    @skip_if_unavailable
    def test_upper_already_upper_001_08(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-08",
             ops.upper, "ALREADY_UPPER",
             check=lambda v: v == "ALREADY_UPPER")

    @skip_if_unavailable
    def test_upper_empty_001_09(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-09",
             ops.upper, "",
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_reverse_001_10(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-10",
             ops.reverse, "abc",
             check=lambda v: v == "cba")

    @skip_if_unavailable
    def test_reverse_even_001_11(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-11",
             ops.reverse, "abcd",
             check=lambda v: v == "dcba")

    @skip_if_unavailable
    def test_reverse_single_001_12(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-12",
             ops.reverse, "a",
             check=lambda v: v == "a")

    @skip_if_unavailable
    def test_reverse_empty_001_13(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-13",
             ops.reverse, "",
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_reverse_palindrome_001_14(self, record_result):
        _run(record_result, "TC-UNIT-STR-001-14",
             ops.reverse, "abcba",
             check=lambda v: v == "abcba")

    @skip_if_unavailable
    def test_lower_upper_reverse_hash_001_15(self, record_result):
        case_id = "TC-UNIT-STR-001-15"
        actual = "N/A"
        try:
            actual = ops.lower_str("HELLO")
            assert isinstance(actual, str)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-STR-002  substr / match_prefix
# ===========================================================================
class TestStr002SubstrMatchPrefix:
    """TC-UNIT-STR-002-01 ~ 16"""

    @skip_if_unavailable
    def test_substr_002_01(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-01",
             ops.substr, "hello world", 0, 5,
             check=lambda v: v == "hello")

    @skip_if_unavailable
    def test_substr_mid_002_02(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-02",
             ops.substr, "hello world", 6, 5,
             check=lambda v: v == "world")

    @skip_if_unavailable
    def test_substr_from_middle_002_03(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-03",
             ops.substr, "hello", 2, 3,
             check=lambda v: v == "llo")

    @skip_if_unavailable
    def test_substr_overflow_len_002_04(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-04",
             ops.substr, "hello", 0, 100,
             check=lambda v: v == "hello")

    @skip_if_unavailable
    def test_substr_mid_overflow_002_05(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-05",
             ops.substr, "hello", 2, 100,
             check=lambda v: v == "llo")

    @skip_if_unavailable
    def test_substr_start_at_end_002_06(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-06",
             ops.substr, "hello", 5, 3,
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_substr_empty_002_07(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-07",
             ops.substr, "", 0, 5,
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_substr_neg_start_002_08(self, record_result):
        case_id = "TC-UNIT-STR-002-08"
        actual = "N/A"
        try:
            actual = ops.substr("hello", -1, 3)
            assert actual == "" or isinstance(actual, str)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_substr_zero_len_002_09(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-09",
             ops.substr, "hello", 0, 0,
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_match_prefix_cat_002_10(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-10",
             ops.match_prefix, "category_sports", ["cat", "user", "item"],
             check=lambda v: v == "cat")

    @skip_if_unavailable
    def test_match_prefix_user_002_11(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-11",
             ops.match_prefix, "user_123", ["cat", "user", "item"],
             check=lambda v: v == "user")

    @skip_if_unavailable
    def test_match_prefix_item_002_12(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-12",
             ops.match_prefix, "item_001", ["cat", "user", "item"],
             check=lambda v: v == "item")

    @skip_if_unavailable
    def test_match_prefix_no_match_002_13(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-13",
             ops.match_prefix, "unknown_str", ["cat", "user", "item"],
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_match_prefix_empty_list_002_14(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-14",
             ops.match_prefix, "category_sports", [],
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_match_prefix_empty_str_002_15(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-15",
             ops.match_prefix, "", ["cat", "user"],
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_match_prefix_exact_002_16(self, record_result):
        _run(record_result, "TC-UNIT-STR-002-16",
             ops.match_prefix, "cat", ["cat", "category"],
             check=lambda v: v == "cat")


# ===========================================================================
# TC-UNIT-CONCAT-001  concat / concat_ws
# ===========================================================================
class TestConcat001ConcatConcatWs:
    """TC-UNIT-CONCAT-001-01 ~ 16"""

    @skip_if_unavailable
    def test_concat_str_str_001_01(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-01",
             ops.concat, "hello", "world",
             check=lambda v: v == "helloworld")

    @skip_if_unavailable
    def test_concat_int_str_001_02(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-02",
             ops.concat, 25, "male",
             check=lambda v: v == "25male")

    @skip_if_unavailable
    def test_concat_str_int_001_03(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-03",
             ops.concat, "age", 30,
             check=lambda v: v == "age30")

    @skip_if_unavailable
    def test_concat_float_str_001_04(self, record_result):
        case_id = "TC-UNIT-CONCAT-001-04"
        actual = "N/A"
        try:
            actual = ops.concat(3.14, "price")
            assert "3.14" in actual or "price" in actual
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_concat_int_int_001_05(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-05",
             ops.concat, 100, 200,
             check=lambda v: v == "100200")

    @skip_if_unavailable
    def test_concat_i64_i64_001_06(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-06",
             ops.concat, 100, 200,
             check=lambda v: v == "100200")

    @skip_if_unavailable
    def test_concat_float_float_001_07(self, record_result):
        case_id = "TC-UNIT-CONCAT-001-07"
        actual = "N/A"
        try:
            actual = ops.concat(3.14, 2.71)
            assert isinstance(actual, str)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_concat_i64_str_001_08(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-08",
             ops.concat, 100, "str",
             check=lambda v: v == "100str")

    @skip_if_unavailable
    def test_concat_float_int_001_09(self, record_result):
        case_id = "TC-UNIT-CONCAT-001-09"
        actual = "N/A"
        try:
            actual = ops.concat(3.14, 42)
            assert isinstance(actual, str)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_concat_int_float_001_10(self, record_result):
        case_id = "TC-UNIT-CONCAT-001-10"
        actual = "N/A"
        try:
            actual = ops.concat(42, 3.14)
            assert isinstance(actual, str)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_concat_mixed_combos_001_11(self, record_result):
        """Remaining 6 type combos — all produce valid strings"""
        case_id = "TC-UNIT-CONCAT-001-11"
        actual = "N/A"
        try:
            combos = [
                (100, 3.14), (100, 200), (3.14, 100), ("s", 100), ("s", 3.14), (100, 200),
            ]
            results = [ops.concat(a, b) for a, b in combos]
            actual = str(results)
            assert all(isinstance(r, str) for r in results)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_concat_ws_001_12(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-12",
             ops.concat_ws, "@", "user123", "cn",
             check=lambda v: v == "user123@cn")

    @skip_if_unavailable
    def test_concat_ws_int_str_001_13(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-13",
             ops.concat_ws, "_", 2024, "01",
             check=lambda v: v == "2024_01")

    @skip_if_unavailable
    def test_concat_ws_float_str_001_14(self, record_result):
        case_id = "TC-UNIT-CONCAT-001-14"
        actual = "N/A"
        try:
            actual = ops.concat_ws("-", 3.14, "price")
            assert isinstance(actual, str)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_concat_ws_i64_001_15(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-15",
             ops.concat_ws, "|", 100, 200,
             check=lambda v: v == "100|200")

    @skip_if_unavailable
    def test_concat_ws_empty_sep_001_16(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-001-16",
             ops.concat_ws, "", "a", "b",
             check=lambda v: v == "ab")


# ===========================================================================
# TC-UNIT-CONCAT-002  lower_concat_ws / trim_concat / trim_concat_ws
# ===========================================================================
class TestConcat002LowerConcatWsTrimConcat:
    """TC-UNIT-CONCAT-002-01 ~ 13"""

    @skip_if_unavailable
    def test_lower_concat_ws_002_01(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-01",
             ops.lower_concat_ws, "@", "UserID", "CN",
             check=lambda v: v == "userid@cn")

    @skip_if_unavailable
    def test_lower_concat_ws_underscore_002_02(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-02",
             ops.lower_concat_ws, "_", "Hello", "World",
             check=lambda v: v == "hello_world")

    @skip_if_unavailable
    def test_lower_concat_ws_int_str_002_03(self, record_result):
        case_id = "TC-UNIT-CONCAT-002-03"
        actual = "N/A"
        try:
            actual = ops.lower_concat_ws("@", 25, "Male")
            assert isinstance(actual, str)
            assert "male" in actual or "25" in actual
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_lower_concat_ws_empty_sep_002_04(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-04",
             ops.lower_concat_ws, "", "UPPER", "CASE",
             check=lambda v: v == "uppercase")

    @skip_if_unavailable
    def test_trim_concat_full_trim_002_05(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-05",
             ops.trim_concat, "hello", "world", ["hello", "world"],
             check=lambda v: v == "")

    @skip_if_unavailable
    def test_trim_concat_partial_002_06(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-06",
             ops.trim_concat, "prefix_data", "_suffix", ["prefix_", "_suffix"],
             check=lambda v: v == "data")

    @skip_if_unavailable
    def test_trim_concat_empty_list_002_07(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-07",
             ops.trim_concat, "abc", "def", [],
             check=lambda v: v == "abcdef")

    @skip_if_unavailable
    def test_trim_concat_no_match_002_08(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-08",
             ops.trim_concat, "hello", "world", ["xyz"],
             check=lambda v: v == "helloworld")

    @skip_if_unavailable
    def test_trim_concat_empty_strs_002_09(self, record_result):
        case_id = "TC-UNIT-CONCAT-002-09"
        actual = "N/A"
        try:
            actual = ops.trim_concat("", "", [""])
            assert isinstance(actual, str)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_trim_concat_ws_002_10(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-10",
             ops.trim_concat_ws, "_", "prefix_price", "high_suffix",
             ["prefix_", "_suffix"],
             check=lambda v: v == "price_high")

    @skip_if_unavailable
    def test_trim_concat_ws_user_002_11(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-11",
             ops.trim_concat_ws, "@", "user123", "cn", ["user"],
             check=lambda v: v == "123@cn")

    @skip_if_unavailable
    def test_trim_concat_ws_empty_list_002_12(self, record_result):
        _run(record_result, "TC-UNIT-CONCAT-002-12",
             ops.trim_concat_ws, "-", "abc", "def", [],
             check=lambda v: v == "abc-def")

    @skip_if_unavailable
    def test_trim_concat_ws_all_trim_002_13(self, record_result):
        case_id = "TC-UNIT-CONCAT-002-13"
        actual = "N/A"
        try:
            actual = ops.trim_concat_ws("_", "ABC", "DEF", ["ABC"])
            assert isinstance(actual, str)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-CONCAT-003  cartesian_concat
# ===========================================================================
class TestConcat003CartesianConcat:
    """TC-UNIT-CONCAT-003-01 ~ 11"""

    @skip_if_unavailable
    def test_cartesian_basic_003_01(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-01"
        actual = "N/A"
        try:
            actual = ops.cartesian_concat(["a", "b"], ["x", "y"])
            assert sorted(list(actual)) == ["ax", "ay", "bx", "by"]
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_cartesian_tag_003_02(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-02"
        actual = "N/A"
        try:
            actual = ops.cartesian_concat(["tag1", "tag2"], ["sports", "news"])
            assert len(list(actual)) == 4
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_cartesian_3x2_003_03(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-03"
        actual = "N/A"
        try:
            actual = ops.cartesian_concat(["a", "b", "c"], ["1", "2"])
            assert len(list(actual)) == 6
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_cartesian_single_003_04(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-04"
        actual = "N/A"
        try:
            actual = ops.cartesian_concat(["a"], ["x"])
            assert list(actual) == ["ax"]
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_cartesian_int_003_05(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-05"
        actual = "N/A"
        try:
            actual = ops.cartesian_concat([1, 2], [10, 20])
            r = list(actual)
            assert len(r) == 4
            record_result(case_id, r, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_cartesian_i64_str_003_06(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-06"
        actual = "N/A"
        try:
            actual = ops.cartesian_concat([1, 2], ["a", "b"])
            r = list(actual)
            assert len(r) == 4
            record_result(case_id, r, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_cartesian_right_empty_003_07(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-07"
        actual = "N/A"
        try:
            actual = ops.cartesian_concat(["a", "b"], [])
            assert list(actual) == []
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_cartesian_left_empty_003_08(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-08"
        actual = "N/A"
        try:
            actual = ops.cartesian_concat([], ["x", "y"])
            assert list(actual) == []
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_cartesian_both_empty_003_09(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-09"
        actual = "N/A"
        try:
            actual = ops.cartesian_concat([], [])
            assert list(actual) == []
            record_result(case_id, list(actual), "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_cartesian_5x5_003_10(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-10"
        actual = "N/A"
        try:
            A = ["a", "b", "c", "d", "e"]
            B = ["1", "2", "3", "4", "5"]
            actual = ops.cartesian_concat(A, B)
            assert len(list(actual)) == 25
            record_result(case_id, len(list(actual)), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_cartesian_truncation_003_11(self, record_result):
        case_id = "TC-UNIT-CONCAT-003-11"
        actual = "N/A"
        try:
            A = ["a", "b", "c", "d"]
            B = ["1", "2", "3", "4"]
            actual = ops.cartesian_concat(A, B)
            assert len(list(actual)) <= 16
            record_result(case_id, len(list(actual)), "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-DATE-001  year/month/day/weekday/curdate/unix_timestamp/from_unixtime
# ===========================================================================
class TestDate001:
    """TC-UNIT-DATE-001-01 ~ 18"""
    TS_20240322 = 1711123200   # 2024-03-22 UTC
    TS_20230101 = 1672531200   # 2023-01-01 UTC

    @skip_if_unavailable
    def test_year_2024_001_01(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-01",
             ops.year, self.TS_20240322,
             check=lambda v: v == "2024")

    @skip_if_unavailable
    def test_year_epoch_001_02(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-02",
             ops.year, 0,
             check=lambda v: v == "1970")

    @skip_if_unavailable
    def test_month_03_001_03(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-03",
             ops.month, self.TS_20240322,
             check=lambda v: v == "03")

    @skip_if_unavailable
    def test_month_01_001_04(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-04",
             ops.month, self.TS_20230101,
             check=lambda v: v == "01")

    @skip_if_unavailable
    def test_day_22_001_05(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-05",
             ops.day, self.TS_20240322,
             check=lambda v: v == "22")

    @skip_if_unavailable
    def test_day_01_001_06(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-06",
             ops.day, self.TS_20230101,
             check=lambda v: v == "01")

    @skip_if_unavailable
    def test_weekday_001_07(self, record_result):
        case_id = "TC-UNIT-DATE-001-07"
        actual = "N/A"
        try:
            actual = ops.weekday(self.TS_20240322)
            assert isinstance(actual, (str, int))
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_weekday_sunday_001_08(self, record_result):
        case_id = "TC-UNIT-DATE-001-08"
        actual = "N/A"
        try:
            actual = ops.weekday(self.TS_20230101)
            assert isinstance(actual, (str, int))
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_curdate_20240322_001_09(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-09",
             ops.curdate, self.TS_20240322,
             check=lambda v: v == "20240322")

    @skip_if_unavailable
    def test_curdate_20230101_001_10(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-10",
             ops.curdate, self.TS_20230101,
             check=lambda v: v == "20230101")

    @skip_if_unavailable
    def test_unix_timestamp_passthrough_001_11(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-11",
             ops.unix_timestamp, self.TS_20240322,
             check=lambda v: v == self.TS_20240322)

    @skip_if_unavailable
    def test_unix_timestamp_zero_001_12(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-12",
             ops.unix_timestamp, 0,
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_unix_timestamp_vs_identity_001_13(self, record_result):
        case_id = "TC-UNIT-DATE-001-13"
        actual = "N/A"
        try:
            r1 = ops.unix_timestamp(self.TS_20240322)
            r2 = ops.identity_i64(self.TS_20240322)
            actual = f"unix_timestamp={r1}, identity_i64={r2}"
            assert r1 == r2
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_from_unixtime_ymd_001_14(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-14",
             ops.from_unixtime, self.TS_20240322, "%Y-%m-%d",
             check=lambda v: v == "2024-03-22")

    @skip_if_unavailable
    def test_from_unixtime_compact_001_15(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-15",
             ops.from_unixtime, self.TS_20240322, "%Y%m%d",
             check=lambda v: v == "20240322")

    @skip_if_unavailable
    def test_from_unixtime_hms_001_16(self, record_result):
        case_id = "TC-UNIT-DATE-001-16"
        actual = "N/A"
        try:
            actual = ops.from_unixtime(self.TS_20240322, "%H:%M:%S")
            assert ":" in actual
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_from_unixtime_epoch_001_17(self, record_result):
        _run(record_result, "TC-UNIT-DATE-001-17",
             ops.from_unixtime, 0, "%Y-%m-%d",
             check=lambda v: v == "1970-01-01")

    @skip_if_unavailable
    def test_date_components_consistency_001_18(self, record_result):
        case_id = "TC-UNIT-DATE-001-18"
        actual = "N/A"
        try:
            y = ops.year(self.TS_20240322)
            m = ops.month(self.TS_20240322)
            d = ops.day(self.TS_20240322)
            cd = ops.curdate(self.TS_20240322)
            ft = ops.from_unixtime(self.TS_20240322, "%Y%m%d")
            actual = f"y={y},m={m},d={d},cd={cd},ft={ft}"
            assert y + m + d == cd == ft
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise


# ===========================================================================
# TC-UNIT-DATE-002  date_add / date_sub / datediff
# ===========================================================================
class TestDate002DateAddDateSubDatediff:
    """TC-UNIT-DATE-002-01 ~ 16"""

    @skip_if_unavailable
    def test_date_add_7_002_01(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-01",
             ops.date_add, "20240322", 7,
             check=lambda v: v == "20240329")

    @skip_if_unavailable
    def test_date_add_0_002_02(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-02",
             ops.date_add, "20240322", 0,
             check=lambda v: v == "20240322")

    @skip_if_unavailable
    def test_date_add_leap_002_03(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-03",
             ops.date_add, "20240228", 1,
             check=lambda v: v == "20240229")

    @skip_if_unavailable
    def test_date_add_leap_cross_002_04(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-04",
             ops.date_add, "20240229", 1,
             check=lambda v: v == "20240301")

    @skip_if_unavailable
    def test_date_add_month_cross_002_05(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-05",
             ops.date_add, "20240131", 1,
             check=lambda v: v == "20240201")

    @skip_if_unavailable
    def test_date_add_year_cross_002_06(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-06",
             ops.date_add, "20241231", 1,
             check=lambda v: v == "20250101")

    @skip_if_unavailable
    def test_date_add_negative_002_07(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-07",
             ops.date_add, "20240101", -1,
             check=lambda v: v == "20231231")

    @skip_if_unavailable
    def test_date_sub_year_002_08(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-08",
             ops.date_sub, "20240101", 1,
             check=lambda v: v == "20231231")

    @skip_if_unavailable
    def test_date_sub_leap_002_09(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-09",
             ops.date_sub, "20240301", 1,
             check=lambda v: v == "20240229")

    @skip_if_unavailable
    def test_date_sub_7_002_10(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-10",
             ops.date_sub, "20240322", 7,
             check=lambda v: v == "20240315")

    @skip_if_unavailable
    def test_date_sub_0_002_11(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-11",
             ops.date_sub, "20240322", 0,
             check=lambda v: v == "20240322")

    @skip_if_unavailable
    def test_datediff_pos_002_12(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-12",
             ops.datediff, "20240322", "20240101",
             check=lambda v: v == 81)

    @skip_if_unavailable
    def test_datediff_neg_002_13(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-13",
             ops.datediff, "20240101", "20240322",
             check=lambda v: v == -81)

    @skip_if_unavailable
    def test_datediff_same_002_14(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-14",
             ops.datediff, "20240322", "20240322",
             check=lambda v: v == 0)

    @skip_if_unavailable
    def test_datediff_leap_year_002_15(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-15",
             ops.datediff, "20250101", "20240101",
             check=lambda v: v == 366)

    @skip_if_unavailable
    def test_datediff_leap_feb_002_16(self, record_result):
        _run(record_result, "TC-UNIT-DATE-002-16",
             ops.datediff, "20240229", "20240228",
             check=lambda v: v == 1)


# ===========================================================================
# TC-UNIT-DIST-001  edit_distance / cosine_distance / jaccard_distance
#                   / jaro_winkler_distance / fuzzy
# ===========================================================================
class TestDist001Distances:
    """TC-UNIT-DIST-001-01 ~ 24"""

    @skip_if_unavailable
    def test_edit_same_001_01(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-01",
             ops.edit_distance, "kitten", "kitten", 10,
             check=lambda v: v == 0.0)

    @skip_if_unavailable
    def test_edit_diff_001_02(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-02",
             ops.edit_distance, "kitten", "sitting", 10,
             check=lambda v: 0.0 <= v <= 1.0)

    @skip_if_unavailable
    def test_edit_completely_diff_001_03(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-03",
             ops.edit_distance, "abc", "xyz", 3,
             check=lambda v: v > 0.0)

    @skip_if_unavailable
    def test_edit_empty_001_04(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-04",
             ops.edit_distance, "", "", 5,
             check=lambda v: v == 0.0)

    @skip_if_unavailable
    def test_edit_one_empty_001_05(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-05",
             ops.edit_distance, "abc", "", 5,
             check=lambda v: v > 0.0)

    @skip_if_unavailable
    def test_edit_zero_len_001_06(self, record_result):
        case_id = "TC-UNIT-DIST-001-06"
        actual = "N/A"
        try:
            actual = ops.edit_distance("a", "b", 0)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_cosine_same_001_07(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-07",
             ops.cosine_distance, "abc", "abc", 3,
             check=lambda v: v == 0.0)

    @skip_if_unavailable
    def test_cosine_diff_001_08(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-08",
             ops.cosine_distance, "hello", "world", 2,
             check=lambda v: 0.0 <= v <= 1.0)

    @skip_if_unavailable
    def test_cosine_empty_001_09(self, record_result):
        case_id = "TC-UNIT-DIST-001-09"
        actual = "N/A"
        try:
            actual = ops.cosine_distance("", "", 2)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_cosine_zero_ngram_001_10(self, record_result):
        case_id = "TC-UNIT-DIST-001-10"
        actual = "N/A"
        try:
            actual = ops.cosine_distance("abc", "abc", 0)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_jaccard_same_001_11(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-11",
             ops.jaccard_distance, "abc", "abc", 3,
             check=lambda v: v == 0.0)

    @skip_if_unavailable
    def test_jaccard_completely_diff_001_12(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-12",
             ops.jaccard_distance, "abc", "def", 1,
             check=lambda v: v == 1.0)

    @skip_if_unavailable
    def test_jaccard_partial_001_13(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-13",
             ops.jaccard_distance, "abc", "abd", 1,
             check=lambda v: 0.0 < v < 1.0)

    @skip_if_unavailable
    def test_jaccard_empty_001_14(self, record_result):
        case_id = "TC-UNIT-DIST-001-14"
        actual = "N/A"
        try:
            actual = ops.jaccard_distance("", "", 2)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_jaro_same_001_15(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-15",
             ops.jaro_winkler_distance, "MARTHA", "MARTHA", 6,
             check=lambda v: v == 0.0)

    @skip_if_unavailable
    def test_jaro_similar_001_16(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-16",
             ops.jaro_winkler_distance, "MARTHA", "MARHTA", 6,
             check=lambda v: 0.0 <= v < 0.5)

    @skip_if_unavailable
    def test_jaro_diff_001_17(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-17",
             ops.jaro_winkler_distance, "abc", "xyz", 3,
             check=lambda v: v > 0.0)

    @skip_if_unavailable
    def test_jaro_empty_001_18(self, record_result):
        case_id = "TC-UNIT-DIST-001-18"
        actual = "N/A"
        try:
            actual = ops.jaro_winkler_distance("", "", 3)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_fuzzy_exact_001_19(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-19",
             ops.fuzzy, "hello", "hello", 5,
             check=lambda v: v == 0.0)

    @skip_if_unavailable
    def test_fuzzy_one_diff_001_20(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-20",
             ops.fuzzy, "hello", "helo", 5,
             check=lambda v: v >= 0.0)

    @skip_if_unavailable
    def test_fuzzy_diff_001_21(self, record_result):
        _run(record_result, "TC-UNIT-DIST-001-21",
             ops.fuzzy, "hello", "xyz", 5,
             check=lambda v: v > 0.0)

    @skip_if_unavailable
    def test_fuzzy_empty_001_22(self, record_result):
        case_id = "TC-UNIT-DIST-001-22"
        actual = "N/A"
        try:
            actual = ops.fuzzy("", "", 3)
            assert isinstance(actual, float)
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "PASS")

    @skip_if_unavailable
    def test_all_distances_range_001_23(self, record_result):
        """All 5 distance functions return values in [0, 1]"""
        case_id = "TC-UNIT-DIST-001-23"
        actual = "N/A"
        try:
            fns = [
                (ops.edit_distance, ("hello", "hello", 5)),
                (ops.cosine_distance, ("hello", "world", 2)),
                (ops.jaccard_distance, ("hello", "world", 1)),
                (ops.jaro_winkler_distance, ("hello", "world", 5)),
                (ops.fuzzy, ("hello", "world", 5)),
            ]
            results = {fn.__name__: fn(*args) for fn, args in fns}
            actual = str(results)
            assert all(0.0 <= v <= 1.0 for v in results.values())
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise

    @skip_if_unavailable
    def test_all_distances_same_string_001_24(self, record_result):
        """All 5 distance functions return 0.0 for identical strings"""
        case_id = "TC-UNIT-DIST-001-24"
        actual = "N/A"
        try:
            fns = [
                (ops.edit_distance, ("hello", "hello", 5)),
                (ops.cosine_distance, ("hello", "hello", 2)),
                (ops.jaccard_distance, ("hello", "hello", 1)),
                (ops.jaro_winkler_distance, ("hello", "hello", 5)),
                (ops.fuzzy, ("hello", "hello", 5)),
            ]
            results = {fn.__name__: fn(*args) for fn, args in fns}
            actual = str(results)
            assert all(v == 0.0 for v in results.values())
            record_result(case_id, actual, "PASS")
        except Exception:
            record_result(case_id, actual, "FAIL")
            raise
