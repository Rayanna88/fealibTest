"""
conftest.py — shared pytest fixtures and utilities for fealib test suite.

Hash helpers reproduce features::hash_string() / hash_int() logic from
util.hpp (MurmurHash3_x64_128, seed=0, take low-64 bits & mask).

Requires:
  pip install pyfealib mmh3 pyyaml pytest pytest-xdist
"""

import csv
import os
import struct
from pathlib import Path
import pytest

try:
    import mmh3
except ImportError:
    mmh3 = None

try:
    import pyfealib
    PYFEALIB_AVAILABLE = True
except ImportError:
    pyfealib = None
    PYFEALIB_AVAILABLE = False


# ============================================================
# Paths
# ============================================================
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fealib-cpp")
)
TESTDATA_DIR = os.path.join(REPO_ROOT, "test", "testdata")
YAML_PATH = os.path.join(TESTDATA_DIR, "fealib.yaml")

# fixtures shipped with this test project
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_YAML = os.path.join(FIXTURE_DIR, "fealib.yaml")

# collection directory search path
COLLECTION_CANDIDATES = [
    os.path.join(TESTDATA_DIR, "collection"),
    os.path.join(REPO_ROOT, "test_data", "itemfeatures", "20260323194530"),
]
COLLECTION_DIR = next(
    (p for p in COLLECTION_CANDIDATES if os.path.isdir(p)), None
)

# CSV test case files
CSV_INPUT_PATH = str(Path.home() / "Downloads" / "OP单元测试用例.csv")
CSV_OUTPUT_PATH = str(Path.home() / "Downloads" / "OP单元测试用例_结果.csv")

# ============================================================
# CSV Result recording — shared state
# ============================================================
# key: sub-case ID (e.g. "TC-UNIT-ARITH-001-01")
# value: {"actual": str, "result": "PASS" | "FAIL" | "SKIP" | "ERROR"}
_csv_results: dict = {}


def _record(case_id: str, actual, status: str) -> None:
    """Low-level recording function; also usable from test helpers."""
    _csv_results[case_id] = {"actual": str(actual), "result": status}


# ============================================================
# Hash helpers
# ============================================================
DEFAULT_MASK = 0x1FFFFFFFFFFFFF  # 2^53 - 1


def _require_mmh3():
    if mmh3 is None:
        pytest.skip("mmh3 not installed")


def _low64(key_bytes: bytes, mask: int) -> int:
    """MurmurHash3_x64_128(seed=0) low-64 bits, then & mask."""
    h = mmh3.hash128(key_bytes, seed=0, x64arch=True)
    return (h & 0xFFFFFFFFFFFFFFFF) & mask


def str_hash(prefix: str, value: str, mask: int = DEFAULT_MASK) -> int:
    """Hash a string feature: low64(mmh3(prefix + value))."""
    _require_mmh3()
    return _low64((prefix + value).encode("utf-8"), mask)


def int64_hash(prefix: str, value: int, mask: int = DEFAULT_MASK) -> int:
    """Hash an int64 feature: low64(mmh3(prefix_bytes + little-endian int64))."""
    _require_mmh3()
    return _low64(prefix.encode("utf-8") + struct.pack("<q", value), mask)


def int32_hash(prefix: str, value: int, mask: int = DEFAULT_MASK) -> int:
    """Hash an int32 feature: low64(mmh3(prefix_bytes + little-endian int32))."""
    _require_mmh3()
    return _low64(prefix.encode("utf-8") + struct.pack("<i", value), mask)


# ============================================================
# Column layout constants  (from fealib.yaml, sorted by slot)
# ============================================================
class SparseCol:
    USER_ID              = 0   # slot  0
    USER_COUNTRY         = 1   # slot  1
    PACKAGENAME          = 2   # slot 10
    CATEGORY             = 3   # slot 11
    CATEGORYNAME         = 4   # slot 12
    LAN                  = 5   # slot 13
    STATUS               = 6   # slot 14
    TOKENS               = 7   # slot 20
    DOWNLOAD             = 8   # slot 40
    EXPOSE7D             = 9   # slot 41
    CLICK7D              = 10  # slot 42
    EXPOSE7D_BY_COUNTRY  = 11  # slot 50  (format feature)
    CROSS_COUNTRY_CAT    = 12  # slot 60
    CROSS_COUNTRY_LAN    = 13  # slot 61


class DenseCol:
    INSTALL_CNT = 0   # slot  2
    STAR        = 1   # slot 30
    SCORE       = 2   # slot 31


TOTAL_SPARSE_COLS = 14
TOTAL_DENSE_COLS  = 3


# ============================================================
# Sample feature builders
# ============================================================
def make_user_features(
    uid: int = 123456,
    country: str = "us",
    install_cnt: float = 5.0,
) -> dict:
    """Build a user feature dict matching pyfealib.Fealib.run() signature."""
    if not PYFEALIB_AVAILABLE:
        pytest.skip("pyfealib not installed")
    DT = pyfealib.DataType
    return {
        "uid":         {"type": DT.kInt64Value,  "value": uid},
        "country":     {"type": DT.kStringValue, "value": country},
        "install_cnt": {"type": DT.kFloatValue,  "value": install_cnt},
    }


def make_context_features(country: str = "us") -> dict:
    """Build a context feature dict."""
    if not PYFEALIB_AVAILABLE:
        pytest.skip("pyfealib not installed")
    DT = pyfealib.DataType
    return {
        "country": {"type": DT.kStringValue, "value": country},
    }


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="session")
def fealib_handler():
    """
    Session-scoped pyfealib.Fealib instance.
    Skips if pyfealib is not installed or collection is not found.
    """
    if not PYFEALIB_AVAILABLE:
        pytest.skip("pyfealib not installed")
    if COLLECTION_DIR is None:
        pytest.skip(
            "Collection directory not found; "
            "set COLLECTION_DIR or place data under test/testdata/collection"
        )
    if not os.path.isfile(YAML_PATH):
        pytest.skip(f"fealib.yaml not found: {YAML_PATH}")

    handler = pyfealib.Fealib(YAML_PATH)
    handler.load(COLLECTION_DIR)
    return handler


@pytest.fixture
def user_feats():
    return make_user_features()


@pytest.fixture
def ctx_feats():
    return make_context_features()


@pytest.fixture
def record_result():
    """
    Fixture providing a callable to record actual result for CSV output.

    Usage inside a test:
        def test_foo(record_result):
            actual = "N/A"
            try:
                actual = ops.add_f32(3.5, 2.0)
                assert abs(actual - 5.5) < 1e-6
                record_result("TC-UNIT-ARITH-001-01", actual, "PASS")
            except Exception:
                record_result("TC-UNIT-ARITH-001-01", actual, "FAIL")
                raise
    """
    return _record


# ============================================================
# pytest hooks — write CSV results at session end
# ============================================================
def pytest_sessionfinish(session, exitstatus):
    """After the full test session, write actual results + test status to CSV."""
    if not _csv_results:
        return

    input_path = CSV_INPUT_PATH
    output_path = CSV_OUTPUT_PATH

    if not os.path.isfile(input_path):
        return

    # Read original CSV
    rows = []
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    header = rows[0]

    # Find column indices for 子用例编号, 实际结果, 测试结果
    try:
        id_col = header.index("子用例编号")
    except ValueError:
        id_col = 0
    try:
        actual_col = header.index("实际结果")
    except ValueError:
        actual_col = 7
    try:
        result_col = header.index("测试结果")
    except ValueError:
        result_col = 8

    # Update rows
    updated = 0
    for row in rows[1:]:
        # Extend row if too short
        while len(row) <= max(id_col, actual_col, result_col):
            row.append("")

        case_id = row[id_col].strip()
        if case_id in _csv_results:
            row[actual_col] = _csv_results[case_id]["actual"]
            row[result_col] = _csv_results[case_id]["result"]
            updated += 1

    # Write output CSV
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n[conftest] CSV results written to: {output_path}  ({updated} rows updated)")
