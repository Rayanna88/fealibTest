import csv
import re
from pathlib import Path

CSV_INPUT = Path.home() / "Downloads" / "OP单元测试用例.csv"
CSV_OUTPUT = Path.home() / "Downloads" / "OP单元测试用例.csv"
TEST_FILE = Path("/Users/shmimarui6/Desktop/Code/fealibTest/tests/unit/test_builtin_ops.py")

# Extract all case IDs from test_builtin_ops.py
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_content = f.read()

# Find all case IDs referenced in the test file
pattern = r'(TC-UNIT-[A-Z0-9]+-\d+-\d+)'
covered_ids = set(re.findall(pattern, test_content))

print(f"Covered test case IDs in test_builtin_ops.py: {len(covered_ids)}")
print(f"Sample: {sorted(list(covered_ids))[:5]}")

# Read the CSV
rows = []
with open(CSV_INPUT, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)

header = rows[0]
# Check if column already exists to avoid duplicates
if "是否在fealibTest覆盖" not in header:
    header.append("是否在fealibTest覆盖")

# Find the column index for 子用例编号
try:
    id_col = header.index("子用例编号")
except ValueError:
    id_col = 0

# Find the index of the coverage column if it already exists
coverage_col_idx = -1
if "是否在fealibTest覆盖" in header:
    coverage_col_idx = header.index("是否在fealibTest覆盖")

covered_count = 0
not_covered_count = 0

for row in rows[1:]:
    case_id = row[id_col].strip() if id_col < len(row) else ""
    is_covered = "是" if case_id in covered_ids else "否"
    if coverage_col_idx >= 0 and coverage_col_idx < len(row):
        # Update existing column
        row[coverage_col_idx] = is_covered
    else:
        # Add new column
        row.append(is_covered)
    if is_covered == "是":
        covered_count += 1
    else:
        not_covered_count += 1

# Write back to original CSV
with open(CSV_OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Covered (是): {covered_count}")
print(f"Not covered (否): {not_covered_count}")
print(f"Total: {covered_count + not_covered_count}")
print(f"Output written to: {CSV_OUTPUT}")