import csv
import sys
from datetime import datetime

COMMON_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y%m%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d/%m/%Y",
    "%d/%m/%y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d-%b-%Y",      # 05-Aug-2025
    "%d-%B-%Y",      # 05-August-2025
    "%b %d %Y",      # Aug 05 2025
    "%b %d, %Y",     # Aug 05, 2025
    "%B %d %Y",      # August 05 2025
    "%B %d, %Y",     # August 05, 2025
]

def parse_date_auto(s: str) -> datetime:
    """Parse a wide range of common date strings into a datetime.
    Falls back to fromisoformat (supports 'YYYY-MM-DD' and timestamps).
    Raises ValueError if no format matches.
    """
    if s is None:
        raise ValueError("Empty date value")
    s = s.strip()
    if not s:
        raise ValueError("Empty date value")

    # Try Python's ISO parser first (handles 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS[.fff]', etc.)
    iso_candidate = s.replace("T", " ")
    if iso_candidate.endswith("Z"):
        # fromisoformat doesn't accept 'Z' → convert to '+00:00'
        iso_candidate = iso_candidate[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(iso_candidate)
    except Exception:
        pass

    # Try the common explicit formats
    for fmt in COMMON_DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue

    raise ValueError(f"Unrecognized date format: '{s}'")

def split_csv(encoding, input_file, column_name, split_date_str, output_file_leq, output_file_gt):
    # Parse the split date (auto-detected)
    split_date = parse_date_auto(split_date_str).date()

    with open(input_file, newline='', encoding=encoding) as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []

        if column_name not in fieldnames:
            raise ValueError(f"Column '{column_name}' not found in CSV file. "
                             f"Available columns: {fieldnames}")

        with open(output_file_leq, "w", newline='', encoding="utf-8") as f_leq, \
             open(output_file_gt, "w", newline='', encoding="utf-8") as f_gt:

            writer_leq = csv.DictWriter(f_leq, fieldnames=fieldnames)
            writer_gt = csv.DictWriter(f_gt, fieldnames=fieldnames)

            writer_leq.writeheader()
            writer_gt.writeheader()

            for i, row in enumerate(reader, start=2):  # start=2 accounts for header line as row 1
                raw_val = row.get(column_name, "")
                try:
                    row_date = parse_date_auto(raw_val).date()
                except ValueError as e:
                    raise ValueError(
                        f"Row {i}: cannot parse date in column '{column_name}': {e}"
                    ) from None

                if row_date <= split_date:
                    writer_leq.writerow(row)
                else:
                    try:
                        writer_gt.writerow(row)
                    except ValueError as e:
                        print(i)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python split_csv.py <input.csv> <column_name> <split_date> <output_leq.csv> <output_gt.csv>")
        print("Examples for <split_date>: 2025-09-24 | 09/24/2025 | Aug 24, 2025 | 20250924")
        sys.exit(1)

    input_file = sys.argv[1]
    column_name = sys.argv[2]
    split_date = sys.argv[3]
    output_file_leq = sys.argv[4]
    output_file_gt = sys.argv[5]

    import chardet

    with open(input_file, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(encoding)


    split_csv(encoding, input_file, column_name, split_date, output_file_leq, output_file_gt)
