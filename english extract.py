#!/usr/bin/env python3
"""
Filter a JSONL file so only English-language rows survive.

Usage:
    python filter_english.py data.jsonl
    # or explicitly set output
    python filter_english.py data.jsonl -o clean.jsonl
"""

import argparse, json, pathlib, sys
from langdetect import detect, LangDetectException

def has_too_many_question_marks(text: str, max_questions: int = 15) -> bool:
    """检查文本中是否包含过多问号"""
    return text.count('?') > max_questions

def is_english(text: str) -> bool:
    """Return True iff langdetect tags the string as English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        # text too short / ambiguous
        return False

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Keep only English rows from a JSONL file.")
    p.add_argument("infile", help="Path to the input .jsonl")
    p.add_argument("-o", "--outfile",
                   help="Output path (default: '<infile stem> eng<suffix>')")
    p.add_argument("--max-questions", type=int, default=3,
                   help="Maximum number of question marks allowed (default: 3)")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    in_path  = pathlib.Path(args.infile)
    if not in_path.is_file():
        sys.exit(f"❌  Input file not found: {in_path}")

    out_path = pathlib.Path(args.outfile) if args.outfile else (
        in_path.parent / f"{in_path.stem} eng{in_path.suffix}"
    )

    total = kept = 0
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            record = json.loads(line)
            # adjust this accessor if your schema differs
            text = record.get("messages", [{}])[0].get("content", "")
            if is_english(text) and not has_too_many_question_marks(text, args.max_questions):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    print(f"✅  Kept {kept:,} English records out of {total:,}.")
    print(f"📄  Filtered file saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
