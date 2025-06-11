"""
sentiment_classifier.py
───────────────────────
Given an Excel file of historical news (timestamp, headline, label) and one or more
"fresh" headlines, this script:

1.  Fetches the N most-recent labelled headlines from <now – lookback_hours>
2.  Builds a prompt:  (system msg) + (examples) + (new headline)
3.  Calls Gemini to return one of {-1, 0, 1}
4.  Appends results to an Excel workbook

Run:  python sentiment_classifier.py --news_db news.xlsx \
        --input_file todays_news.xlsx --output_file labelled.xlsx
"""

import argparse, os, sys, json
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparse

import pandas as pd
import google.generativeai as genai
from openpyxl import load_workbook


# ──────────────────────────── CONFIG KNOBS ────────────────────────────── #

DEFAULT_MODEL_ID    = "gemini-1.5-flash-32k"  # Default model to use
LOOKBACK_HOURS     = 1                     # window for context retrieval
N_EXAMPLES         = 30                     # max labelled rows passed to LLM
MAX_OUTPUT_TOKENS  = 1                     # we only need "-1", "0", or "1"
SYSTEM_PROMPT      = (
    
)

# ────────────────────────── UTILITY FUNCTIONS ─────────────────────────── #

def parse_iso(ts) -> datetime:
    """Handle both string timestamps and existing Timestamp objects."""
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    if isinstance(ts, datetime):
        return ts
    dt = dtparse.parse(str(ts))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def build_prompt(past_rows: pd.DataFrame, new_headline: str) -> str:
    """Assemble the prompt for Gemini API."""
    examples = "\n".join(
        f"{r.Time}: [{r.Label}] {r.News}"
        for r in past_rows.itertuples(index=False)
    ) or "None in the last hour."
    
    prompt = (
        f"Recent labelled headlines:\n{examples}\n\n"
        f"Now classify the sentiment of this headline:\n{new_headline}\n\n"
        "Respond with exactly one number: -1 (bearish), 0 (neutral), or 1 (bullish). "
        "Do not include any other text or explanation."
    )
    return prompt

def call_gemini(prompt: str, model_id: str) -> int:
    """Send prompt to Gemini and get sentiment label."""
    model = genai.GenerativeModel(model_id)
    try:
        resp = model.generate_content(
            prompt,
            generation_config=dict(max_output_tokens=MAX_OUTPUT_TOKENS)
        )
        return int(resp.text.strip())
    except ValueError:
        # Fallback: heuristic mapping
        txt = resp.text.lower()
        if "bear" in txt or "-" in txt:
            return -1
        if "bull" in txt or "+" in txt:
            return 1
        return 0
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return 0  # 发生错误时返回中性标签

def safe_read_excel(file_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """Safely read Excel file"""
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")

def safe_write_excel(df: pd.DataFrame, file_path: str, sheet_name: str = "labels"):
    """Safely write to Excel file"""
    try:
        if os.path.exists(file_path):
            # If file exists, read existing data first
            existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
            # Merge data
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Write merged data
            combined_df.to_excel(file_path, sheet_name=sheet_name, index=False)
        else:
            # If file doesn't exist, write directly
            df.to_excel(file_path, sheet_name=sheet_name, index=False)
    except Exception as e:
        raise Exception(f"Error writing to Excel file: {str(e)}")

# ──────────────────────────────  DRIVER  ──────────────────────────────── #

def main(args):
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("❌ Set GOOGLE_API_KEY env-var first!")

    # Load historical database and normalize timestamps
    db = safe_read_excel(args.news_db)
    # 检查时间列的类型并转换
    if not isinstance(db["Time"].iloc[0], (pd.Timestamp, datetime)):
        db["Time"] = db["Time"].apply(parse_iso)
    else:
        db["Time"] = db["Time"].apply(lambda x: x.to_pydatetime() if isinstance(x, pd.Timestamp) else x)

    # Load incoming headlines (may be one or many)
    fresh = safe_read_excel(args.input_file)
    # 检查时间列的类型并转换
    if not isinstance(fresh["Time"].iloc[0], (pd.Timestamp, datetime)):
        fresh["Time"] = fresh["Time"].apply(parse_iso)
    else:
        fresh["Time"] = fresh["Time"].apply(lambda x: x.to_pydatetime() if isinstance(x, pd.Timestamp) else x)

    results = []
    for row in fresh.itertuples(index=False):
        window_start = row.Time - timedelta(hours=args.lookback)
        ctx = (
            db[(db["Time"] >= window_start) &
               (db["Time"] <  row.Time)]
            .sort_values("Time", ascending=False)
            .head(args.examples)
        )
        prompt = build_prompt(ctx, row.News)
        try:
            label = call_gemini(prompt, args.model_id)
            print(f"Processing: {row.News[:60]:60s} → {label}")
        except Exception as e:
            print(f"Error processing headline: {row.News[:60]}")
            print(f"Error details: {str(e)}")
            label = 0  # 发生错误时使用中性标签
        results.append(
            dict(Time=row.Time.isoformat(),
                 News=row.News,
                 Label=label)
        )

    # Write/append to Excel
    out_df = pd.DataFrame(results)
    safe_write_excel(out_df, args.output_file)
    print(f"✅ Saved {len(results)} rows to {args.output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--news_db",     required=True,
                   help="Excel file containing Time,News,Label columns")
    p.add_argument("--input_file",  required=True,
                   help="Excel file containing Time,News columns")
    p.add_argument("--output_file", required=True,
                   help="Excel workbook to create/append")
    p.add_argument("--lookback",    type=int, default=LOOKBACK_HOURS,
                   help="Hours to look back for context retrieval")
    p.add_argument("--examples",    type=int, default=N_EXAMPLES,
                   help="Maximum number of historical rows to feed to LLM")
    p.add_argument("--model_id",    type=str, default=DEFAULT_MODEL_ID,
                   help="Gemini model ID to use (e.g., gemini-1.5-flash-32k, gemini-1.5-pro)")
    main(p.parse_args())
