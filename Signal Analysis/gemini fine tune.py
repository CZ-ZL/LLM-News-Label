import pandas as pd
from datetime import timedelta
import json
import argparse
import os

def extract_to_jsonl(input_excel, output_jsonl,
                     timestamp_col='Time', news_col='News', label_col='Label'):
    # Load the Excel file
    df = pd.read_excel(input_excel)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)

    with open(output_jsonl, 'w', encoding='utf-8') as outfile:
        for _, row in df.iterrows():
            current_time = row[timestamp_col]
            window_start = current_time - timedelta(hours=1)
            
            # Select past news entries in the past hour (excluding the current row)
            past_df = df[(df[timestamp_col] >= window_start) & (df[timestamp_col] < current_time)]
            
            # Build the user content:
            # 1. Past news joined together with their labels
            # 2. Then the current news line (without its label)
            past_lines = [
                f"{r[timestamp_col].isoformat()} {r[news_col]} {r[label_col]}"
                for _, r in past_df.iterrows()
            ]
            current_line = f"{current_time.isoformat()} {row[news_col]}"
            user_lines = past_lines + [current_line]
            user_content = "\n".join(user_lines).strip()
            
            # The model content is the label for the current news row
            model_content = str(row[label_col])

            # Skip if no past news entries
            if not past_lines:
                continue
            
            # Build the JSONL record using the simplified messages schema
            record = {
                "messages": [
                    {"role": "user",  "content": user_content},
                    {"role": "model", "content": model_content}
                ]
            }
            # Write as a single JSON line
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract news context and labels from Excel into JSONL for fine-tuning."
    )
    parser.add_argument("input_excel", help="Path to the input Excel file")
    parser.add_argument("output_jsonl", nargs='?', default=None,
                        help="Path for the output JSONL file (defaults to input filename with .jsonl)")
    parser.add_argument("--timestamp_col", default="Time", help="Name of the timestamp column")
    parser.add_argument("--news_col", default="News", help="Name of the news text column")
    parser.add_argument("--label_col", default="Label", help="Name of the label column")
    
    args = parser.parse_args()

    # Derive default output filename if not provided
    if args.output_jsonl:
        output_path = args.output_jsonl
    else:
        base, _ = os.path.splitext(args.input_excel)
        output_path = base + ".jsonl"

    extract_to_jsonl(
        args.input_excel, output_path,
        timestamp_col=args.timestamp_col,
        news_col=args.news_col,
        label_col=args.label_col
    )

# Usage examples:
# python extract_news.py "news_data.xlsx"
#   -> writes "news_data.jsonl"
# python extract_news.py "news_data.xlsx" "output.jsonl"
