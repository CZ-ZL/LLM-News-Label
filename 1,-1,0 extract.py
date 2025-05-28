#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_labels.py
----
python sample_labels.py \
    --input news_raw.xlsx \
    --output news_sampled.xlsx \
    --label-col sentiment \
    --seed 2025
"""

from pathlib import Path
import pandas as pd
import argparse
import sys
import random

def read_table(path: Path) -> pd.DataFrame:

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, engine="openpyxl")
    elif suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    else:
        sys.exit(f"❌ not recognized：{suffix}（only Excel/CSV）")

def write_table(df: pd.DataFrame, path: Path) -> None:
    
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df.to_excel(path, index=False, engine="openpyxl")
    elif suffix in {".csv", ".txt"}:
        df.to_csv(path, index=False)
    else:
        sys.exit(f"❌ not recognized：{suffix}（only Excel/CSV）")

def main():
    parser = argparse.ArgumentParser(
        description="geting 1,-1 and 0"
    )
    parser.add_argument("--input", required=True, help="input files")
    parser.add_argument("--output", required=True, help="output files")
    parser.add_argument("--label-col", required=True, help="label column")
    parser.add_argument("--seed", type=int, default=42, help="set seed")
    parser.add_argument("--news-col", default="News", help="news column")
    args = parser.parse_args()

    random.seed(args.seed)
    df = read_table(Path(args.input))

    if args.label_col not in df.columns:
        sys.exit(f"❌ column '{args.label_col}' does not exist. here is the list in the file：{list(df.columns)}")
    if args.news_col not in df.columns:
        sys.exit(f"❌ column '{args.news_col}' does not exist. here is the list in the file：{list(df.columns)}")

   
    mask_pos_neg = df[args.label_col].isin([1, -1])
    pos_neg_df = df[mask_pos_neg]
    n_pos_neg = len(pos_neg_df)

   
    n_zero_sample = max(n_pos_neg // 2, 1)  
    zeros_df = df[df[args.label_col] == 0]
    if n_zero_sample > len(zeros_df):
        sys.exit("❌ not enough 0")

    zeros_sampled = zeros_df.sample(n=n_zero_sample, random_state=args.seed)

    
    final_df = pd.concat([pos_neg_df, zeros_sampled]).sample(
        frac=1, random_state=args.seed
    )

    
    final_df = final_df[[args.news_col, args.label_col]].rename(
        columns={args.news_col: "News", args.label_col: "Label"}
    )

    write_table(final_df, Path(args.output))
    print(f"✅ finished，{len(final_df)} written in {args.output}")

if __name__ == "__main__":
    main()
