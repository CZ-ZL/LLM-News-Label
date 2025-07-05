import pandas as pd
import numpy as np
import argparse

# ── 1. CLI arguments ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Label news with FX-movement ground truth (binary).')
parser.add_argument('--fx_file',      type=str,   required=True, help='Path to the FX data file')
parser.add_argument('--news_file',    type=str,   required=True, help='Path to the news data file')
parser.add_argument('--spread',       type=float, required=True, help='threshold = 3 * spread')
parser.add_argument('--t',            type=float, required=True, help='t-minute window')
parser.add_argument('--output_file',  type=str,   required=True, help='Path to the output file')
args = parser.parse_args()

# ── 2. Load & pre-process data ──────────────────────────────────────────────────
fx_df   = pd.read_excel(args.fx_file)
news_df = pd.read_excel(args.news_file)

fx_df['timestamp']   = pd.to_datetime(fx_df['Time'])
news_df['timestamp'] = pd.to_datetime(news_df['Time'])

fx_df.sort_values('timestamp', inplace=True)
news_df.sort_values('timestamp', inplace=True)

# ── 3. Core loop ────────────────────────────────────────────────────────────────
labels, tps, rate_changes = [], [], []

threshold = 3 * args.spread          # one calculation, reuse inside loop
window = pd.Timedelta(minutes=args.t)

for _, news in news_df.iterrows():
    T          = news['timestamp']
    window_end = T + window
    fx_slice   = fx_df[(fx_df['timestamp'] >= T) & (fx_df['timestamp'] <= window_end)]

    if fx_slice.empty:
        labels.append("no_data")
        tps.append(np.nan)
        rate_changes.append(np.nan)
        continue

    init_rate  = fx_slice.iloc[0]['Rate']
    final_rate = fx_slice.iloc[-1]['Rate']
    r_change   = final_rate - init_rate
    rate_changes.append(r_change)

    # ── 3a. Binary label: +1 if |Δrate| ≥ threshold, else 0 ────────────────────
    if abs(r_change) < threshold:
        labels.append("0")
        tps.append(np.nan)
        continue

    labels.append("1")

    # ── 3b. Trading-period (first hit of |Δrate| ≥ threshold) ──────────────────
    tp = np.nan
    for _, row in fx_slice.iterrows():
        if abs(row['Rate'] - init_rate) >= threshold:
            tp = (row['timestamp'] - T).total_seconds()
            break
    tps.append(tp)

# ── 4. Save results ─────────────────────────────────────────────────────────────
news_df.drop(columns=['timestamp'], inplace=True)
col_name = f't={args.t},spread={args.spread}_binary'
news_df[col_name] = labels
news_df.to_excel(args.output_file, index=False)

print(f"Finished. Binary-labelled file written to {args.output_file}")
