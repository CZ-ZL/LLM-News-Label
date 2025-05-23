import pandas as pd
import numpy as np
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--fx_file', type=str, required=True, help='Path to the FX data file')
parser.add_argument('--news_file', type=str, required=True, help='Path to the news data file')
parser.add_argument('--spread', type=float, required=True, help='threshold = 3 * spread')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
# 解析命令行参数
args = parser.parse_args()

# 根据用户提交的文件路径读取数据
fx_df = pd.read_excel(args.fx_file)
news_df = pd.read_excel(args.news_file)

# Convert the relevant timestamp columns to datetime objects
fx_df['timestamp'] = pd.to_datetime(fx_df['Time'])
news_df['timestamp'] = pd.to_datetime(news_df['Time'])

# Sort both DataFrames by timestamp to ensure proper ordering
fx_df.sort_values('timestamp', inplace=True)
news_df.sort_values('timestamp', inplace=True)

# Lists to hold results for each news article
labels = []
trading_periods = []
rate_changes = []
window_starts = []
window_ends = []

# Process each news article one by one
for idx, news in news_df.iterrows():
    T = news['timestamp']
    window_end = T + pd.Timedelta(minutes=2)
    
    # Save the window boundaries
    window_starts.append(T)
    window_ends.append(window_end)
    
    # Get FX data for the 2-minute observation window [T, T + 2 minutes]
    fx_window = fx_df[(fx_df['timestamp'] >= T) & (fx_df['timestamp'] <= window_end)]
    print(f"Processing article at {T} (window ends at {window_end}), found {len(fx_window)} FX records")
    
    if fx_window.empty:
        labels.append("no_data")
        trading_periods.append(np.nan)
        rate_changes.append(np.nan)
        continue

    # Use the first available FX record in the window as the initial data point
    initial_record = fx_window.iloc[0]
    initial_rate = initial_record['Rate']
    # Here we use a fixed spread value of 0.0005 (modify as needed)
    threshold = 3 * args.spread
     
    final_record = fx_window.iloc[-1]
    final_rate = final_record['Rate']
    
    # Compute the change in the exchange rate
    r_change = final_rate - initial_rate
    rate_changes.append(r_change)
    
    # If the absolute rate change is less than the threshold, label as neutral
    if abs(r_change) < threshold:
        labels.append("0")
        trading_periods.append(np.nan)
        continue

    # Determine the label:
    # "positive" if the rate decreased (i.e. CNH strengthened),
    # "negative" if the rate increased (i.e. CNH weakened)
    label = "1" if r_change < 0 else "-1"
    
    # Find the first timestamp within the window where the rate change meets/exceeds the threshold
    tp = np.nan  # initialize trading period (in seconds)
    for _, row in fx_window.iterrows():
        current_rate = row['Rate']
        if abs(current_rate - initial_rate) >= threshold:
            tp = (row['timestamp'] - T).total_seconds()
            break

    labels.append(label)
    trading_periods.append(tp)
    print(f"Article at {T} labeled {label} with trading period {tp} seconds, rate change {r_change}")

# Create a new DataFrame with the required columns:
# - Time (from the original news file)
# - Content (the article's content)
# - Window_Start (start time of the 2-minute window)
# - Window_End (end time of the 2-minute window)
# - Rate_Change (the difference between the final and initial FX rate)
# - Trading_Period_sec (the trading period in seconds)
# - Label (the computed label)
news_df.drop(columns=['timestamp'], inplace=True)
news_df['Market Direction'] = labels

# Save the new DataFrame to a new Excel file
output_path = args.output_file
news_df.to_excel(output_path, index=False)
print(f"New Excel file '{output_path}' has been created with Time, Content, Window_Start, Window_End, Rate_Change, Trading_Period_sec, and Label.")
