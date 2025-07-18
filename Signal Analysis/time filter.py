import pandas as pd
import argparse
import os
import pandas_market_calendars as mcal
from datetime import time

def get_us_market_holidays(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    # 获取指定日期范围内的所有节假日（返回 DatetimeIndex）
    holidays = nyse.holidays().holidays
    # 如果 holidays 不是 DatetimeIndex，则转成 DatetimeIndex
    holidays = pd.DatetimeIndex(holidays)
    # 只保留在指定日期范围内的节假日
    holidays = holidays[(holidays >= pd.Timestamp(start_date)) & (holidays <= pd.Timestamp(end_date))]
    return holidays

def filter_news(input_file, time_column='Timestamp', start='09:00', end='16:30'):
    # Load data
    df = pd.read_excel(input_file)
    df[time_column] = pd.to_datetime(df[time_column])

    # Set trading time range
    start_time = pd.to_datetime(start).time()
    end_time = pd.to_datetime(end).time()

    # Get date range
    min_date = df[time_column].min().date()
    max_date = df[time_column].max().date()

    # Get US market holidays
    us_holidays = get_us_market_holidays(min_date, max_date)
    print("节假日列表：", us_holidays)

    # 先筛选出所有工作日
    workdays = df[df[time_column].dt.weekday < 5]
    print("所有工作日的日期和时间：")
    print(workdays[time_column].dt.strftime('%Y-%m-%d %A %H:%M:%S'))

    # Filtering logic:
    df_filtered = df[
        (df[time_column].dt.weekday < 5) &
        (df[time_column].dt.time >= start_time) &
        (df[time_column].dt.time <= end_time)
    ]

    # Output file name
    filename, ext = os.path.splitext(input_file)
    output_file = f"{filename}_{start.replace(':', '-')}-{end.replace(':', '-')}{ext}"

    # Save result
    df_filtered.to_excel(output_file, index=False)

    print(f"✅ Input file: {input_file}")
    print(f"🗓️ Time range: {start} - {end}, weekdays only, excluding US market holidays")
    print(f"📁 Output file: {output_file}")
    print(f"📰 Number of news retained: {len(df_filtered)}")
    print(df_filtered[time_column].dt.strftime('%Y-%m-%d %A'))

    print(df[df[time_column].isna()])
    #print(df[df[time_column].dt.strftime('%Y-%m-%d %H:%M:%S') == '2025-04-29 12:40:19'])

    workdays = df[df[time_column].dt.weekday < 5]
    in_time = workdays[
        (workdays[time_column].dt.time >= start_time) &
        (workdays[time_column].dt.time <= end_time)
    ]
    print(in_time[[time_column]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter news by trading time, weekday and US market holidays.")
    parser.add_argument("input_file", help="Path to the Excel file.")
    parser.add_argument("--time_column", default="Timestamp", help="Name of the time column.")
    parser.add_argument("--start", default="09:00", help="Start of trading time (HH:MM).")
    parser.add_argument("--end", default="16:30", help="End of trading time (HH:MM).")

    args = parser.parse_args()
    filter_news(args.input_file, args.time_column, args.start, args.end)
