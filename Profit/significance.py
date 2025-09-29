import math
import csv
import random
import argparse
from typing import List
from datetime import datetime
from collections import defaultdict, OrderedDict

def read_pnl_usd(filename: str) -> List[float]:
    """
    Read pnl_usd values from a CSV with headers:
    side,entry_time,exit_time,size_usd,entry_bid,entry_ask,exit_bid,exit_ask,cny_amount,pnl_usd
    Returns a list of floats (one per trade).
    """
    pnls = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "pnl_usd" not in reader.fieldnames:
            raise ValueError("CSV is missing required 'pnl_usd' header.")
        for row in reader:
            # Skip empty/malformed rows gracefully
            val = row.get("pnl_usd", "").strip()
            if val == "":
                continue
            pnls.append(float(val))
    if not pnls:
        raise ValueError("No pnl_usd values found in the file.")
    return pnls

def min_max_datetime(filename: str):
    """
    Read date values from a CSV with headers:
    side,entry_time,exit_time,size_usd,entry_bid,entry_ask,exit_bid,exit_ask,cny_amount,pnl_usd
    Returns min/max
    """
    entry_times = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "entry_time" not in reader.fieldnames:
            raise ValueError("CSV is missing required 'entry_time' header.")
        for row in reader:
            # Skip empty/malformed rows gracefully
            val = row.get("entry_time", "").strip()
            if val == "":
                continue
            entry_times.append(datetime.fromisoformat(val))

    if not entry_times:
        raise ValueError("No entry_times values found in the file.")
    return (min(entry_times), max(entry_times))

def single_pass(pnls: List[float]) -> bool:
    """
    One pass over the trades:
    - random_pnl_usd starts at 0
    - trade_pnl_usd starts at 0
    - For each pnl in pnls, flip a fair coin:
        heads  -> random += pnl
        tails  -> random -= pnl
      Always add pnl to trade_pnl_usd.
    - Return True if trade_pnl_usd > random_pnl_usd, else False.
    """
    random_pnl_usd = 0.0
    trade_pnl_usd = 0.0
    for pnl in pnls:
        if random.random() < 0.5:   # heads
            random_pnl_usd += pnl
        else:                        # tails
            random_pnl_usd -= pnl
        trade_pnl_usd += pnl
    return trade_pnl_usd > random_pnl_usd

def read_trades(csv_path):
    """
    Returns a list of (datetime, pnl_usd) tuples sorted by time.
    Expects headers including 'datetime' and 'pnl_usd'.
    """
    trades = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'entry_time' not in reader.fieldnames or 'pnl_usd' not in reader.fieldnames:
            raise ValueError("CSV must contain headers 'entry_time' and 'pnl_usd'.")

        for i, row in enumerate(reader, start=2):
            dt_raw = row['entry_time'].strip()
            pnl_raw = row['pnl_usd'].strip()
            if not dt_raw:
                # skip blank line
                continue
            try:
                # Flexible parse: try common formats; fall back to fromisoformat
                try:
                    dt = datetime.fromisoformat(dt_raw)
                except ValueError:
                    # Try a couple of other typical patterns
                    for fmt in ("%Y-%m-%d %H:%M:%S",
                                "%Y/%m/%d %H:%M:%S",
                                "%m/%d/%Y %H:%M",
                                "%Y-%m-%d"):
                        try:
                            dt = datetime.strptime(dt_raw, fmt)
                            break
                        except ValueError:
                            dt = None
                    if dt is None:
                        raise
                pnl = float(pnl_raw)
            except Exception as e:
                raise ValueError(f"Parse error on line {i}: {e}")

            trades.append((dt, pnl))

    trades.sort(key=lambda x: x[0])
    return trades

def group_daily_pnl(trades):
    """
    Aggregates P&L per calendar date.
    Returns an OrderedDict: date -> total_pnl_usd (float),
    ordered by date ascending.
    """
    day_totals = defaultdict(float)
    for dt, pnl in trades:
        day = dt.date()
        day_totals[day] += pnl

    # Order by date
    ordered = OrderedDict(sorted(day_totals.items(), key=lambda kv: kv[0]))
    return ordered

def compute_daily_returns(daily_pnl_by_date, initial_equity=1_000_000.0):
    """
    Given an OrderedDict {date: daily_pnl}, compute daily returns and ending equity.

    Return: list of dicts:
        [{"date": date, "daily_return": r, "equity_end": eq_end, "daily_pnl": pnl}, ...]
    where daily_return = daily_pnl / equity_start_of_that_day.
    """
    if initial_equity <= 0:
        raise ValueError("Initial equity must be positive.")

    results = []
    equity = float(initial_equity)

    for day, pnl in daily_pnl_by_date.items():
        equity_start = equity
        daily_return = pnl / equity_start
        equity_end = equity_start + pnl
        results.append({
            "date": day.isoformat(),
            "daily_return": daily_return,
            "equity_end": equity_end,
            "daily_pnl": pnl
        })
        equity = equity_end

    return results

def calculate_sharpe_ratio(returns, risk_free_rate=0.0001557, annualization_factor=1):
    """
    Calculates the Sharpe Ratio for a series of returns.

    Args:
        returns (list): A list of numerical returns (e.g., daily returns).
        risk_free_rate (float): The risk-free rate of return. Defaults to daily rate that compounds to 4% annually (approximately)
        annualization_factor (int): The factor to annualize the Sharpe Ratio.
                                   For daily returns, use 252 (trading days).
                                   For monthly returns, use 12. Defaults to 1 (no annualization).

    Returns:
        float: The calculated Sharpe Ratio.
    """
    if not returns:
        return 0.0  # Handle empty returns list

    # Calculate average return
    sum_returns = 0
    for r in returns:
        sum_returns += r
    average_return = sum_returns / len(returns)

    # Calculate excess returns
    excess_returns = []
    for r in returns:
        excess_returns.append(r - risk_free_rate)

    # Calculate standard deviation of excess returns
    if len(excess_returns) < 2:
        return 0.0  # Cannot calculate standard deviation with less than 2 data points

    sum_squared_diff = 0
    for er in excess_returns:
        sum_squared_diff += (er - (sum(excess_returns) / len(excess_returns))) ** 2
    standard_deviation = math.sqrt(sum_squared_diff / (len(excess_returns) - 1))

    # Calculate Sharpe Ratio
    if standard_deviation == 0:
        return 0.0  # Avoid division by zero

    sharpe_ratio = (average_return - risk_free_rate) / standard_deviation

    # Annualize the Sharpe Ratio
    return sharpe_ratio * math.sqrt(annualization_factor)

# Example usage with daily returns and a risk-free rate
#daily_returns = [0.01, -0.005, 0.02, -0.01, 0.008, 0.015, -0.002]
#risk_free_rate_daily = 0.0001  # Example daily risk-free rate
#annualization_factor_daily = 252  # For daily returns


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap probability that the trade history is not random."
    )
    parser.add_argument(
        "--file", "-f", default="trades.csv",
        help="Path to trades CSV (default: trades.csv)"
    )
    parser.add_argument(
        "--trials", "-n", type=int, default=100000,
        help="Number of bootstrap single-pass trials (default: 100000)"
    )
    args = parser.parse_args()

    pnls = read_pnl_usd(args.file)
    print(pnls[0:3])
#    (min_datetime, max_datetime) = min_max_datetime(args.file)
    dr = compute_daily_returns(group_daily_pnl(read_trades(args.file)))
    print(dr[0:3])
    total_daily_pnl = sum([ i['daily_pnl'] for i in dr])
    total_trade_days = len(dr)
    annualized_return_percentage = 100*((total_daily_pnl * (252.0/float(total_trade_days)))/1_000_000.0)
    
    true_count = 0
    for _ in range(args.trials):
        if single_pass(pnls):
            true_count += 1

    probability = true_count / args.trials
    sr = calculate_sharpe_ratio([ i['daily_return'] for i in dr ], risk_free_rate=0.0,annualization_factor=len(dr))

    print(f"Trials: {args.trials}")
    print(f"True results: {true_count}")
    print(f"Estimated probability trade history is NOT random: {probability:.3f} significance: {1-probability:.3f} sharpe: {sr:.3f} total_daily_pnl: {total_daily_pnl:.2f} total_trade_days: {total_trade_days} annualized_return_percentage: {annualized_return_percentage:.3f}")

if __name__ == "__main__":
    main()
