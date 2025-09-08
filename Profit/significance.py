import csv
import random
import argparse
from typing import List

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

    true_count = 0
    for _ in range(args.trials):
        if single_pass(pnls):
            true_count += 1

    probability = true_count / args.trials
    print(f"Trials: {args.trials}")
    print(f"True results: {true_count}")
    print(f"Estimated probability trade history is NOT random: {probability:.6f}")

if __name__ == "__main__":
    main()
