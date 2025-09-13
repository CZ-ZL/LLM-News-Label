#!/usr/bin/env python3
"""
Forex news-driven simulation between USD and CNY with LLM labels.

Core logic
----------
- Reads a "news labels" CSV (timestamps + label column(s))
- Reads an FX quotes CSV with bid/ask + timestamps OR exchange rate column
- For each signal:
  * label == +1  -> buy CNY (sell USD) for `trade_amount_usd`, hold N minutes, then convert back to USD
  * label == -1  -> sell CNY (short CNY / buy USD) with notional `trade_amount_usd`, hold N minutes, then cover using USD
  * label == 0   -> do nothing
- At the end, fully liquidate to USD.
- Outputs: trades CSV, equity curve CSV, summary TXT, and comparison results (CSV + PNG) for all modes.

Features
--------
- Single label column: Use --label_col for traditional single-label trading
- Multiple label columns: Use --label_cols with comma-separated list for multi-label comparison
- Multiple hold periods: Use --hold_minutes_list for hold period optimization
- Combined comparison: Use both --label_cols and --hold_minutes_list for comprehensive strategy evaluation
- Exchange rate trading: Use --rate_col instead of bid/ask for simplified pricing

Assumptions
-----------
- FX quotes are for one pair with consistent bid/ask. Default quote convention is "USDCNY"
  meaning price = CNY per 1 USD. You can change it with --quote_convention CNYUSD.
- You can use either bid/ask columns OR a single exchange rate column (--rate_col).
  When using rate_col, the same rate is used for both buy and sell operations.
- Multiple label columns can be compared by providing --label_cols with comma-separated column names.
- Multiple hold periods can be tested by providing --hold_minutes_list with comma-separated values.
- Combined comparison runs all label column × hold period combinations for comprehensive analysis.
- Timestamps are parsed with pandas.to_datetime without forcing timezone. Ensure your files
  use the same timezone or pre-align them.
- Matching price to signals supports methods: nearest (default), ffill, exact, with tolerance.
- No overlapping positions by default; signals during a hold are ignored (configurable).

Example
-------
# Single label column (original method):
python forex_news_sim.py \
  --news_csv news.csv --news_time_col time --label_col label \
  --fx_csv fx.csv --fx_time_col time --bid_col bid --ask_col ask \
  --hold_minutes 3 --trade_amount_usd 1000 --initial_usd 10000

# Multiple label columns comparison:
python forex_news_sim.py \
  --news_csv news.csv --news_time_col time \
  --label_cols "label1,label2,label3" \
  --fx_csv fx.csv --fx_time_col time --bid_col bid --ask_col ask \
  --hold_minutes 3 --trade_amount_usd 1000 --initial_usd 10000

# Multiple hold periods comparison:
python forex_news_sim.py \
  --news_csv news.csv --news_time_col time --label_col label \
  --fx_csv fx.csv --fx_time_col time --bid_col bid --ask_col ask \
  --hold_minutes_list "1,2,3,4,5,15" --trade_amount_usd 1000 --initial_usd 10000

# Combined comparison (multiple labels × multiple hold periods):
python forex_news_sim.py \
  --news_csv news.csv --news_time_col time \
  --label_cols "model1,model2,model3" \
  --fx_csv fx.csv --fx_time_col time --bid_col bid --ask_col ask \
  --hold_minutes_list "1,3,5,10" --trade_amount_usd 1000 --initial_usd 10000

# Using exchange rate column (no need for bid/ask):
python forex_news_sim.py \
  --news_csv news.csv --news_time_col time --label_col label \
  --fx_csv fx.csv --fx_time_col time --rate_col exchange_rate \
  --hold_minutes 3 --trade_amount_usd 1000 --initial_usd 10000
"""

# 导入必要的Python库和模块
import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 定义持仓数据类，用于存储交易信息
@dataclass
class Position:
    side: str  # 'LONG_CNY' or 'SHORT_CNY'
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    size_usd: float  # fixed USD notional for each trade
    entry_bid: float
    entry_ask: float
    exit_bid: Optional[float] = None
    exit_ask: Optional[float] = None
    cny_amount: Optional[float] = None  # For LONG_CNY: +CNY bought; For SHORT_CNY: -CNY borrowed
    pnl_usd: Optional[float] = None


# 解析命令行参数，设置程序运行的各种参数
def parse_args():
    p = argparse.ArgumentParser(description="Simulate USD/CNY trading from LLM news labels.")
    # 新闻数据相关参数
    p.add_argument("--news_csv", required=True, help="Path to labels/news file (CSV or Excel: .csv/.xlsx/.xls)")
    p.add_argument("--news_time_col", required=True, help="Column name with news timestamps")
    p.add_argument("--label_col", default=None, help="Column name with label values (+1/0/-1 by default)")
    p.add_argument("--label_cols", default=None, help="Comma-separated list of multiple label columns (e.g., 'col1,col2,col3')")
    p.add_argument("--signal_value_long", type=str, default="1", help="Value in label_col that means LONG CNY (default '1')")
    p.add_argument("--signal_value_short", type=str, default="-1", help="Value in label_col that means SHORT CNY (default '-1')")
    p.add_argument("--signal_value_flat", type=str, default="0", help="Value in label_col that means do nothing (default '0')")

    # 外汇数据相关参数
    p.add_argument("--fx_csv", required=True, help="Path to FX quotes file (CSV or Excel: .csv/.xlsx/.xls)")
    p.add_argument("--fx_time_col", required=True, help="Column name with FX timestamps")
    p.add_argument("--bid_col", default=None, help="Column name with Bid (required if not using --rate_col)")
    p.add_argument("--ask_col", default=None, help="Column name with Ask (required if not using --rate_col)")
    p.add_argument("--rate_col", default=None, help="Column name with exchange rate (if provided, will use this instead of bid/ask)")
    p.add_argument("--quote_convention", choices=["USDCNY", "CNYUSD"], default="USDCNY",
                   help="If USDCNY: price is CNY per 1 USD. If CNYUSD: price is USD per 1 CNY. Default USDCNY")

    # 模拟参数设置
    p.add_argument("--initial_usd", type=float, default=1000000.0, help="Starting USD cash (default 1,000,000)")
    p.add_argument("--trade_amount_usd", type=float, default=10000.0, help="USD notional per trade (default 10,000)")
    p.add_argument("--hold_minutes", type=int, default=3, help="Hold duration in minutes (default 3)")
    p.add_argument("--hold_minutes_list", default=None,
                   help="Comma-separated hold durations to compare, e.g. '1,2,3,4,5,15'. If provided, outputs only comparison files.")
    p.add_argument("--match_method", choices=["nearest", "ffill", "exact"], default="nearest",
                   help="How to match prices to timestamps (default nearest)")
    p.add_argument("--time_tolerance_secs", type=int, default=90,
                   help="Max allowed gap between signal time and matched price (seconds). Default 90s")
    p.add_argument("--allow_overlap", action="store_true", help="Allow entering a new trade while one is open (default False)")

    # Excel表格名称参数（CSV文件时忽略）
    p.add_argument("--news_sheet", default=None, help="Excel sheet name for news file (if Excel)")
    p.add_argument("--fx_sheet", default=None, help="Excel sheet name for FX file (if Excel)")

    # 输出文件参数
    p.add_argument("--trades_csv", default="trades.csv", help="Output CSV for trades (default trades.csv)")
    p.add_argument("--equity_csv", default="equity.csv", help="Output CSV for equity curve (default equity.csv)")
    p.add_argument("--equity_plot", default="equity.png", help="Output image for equity curve (default equity.png)")
    p.add_argument("--summary_txt", default="summary.txt", help="Output text file for summary stats (default summary.txt)")
    p.add_argument("--cmp_csv", default="comparison.csv", help="Output CSV for all comparison results (multi-label, multi-hold, or combined)")
    p.add_argument("--cmp_plot", default="comparison.png", help="Output image for all comparison equity curves")

    args = p.parse_args()
    
    # 验证参数组合
    if args.rate_col is None and (args.bid_col is None or args.ask_col is None):
        p.error("Either --rate_col OR both --bid_col and --ask_col must be provided")
    
    if args.rate_col is not None and (args.bid_col is not None or args.ask_col is not None):
        print("Warning: --rate_col is provided, --bid_col and --ask_col will be ignored")
    
    return args


# 读取表格文件（支持CSV和Excel格式）
def read_tabular(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read a tabular file that may be CSV or Excel.
    - CSV/.txt -> pandas.read_csv
    - .xlsx/.xls -> pandas.read_excel with optional sheet_name
    Fallback: try CSV first, then Excel.
    """
    lower = path.lower()
    try:
        if lower.endswith((".csv", ".txt")):
            return pd.read_csv(path)
        if lower.endswith((".xlsx", ".xls")):
            # If no sheet specified, default to first sheet (0)
            sn = 0 if sheet_name is None else sheet_name
            df = pd.read_excel(path, sheet_name=sn)
            if isinstance(df, dict):
                # If a dict is returned (multiple sheets), take the first sheet by order
                first_key = next(iter(df.keys()))
                return df[first_key]
            return df
        # Unknown extension: try CSV then Excel (first sheet)
        try:
            return pd.read_csv(path)
        except Exception:
            sn = 0 if sheet_name is None else sheet_name
            df = pd.read_excel(path, sheet_name=sn)
            if isinstance(df, dict):
                first_key = next(iter(df.keys()))
                return df[first_key]
            return df
    except Exception:
        # Re-raise with clearer message
        raise


# 加载新闻标签数据，处理时间戳和标签列
def load_news(args) -> pd.DataFrame:
    df = read_tabular(args.news_csv, sheet_name=args.news_sheet)
    if args.news_time_col not in df.columns:
        raise ValueError(f"news_time_col '{args.news_time_col}' not in news file")
    if args.label_col not in df.columns and args.label_cols is None:
        raise ValueError(f"label_col '{args.label_col}' not in news file")

    # 如果指定了多个标签列，合并它们
    if args.label_cols:
        label_cols = [c.strip() for c in args.label_cols.split(',') if c.strip()]
        if not label_cols:
            raise ValueError("label_cols is provided but no columns are specified.")
        df = df[label_cols].copy()
        df.columns = label_cols # 确保列名与标签值匹配
    else:
        df = df[[args.news_time_col, args.label_col]].copy()

    df[args.news_time_col] = pd.to_datetime(df[args.news_time_col])
    # Normalize labels to string for matching; preserve original for reference
    df["_label_raw"] = df[args.label_col] if args.label_col else df.iloc[:, 0] # 使用第一个标签列作为原始标签
    df["_label_str"] = df[args.label_col] if args.label_col else df.iloc[:, 0].astype(str) # 使用第一个标签列作为字符串标签
    df = df.sort_values(args.news_time_col).reset_index(drop=True)
    return df


# 加载外汇报价数据，处理时间戳、买价和卖价列
def load_fx(args) -> pd.DataFrame:
    df = read_tabular(args.fx_csv, sheet_name=args.fx_sheet)
    
    # 如果提供了汇率列，使用汇率列替代bid/ask
    if args.rate_col is not None:
        if args.rate_col not in df.columns:
            raise ValueError(f"rate_col '{args.rate_col}' not in FX file")
        
        # 使用汇率列创建bid和ask列（都设为相同的汇率值）
        df = df[[args.fx_time_col, args.rate_col]].copy()
        df[args.bid_col] = df[args.rate_col]
        df[args.ask_col] = df[args.rate_col]
        print(f"{args.rate_col=}")
        print(f"{args.bid_col=}")
        print(f"{args.ask_col=}")
        print(f"{args.fx_time_col=}")
    else:
        # 原有的bid/ask逻辑
        for c in [args.fx_time_col, args.bid_col, args.ask_col]:
            if c not in df.columns:
                raise ValueError(f"'{c}' not in FX file")
        
        df = df[[args.fx_time_col, args.bid_col, args.ask_col]].copy()
        
        # Remove obviously bad rows
        df = df[np.isfinite(df[args.bid_col]) & np.isfinite(df[args.ask_col])]
        # Ensure bid <= ask
        bad = df[df[args.bid_col] > df[args.ask_col]]
        if not bad.empty:
            # Swap where bad if it's likely mislabeled, else drop
            swapped = bad.copy()
            df.loc[bad.index, [args.bid_col, args.ask_col]] = swapped[[args.ask_col, args.bid_col]].values

    df[args.fx_time_col] = pd.to_datetime(df[args.fx_time_col])
    df = df.sort_values(args.fx_time_col).reset_index(drop=True)
    
    return df


# 根据时间匹配外汇价格，支持多种匹配方法
def match_price_at(df_fx: pd.DataFrame, t: pd.Timestamp, args, side: str) -> Optional[Tuple[pd.Timestamp, float, float]]:
    """
    Match a (bid, ask) quote to desired timestamp t.
    Returns (matched_time, bid, ask) or None if tolerance fail.
    """
    times = df_fx[args.fx_time_col].values
    bid = df_fx[args.bid_col].values
    ask = df_fx[args.ask_col].values

    if args.match_method == "exact":
        mask = df_fx[args.fx_time_col] == t
        if not mask.any():
            return None
        row = df_fx.loc[mask].iloc[-1]
        return (row[args.fx_time_col], float(row[args.bid_col]), float(row[args.ask_col]))

    elif args.match_method == "ffill":
        # last known at or before t
        idx = df_fx[args.fx_time_col].searchsorted(t, side="right") - 1
        if idx < 0:
            return None
        row = df_fx.iloc[idx]
        # Check tolerance (how far back)
        dt = (t - row[args.fx_time_col]).total_seconds()
        if dt > args.time_tolerance_secs:
            return None
        return (row[args.fx_time_col], float(row[args.bid_col]), float(row[args.ask_col]))

    else:  # nearest
        idx = df_fx[args.fx_time_col].searchsorted(t)
        candidates = []
        if idx > 0:
            candidates.append(df_fx.iloc[idx - 1])
        if idx < len(df_fx):
            candidates.append(df_fx.iloc[idx])
        # pick the closer one
        if not candidates:
            print(f"no match candidates {t=}")
            return None
        best = min(candidates, key=lambda r: abs((r[args.fx_time_col] - t).total_seconds()))
        if abs((best[args.fx_time_col] - t).total_seconds()) > args.time_tolerance_secs:
            print(f"no match tolerance_secs {t=}")
            return None
        return (best[args.fx_time_col], float(best[args.bid_col]), float(best[args.ask_col]))


# 将美元转换为人民币，根据报价约定使用正确的买价或卖价
def usd_to_cny(usd: float, bid: float, ask: float, convention: str) -> float:
    """Convert USD to CNY using correct bid/ask given quote convention."""
    if convention == "USDCNY":
        # Selling USD -> receive CNY at dealer's USD bid
        return usd * bid
    else:  # CNYUSD (USD per 1 CNY)
        # Buying CNY with USD; dealer sells CNY at ask (USD per CNY). CNY = USD / ask
        return usd / ask


# 将人民币转换为美元，根据报价约定使用正确的买价或卖价
def cny_to_usd(cny: float, bid: float, ask: float, convention: str) -> float:
    """Convert CNY to USD using correct bid/ask given quote convention."""
    if convention == "USDCNY":
        # Selling CNY (buying USD); dealer sells USD at ask (CNY per USD). USD = CNY / ask
        return cny / ask
    else:  # CNYUSD
        # Dealer buys CNY at bid (USD per CNY). USD = CNY * bid
        return cny * bid


# 主要的模拟函数，执行外汇新闻驱动的交易模拟
def simulate(args) -> Tuple[pd.DataFrame, pd.DataFrame, List[Position], dict]:
    # 加载新闻和外汇数据
    news = load_news(args)
    fx = load_fx(args)

    # 初始化现金和持仓变量
    usd_cash = float(args.initial_usd)
    cny_cash = 0.0
    # Open positions (can hold multiple when allow_overlap=True)
    open_positions: List[Position] = []
    # Closed positions
    positions: List[Position] = []
    equity_points = []  # (time, equity_usd)

    # 诊断统计变量
    diag_total_signals = int(len(news))
    diag_unknown_label = 0
    diag_flat_signals = 0
    diag_actionable_signals = 0  # +1 or -1
    diag_overlap_skipped = 0
    diag_time_out_of_range = 0
    diag_entry_match_fail = 0
    diag_entry_matched = 0
    diag_insufficient_usd = 0

    # 获取外汇数据的时间范围
    fx_time_min = fx[args.fx_time_col].min()
    fx_time_max = fx[args.fx_time_col].max()
    tol_delta = pd.Timedelta(seconds=int(args.time_tolerance_secs))

    # 计算指定时间的权益值（包括未平仓头寸的市值）
    def equity_at(t: pd.Timestamp) -> Optional[float]:
        # Mark-to-market using nearest quote (mid) at time t
        m = match_price_at(fx, t, args, side="mark")
        if m is None:
            return None
        _, bid, ask = m
        mid = (bid + ask) / 2.0
        # Include open position CNY exposures (long positive, short negative)
        total_cny_exposure = cny_cash + sum((p.cny_amount or 0.0) for p in open_positions)
        if args.quote_convention == "USDCNY":
            # USD value of CNY at mid: USD = CNY / mid
            cny_usd = total_cny_exposure / mid
        else:  # CNYUSD
            # USD = CNY * mid
            cny_usd = total_cny_exposure * mid
        return usd_cash + cny_usd

    # 尝试平仓到期的持仓
    def try_close_due_positions(cutoff_time: pd.Timestamp):
        nonlocal usd_cash, cny_cash
        # Close any open positions whose exit_time <= cutoff_time
        still_open: List[Position] = []
        for pos in open_positions:
            if pos.exit_time <= cutoff_time:
                m_exit = match_price_at(fx, pos.exit_time, args, side="exit")
                if m_exit is None:
                    last = fx.iloc[-1]
                    m_exit = (last[args.fx_time_col], float(last[args.bid_col]), float(last[args.ask_col]))
                mt, ebid, eask = m_exit
                pos.exit_bid = ebid
                pos.exit_ask = eask
                if pos.side == "LONG_CNY":
                    # Convert position CNY back to USD
                    usd_recv = cny_to_usd(pos.cny_amount, ebid, eask, args.quote_convention)
                    usd_cash += usd_recv
                    # No need to modify cny_cash since position CNY was never in cny_cash
                    pos.pnl_usd = usd_recv - pos.size_usd
                else:  # SHORT_CNY
                    cny_borrowed = -pos.cny_amount
                    if args.quote_convention == "USDCNY":
                        usd_needed = cny_borrowed / ebid
                    else:
                        usd_needed = cny_borrowed * eask
                    usd_cash -= usd_needed
                    pos.pnl_usd = pos.size_usd - usd_needed
                positions.append(pos)
                # Equity point at close
                eqc = equity_at(mt)
                if eqc is not None:
                    equity_points.append((mt, float(eqc)))
            else:
                still_open.append(pos)
        open_positions[:] = still_open

    # 主循环：处理每个新闻信号
    for idx, row in news.iterrows():
        t_signal = row[args.news_time_col]
        lab = str(row["_label_str"])

        # 检查标签是否有效
        if lab not in {args.signal_value_long, args.signal_value_short, args.signal_value_flat}:
            # Unknown label: skip
            print(f"unknown label {lab=}")
            diag_unknown_label += 1
            continue

        # 处理中性信号（不做任何交易）
        if lab == args.signal_value_flat:
            # record equity point and continue
            eq = equity_at(t_signal)
            if eq is not None:
                equity_points.append((t_signal, float(eq)))
            diag_flat_signals += 1
            continue

        # Before acting on new signal, close any due positions up to this signal time
        try_close_due_positions(t_signal)

        # If not allowing overlap and still have open positions, skip this signal
        if (not args.allow_overlap) and open_positions:
            diag_overlap_skipped += 1
            continue

        # Now evaluate new signal
        diag_actionable_signals += 1

        # Check if signal time lies within FX time range (+/- tolerance)
        if (t_signal < (fx_time_min - tol_delta)) or (t_signal > (fx_time_max + tol_delta)):
            diag_time_out_of_range += 1
            continue

        # 尝试匹配入场价格
        m_entry = match_price_at(fx, t_signal, args, side="entry")
        if m_entry is None:
            # can't execute; skip
            diag_entry_match_fail += 1
            continue
        t_quote, bid, ask = m_entry
        diag_entry_matched += 1
        exit_time = t_signal + pd.Timedelta(minutes=args.hold_minutes)

        # 处理做多人民币信号
        if lab == args.signal_value_long:
            # Buy CNY with USD
            if usd_cash < args.trade_amount_usd:
                # Not enough USD cash; skip
                diag_insufficient_usd += 1
                continue
            cny_bought = usd_to_cny(args.trade_amount_usd, bid, ask, args.quote_convention)
            usd_cash -= args.trade_amount_usd
            # Don't add to cny_cash - it's already tracked in the position
            open_positions.append(Position(
                side="LONG_CNY",
                entry_time=t_quote,
                exit_time=exit_time,
                size_usd=float(args.trade_amount_usd),
                entry_bid=float(bid),
                entry_ask=float(ask),
                cny_amount=float(cny_bought),
            ))
        # 处理做空人民币信号
        elif lab == args.signal_value_short:
            # Short CNY / Buy USD: borrow CNY, convert to USD now, repay later
            # Choose borrowed CNY so that USD_received == trade_amount_usd at entry
            if args.quote_convention == "USDCNY":
                cny_borrow = args.trade_amount_usd * ask  # buy USD with CNY at ask
                usd_received = cny_borrow / ask
            else:  # CNYUSD
                cny_borrow = args.trade_amount_usd / bid  # buy USD with CNY at bid (USD per CNY)
                usd_received = cny_borrow * bid

            usd_cash += usd_received  # should equal trade_amount_usd
            open_positions.append(Position(
                side="SHORT_CNY",
                entry_time=t_quote,
                exit_time=exit_time,
                size_usd=float(args.trade_amount_usd),
                entry_bid=float(bid),
                entry_ask=float(ask),
                cny_amount=float(-cny_borrow),  # negative indicates short (borrowed)
            ))

        # Equity snapshot at entry
        eq = equity_at(t_quote)
        if eq is not None:
            equity_points.append((t_quote, float(eq)))

    # 平仓所有剩余的未平仓头寸
    if open_positions:
        # Ensure all positions with exit_time after last quote are closed using last available price fallback
        for pos in sorted(open_positions, key=lambda p: p.exit_time):
            m_exit = match_price_at(fx, pos.exit_time, args, side="final_exit")
            if m_exit is None:
                last = fx.iloc[-1]
                m_exit = (last[args.fx_time_col], float(last[args.bid_col]), float(last[args.ask_col]))
            mt, ebid, eask = m_exit
            pos.exit_bid = ebid
            pos.exit_ask = eask
            if pos.side == "LONG_CNY":
                # Convert position CNY back to USD
                usd_recv = cny_to_usd(pos.cny_amount, ebid, eask, args.quote_convention)
                usd_cash += usd_recv
                # No need to modify cny_cash since position CNY was never in cny_cash
                pos.pnl_usd = usd_recv - pos.size_usd
            else:
                cny_borrowed = -pos.cny_amount
                if args.quote_convention == "USDCNY":
                    usd_needed = cny_borrowed / ebid
                else:
                    usd_needed = cny_borrowed * eask
                usd_cash -= usd_needed
                pos.pnl_usd = pos.size_usd - usd_needed
            positions.append(pos)
            eqc = equity_at(mt)
            if eqc is not None:
                equity_points.append((mt, float(eqc)))
        open_positions = []

    # 最后，将剩余的人民币现金转换为美元
    # Note: cny_cash should be 0.0 now since all CNY is tracked in positions
    last = fx.iloc[-1]
    last_bid, last_ask = float(last[args.bid_col]), float(last[args.ask_col])
    if cny_cash != 0.0:
        if cny_cash > 0:
            usd_recv = cny_to_usd(cny_cash, last_bid, last_ask, args.quote_convention)
            usd_cash += usd_recv
            cny_cash = 0.0
        else:
            # Negative CNY cash (shouldn't happen) -> need to buy CNY to flat
            cny_needed = -cny_cash
            if args.quote_convention == "USDCNY":
                usd_needed = cny_needed / last_bid
            else:
                usd_needed = cny_needed * last_ask
            usd_cash -= usd_needed
            cny_cash = 0.0

    # 构建交易数据框
    if positions:
        trades = pd.DataFrame([{
            "side": p.side,
            "entry_time": p.entry_time,
            "exit_time": p.exit_time,
            "size_usd": p.size_usd,
            "entry_bid": p.entry_bid,
            "entry_ask": p.entry_ask,
            "exit_bid": p.exit_bid,
            "exit_ask": p.exit_ask,
            "cny_amount": p.cny_amount,
            "pnl_usd": p.pnl_usd,
        } for p in positions])
    else:
        trades = pd.DataFrame(columns=["side","entry_time","exit_time","size_usd",
                                       "entry_bid","entry_ask","exit_bid","exit_ask","cny_amount","pnl_usd"])

    # 构建权益曲线数据
    if equity_points:
        equity_df = pd.DataFrame(equity_points, columns=["time", "equity_usd"]).sort_values("time")
        # Ensure the final point reflects final equity (after full liquidation)
        final_equity = float(usd_cash)  # cny_cash should be 0 now
        equity_df = pd.concat([equity_df, pd.DataFrame([{"time": last[args.fx_time_col], "equity_usd": final_equity}])])
        equity_df = equity_df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    else:
        equity_df = pd.DataFrame([{"time": last[args.fx_time_col], "equity_usd": float(usd_cash)}])

    # 计算汇总统计信息
    total_pnl = float(trades["pnl_usd"].sum()) if not trades.empty else 0.0
    n_trades = int(len(trades))
    win_rate = float((trades["pnl_usd"] > 0).mean()) if n_trades > 0 else np.nan
    avg_pnl = float(trades["pnl_usd"].mean()) if n_trades > 0 else np.nan
    max_drawdown = compute_max_drawdown(equity_df["equity_usd"].values)

    # 构建汇总字典
    summary = {
        "initial_usd": float(args.initial_usd),
        "final_usd": float(usd_cash),
        "total_pnl_usd": total_pnl,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_pnl_usd": avg_pnl,
        "max_drawdown_usd": max_drawdown,
        # Diagnostics
        "diag_total_signals": diag_total_signals,
        "diag_unknown_label": diag_unknown_label,
        "diag_flat_signals": diag_flat_signals,
        "diag_actionable_signals": diag_actionable_signals,
        "diag_overlap_skipped": diag_overlap_skipped,
        "diag_time_out_of_range": diag_time_out_of_range,
        "diag_entry_match_fail": diag_entry_match_fail,
        "diag_entry_matched": diag_entry_matched,
        "diag_insufficient_usd": diag_insufficient_usd,
    }

    return trades, equity_df, positions, summary


# 计算最大回撤
def compute_max_drawdown(series: np.ndarray) -> float:
    if series.size == 0:
        return 0.0
    peak = -np.inf
    mdd = 0.0
    for x in series:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > mdd:
            mdd = dd
    return float(mdd)


# 保存输出文件，包括交易记录、权益曲线图表和汇总报告
def save_outputs(trades: pd.DataFrame, equity_df: pd.DataFrame, summary: dict, args):
    trades.to_csv(args.trades_csv, index=False)
    equity_df.to_csv(args.equity_csv, index=False)

    # 绘制权益曲线图
    plt.figure()
    plt.plot(equity_df["time"], equity_df["equity_usd"])
    plt.title("Equity Curve (USD)")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    plt.tight_layout()
    plt.savefig(args.equity_plot, dpi=150)
    plt.close()

    # 保存汇总报告
    with open(args.summary_txt, "w", encoding="utf-8") as f:
        f.write("=== Simulation Summary ===\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")


# 主函数，程序入口点
def main():
    args = parse_args()
    
    # 如果用户同时提供了多个标签列和多个持仓时间，运行组合比较
    if args.label_cols and args.hold_minutes_list:
        print("检测到多标签列和多持仓时间，开始组合比较...")
        
        # 解析标签列和持仓时间列表
        label_columns = [col.strip() for col in args.label_cols.split(',') if col.strip()]
        hold_parts = [p.strip() for p in str(args.hold_minutes_list).split(',') if p.strip()]
        hold_list = [int(p) for p in hold_parts]
        
        print(f"标签列: {label_columns}")
        print(f"持仓时间: {hold_list}")
        print(f"总共需要运行 {len(label_columns) * len(hold_list)} 次模拟\n")
        
        rows = []
        curves: List[Tuple[str, int, pd.DataFrame]] = []
        
        # 对每个标签列和持仓时间的组合运行模拟
        for label_col in label_columns:
            for hold_minutes in hold_list:
                print(f"正在处理: 标签列={label_col}, 持仓时间={hold_minutes}分钟")
                
                # 克隆参数以避免副作用
                args_i = argparse.Namespace(**vars(args))
                args_i.label_col = label_col
                args_i.label_cols = None  # 避免递归调用
                args_i.hold_minutes = hold_minutes
                args_i.hold_minutes_list = None  # 避免递归调用
                
                # 运行模拟
                trades_i, equity_df_i, positions_i, summary_i = simulate(args_i)
                curves.append((label_col, hold_minutes, equity_df_i))
                
                # 准备结果行
                row = {
                    "label_column": label_col,
                    "hold_minutes": hold_minutes
                }
                row.update(summary_i)
                rows.append(row)
                
                print(f"  完成: 最终USD: {summary_i['final_usd']:.2f}, 总盈亏: {summary_i['total_pnl_usd']:.2f}, 交易次数: {summary_i['n_trades']}")
        
        # 保存组合比较结果CSV
        cmp_df = pd.DataFrame(rows).sort_values(["total_pnl_usd", "final_usd"], ascending=[False, False])
        cmp_df.to_csv(args.cmp_csv, index=False)
        
        # 绘制组合权益曲线图
        plt.figure(figsize=(14, 10))
        
        # 按标签列分组绘制
        for label_col in label_columns:
            for hold_minutes in hold_list:
                # 找到对应的权益数据
                for curve_label, curve_hold, eqdf in curves:
                    if curve_label == label_col and curve_hold == hold_minutes:
                        plt.plot(eqdf["time"], eqdf["equity_usd"], 
                                label=f"{label_col} (t={hold_minutes})", 
                                linewidth=1.5, alpha=0.8)
                        break
        
        plt.title("多标签列和多持仓时间组合权益曲线对比", fontsize=14)
        plt.xlabel("时间", fontsize=12)
        plt.ylabel("权益 (USD)", fontsize=12)
        plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.cmp_plot, dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n组合比较完成！")
        print(f"比较结果CSV -> {args.cmp_csv}")
        print(f"比较结果图表 -> {args.cmp_plot}")
        
        if not cmp_df.empty:
            best = cmp_df.iloc[0]
            print(f"\n最佳组合:")
            print(f"标签列: {best['label_column']}, 持仓时间: {int(best['hold_minutes'])}分钟")
            print(f"最终USD: {best['final_usd']:.2f} | 总盈亏: {best['total_pnl_usd']:.2f} | 交易次数: {int(best['n_trades'])}")
            
            # 显示前10名结果
            print(f"\n前10名组合结果:")
            for i, (_, row) in enumerate(cmp_df.head(10).iterrows()):
                print(f"{i+1}. {row['label_column']} (t={int(row['hold_minutes'])}): "
                      f"最终USD {row['final_usd']:.2f}, 盈亏 {row['total_pnl_usd']:.2f}, 交易 {int(row['n_trades'])}")
            
            # 按标签列分组显示最佳持仓时间
            print(f"\n每个标签列的最佳持仓时间:")
            for label_col in label_columns:
                label_results = cmp_df[cmp_df['label_column'] == label_col]
                if not label_results.empty:
                    best_for_label = label_results.iloc[0]
                    print(f"  {label_col}: 最佳持仓时间 {int(best_for_label['hold_minutes'])}分钟, "
                          f"盈亏 {best_for_label['total_pnl_usd']:.2f}")
    
    # 如果用户提供了多个标签列，运行标签比较
    elif args.label_cols:
        # 解析标签列列表
        label_columns = [col.strip() for col in args.label_cols.split(',') if col.strip()]
        
        rows = []
        curves: List[Tuple[str, pd.DataFrame]] = []
        
        # 对每个标签列运行模拟
        for label_col in label_columns:
            print(f"正在处理标签列: {label_col}")
            
            # 克隆参数以避免副作用
            args_i = argparse.Namespace(**vars(args))
            args_i.label_col = label_col
            args_i.label_cols = None  # 避免递归调用
            
            # 运行模拟
            trades_i, equity_df_i, positions_i, summary_i = simulate(args_i)
            curves.append((label_col, equity_df_i))
            
            # 准备结果行
            row = {"label_column": label_col}
            row.update(summary_i)
            rows.append(row)
            
            print(f"  完成 {label_col}: 最终USD: {summary_i['final_usd']:.2f}, 总盈亏: {summary_i['total_pnl_usd']:.2f}, 交易次数: {summary_i['n_trades']}")
        
        # 保存比较结果CSV
        cmp_df = pd.DataFrame(rows).sort_values(["total_pnl_usd", "final_usd"], ascending=[False, False])
        cmp_df.to_csv(args.cmp_csv, index=False)
        
        # 绘制组合权益曲线图
        plt.figure(figsize=(12, 8))
        for label_col, eqdf in curves:
            plt.plot(eqdf["time"], eqdf["equity_usd"], label=f"标签列: {label_col}", linewidth=2)
        
        plt.title("多标签列权益曲线对比", fontsize=14)
        plt.xlabel("时间", fontsize=12)
        plt.ylabel("权益 (USD)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.cmp_plot, dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n多标签列比较完成！")
        print(f"比较结果CSV -> {args.cmp_csv}")
        print(f"比较结果图表 -> {args.cmp_plot}")
        
        if not cmp_df.empty:
            best = cmp_df.iloc[0]
            print(f"\n最佳标签列: {best['label_column']}")
            print(f"最终USD: {best['final_usd']:.2f} | 总盈亏: {best['total_pnl_usd']:.2f} | 交易次数: {int(best['n_trades'])}")
            
            # 显示所有标签列的结果排名
            print("\n所有标签列结果排名:")
            for i, (_, row) in enumerate(cmp_df.iterrows()):
                print(f"{i+1}. {row['label_column']}: 最终USD {row['final_usd']:.2f}, 盈亏 {row['total_pnl_usd']:.2f}, 交易 {int(row['n_trades'])}")
    
    # 如果用户提供了持仓时间列表，运行批量比较
    elif args.hold_minutes_list:
        # Parse list like "1,2,3,4,5,15"
        parts = [p.strip() for p in str(args.hold_minutes_list).split(',') if p.strip()]
        hold_list: List[int] = [int(p) for p in parts]

        rows = []
        curves: List[Tuple[int, pd.DataFrame]] = []

        # 对每个持仓时间运行模拟
        for hm in hold_list:
            # Clone args to avoid side-effects
            args_i = argparse.Namespace(**vars(args))
            args_i.hold_minutes = int(hm)
            trades_i, equity_df_i, positions_i, summary_i = simulate(args_i)
            curves.append((hm, equity_df_i))

            row = {"t_minutes": int(hm)}
            row.update(summary_i)
            rows.append(row)

        # 保存比较结果CSV
        cmp_df = pd.DataFrame(rows).sort_values(["total_pnl_usd", "final_usd"], ascending=[False, False])
        cmp_df.to_csv(args.cmp_csv, index=False)

        # 绘制组合权益曲线图
        plt.figure()
        for hm, eqdf in curves:
            plt.plot(eqdf["time"], eqdf["equity_usd"], label=f"t={hm}")
        plt.title("Equity Curves by Hold Minutes")
        plt.xlabel("Time")
        plt.ylabel("Equity (USD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.cmp_plot, dpi=150)
        plt.close()

        print("Done multi-hold comparison.")
        print(f"Comparison CSV -> {args.cmp_csv}")
        print(f"Comparison Plot -> {args.cmp_plot}")
        if not cmp_df.empty:
            best = cmp_df.iloc[0]
            print(
                f"Best t={int(best['t_minutes'])} | Final USD: {best['final_usd']:.2f} | Total PnL: {best['total_pnl_usd']:.2f} | Trades: {int(best['n_trades'])}"
            )
    else:
        # 运行单次模拟
        trades, equity_df, positions, summary = simulate(args)
        save_outputs(trades, equity_df, summary, args)
        print("Done.")
        print(f"Trades -> {args.trades_csv}")
        print(f"Equity -> {args.equity_csv}")
        print(f"Plot   -> {args.equity_plot}")
        print(f"Summary-> {args.summary_txt}")
        print(f"Final USD: {summary['final_usd']:.2f} Total PnL: {summary['total_pnl_usd']:.2f} Trades: {summary['n_trades']}")

# 程序入口点
if __name__ == "__main__":
    main()
