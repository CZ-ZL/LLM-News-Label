#!/bin/bash

news_csv=("usdcnh-news-train.csv")
news_time_col=("date")
label_col=("competitor_label")
label=("Competitor")
fx_csv=("usdcnh-fx-train.csv")
fx_time_col=("date")
rate_col=("mid_price")
quote_convention=("USDCNY")
currency=("USDCNY")
trade_amount=10000
set -x
echo currency,label,hold_minutes,trade_amount,pnl,trades,pnl_per_trade,significance,confidence_level,mean_lower,mean_upper,bias_mean_lower,bias_mean_upper,sharpe,notes > process.csv
# OVERLAP="--allow_overlap"
OVERLAP=""
for i in 0; do
    #for hold_minutes in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150; do
    for hold_minutes in 5; do
#    ./forex_news_sim.py --news_csv 2024\ news\ file.csv --news_time_col Time --label_col "Expert Prompt Label" --fx_csv 2024\ Training\ Currency\ File.csv --fx_time_col Time --rate_col "Rate" --hold_minutes 150 --trade_amount_usd 10000 --initial_usd 1000000 --allow_overlap --quote_convention USDCNY
        echo $hold_minutes
        date
        ./forex_news_sim.py --news_csv "${news_csv[$i]}" --news_time_col "${news_time_col[$i]}" --label_col "${label_col[$i]}" --fx_csv "${fx_csv[$i]}" --fx_time_col "${fx_time_col[$i]}" --rate_col "${rate_col[$i]}" --hold_minutes $hold_minutes --trade_amount_usd $trade_amount --initial_usd 1000000 --quote_convention ${quote_convention[$i]} $OVERLAP > forex_out.txt
        python3 significance.py > significance_out.txt
        python3 mean_confidence_interval.py > mean_interval_out.txt
	python3 plot_time_of_day.py trades.csv entry_time pnl_usd --output "plots/${label[$i]}.${hold_minutes}.$OVERLAP.png"
        x=(`tail -1 forex_out.txt`)
        pnl=${x[5]}
        trades=${x[7]}
        x=(`tail -1 significance_out.txt`)
        significance=${x[9]}
	sharpe=${x[11]}
	total_daily_pnl=${x[13]}
	total_trade_days=${x[15]}
	annualized_return_percentage=${x[17]}
        x=(`tail -1 mean_interval_out.txt`)
        confidence_level=${x[2]}
        mean_lower=${x[10]}
        mean_upper=${x[12]}
        bias_mean_lower=${x[17]}
        bias_mean_upper=${x[19]}
	pnl_per_trade=$(echo "scale=4;$pnl/$trades"|bc)
        echo ${currency[$i]},${label[$i]},$hold_minutes,$trade_amount,$pnl,$trades,$pnl_per_trade,$significance,$confidence_level,$mean_lower,$mean_upper,$bias_mean_lower,$bias_mean_upper,$sharpe,$total_daily_pnl,$total_trade_days,$annualized_return_percentage,${news_csv[$i]} >> process.csv
    done
done
