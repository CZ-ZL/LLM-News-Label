import argparse
import csv
import matplotlib.pyplot as plt
from datetime import datetime

def get_hour(timestamp_str):
    """Extract hour (0–23) from timestamp string."""
    try:
        dt = datetime.fromisoformat(timestamp_str)  # expects ISO format like 2025-09-22 14:35:00
    except ValueError:
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return dt.hour


def plot_hourly_bar_from_csv(
    csv_file,
    time_column,
    value_column,
    title="Bar Chart: Hour of Day vs Value",
    xlabel="Hour of Day",
    ylabel="Total Value",
    output=None
):
    """
    Plot a bar chart grouped by hour of day.

    Args:
        csv_file (str): Path to the CSV file.
        time_column (str): Column with timestamps.
        value_column (str): Column with numeric values.
        title (str): Chart title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        output (str): File to save plot (if None, shows interactively).
    """
    hourly_totals = {h: 0 for h in range(24)}  # initialize all hours to 0

    try:
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if time_column not in reader.fieldnames or value_column not in reader.fieldnames:
                print(f"Columns '{time_column}' or '{value_column}' not found in CSV file.")
                return

            for row in reader:
                timestamp = row.get(time_column)
                val = row.get(value_column)
                if timestamp and val:
                    hour = get_hour(timestamp)
                    if hour is not None:
                        try:
                            hourly_totals[hour] += float(val)
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Plot bar chart
    hours = list(hourly_totals.keys())
    totals = list(hourly_totals.values())

    plt.bar(hours, totals, edgecolor="black")
    plt.xticks(range(0, 24))  # show all hours 0–23
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if output:
        plt.savefig(output)
        print(f"Bar chart saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot a bar chart grouped by hour of day from a CSV file.")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("time_column", help="Column containing timestamps")
    parser.add_argument("value_column", help="Column containing numeric values")
    parser.add_argument("--title", default="Bar Chart: Hour of Day vs Value", help="Title of the chart")
    parser.add_argument("--xlabel", default="Hour of Day", help="X-axis label")
    parser.add_argument("--ylabel", default="Total Value", help="Y-axis label")
    parser.add_argument("--output", default=None, help="Output file to save the plot (optional)")

    args = parser.parse_args()

    plot_hourly_bar_from_csv(
        csv_file=args.csv_file,
        time_column=args.time_column,
        value_column=args.value_column,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        output=args.output,
    )


if __name__ == "__main__":
    main()
