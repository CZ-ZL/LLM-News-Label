import argparse
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------------------------------- #
#                                CLI PARSER                                   #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a confusion matrix + precision/recall/F1 from an Excel "
            "sheet, plot it, and append the results to a new worksheet."
        )
    )
    p.add_argument("-f", "--file", required=True,
                   help="Path to the Excel workbook (.xlsx, .xlsm, …)")
    p.add_argument("-s", "--sheet", default=0,
                   help="Sheet name *or* index (default: first sheet)")
    p.add_argument("-actual-col", required=True,
                   help="Column header with the *true* labels")
    p.add_argument("-pred-col", required=True,
                   help="Column header with the *predicted* labels")

    p.add_argument("-normalize", choices=["true", "pred", "all", "none"],
                   default="none",
                   help="Normalise confusion matrix by rows (true), columns "
                        "(pred), all, or not at all (none). Default: none")

    p.add_argument("-savefig",
                   help="Save the confusion‑matrix plot instead of showing it")
    p.add_argument("-dpi", type=int, default=100,
                   help="DPI for saved figure (default: 100)")

    p.add_argument("-sheet-out",
                   help=("Name of worksheet to write results to "
                         "(default: ConfusionMatrix_YYYYMMDD_HHMMSS)"))
    p.add_argument("-overwrite", action="store_true",
                   help="If --sheet-out exists, overwrite it (otherwise fail)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#                             HELPER FUNCTIONS                                #
# --------------------------------------------------------------------------- #
def normalise(cm: np.ndarray, mode: str) -> np.ndarray:
    """Return cm normalized by rows, columns, or globally."""
    if mode == "true":
        return cm / cm.sum(axis=1, keepdims=True)
    if mode == "pred":
        return cm / cm.sum(axis=0, keepdims=True)
    if mode == "all":
        return cm / cm.sum()
    return cm  # "none"


def save_to_excel(
    wb_path: str,
    cm_df: pd.DataFrame,
    report_df: pd.DataFrame,
    sheet_name: str,
    overwrite: bool,
) -> None:
    """Append (or replace) confusion matrix & report in a workbook sheet."""
    mode = "a"  # append
    if_sheet_exists = "replace" if overwrite else "error"

    with pd.ExcelWriter(wb_path, mode=mode, engine="openpyxl",
                        if_sheet_exists=if_sheet_exists) as writer:
        cm_df.to_excel(writer, sheet_name=sheet_name, startrow=0, index=True)
        report_df.to_excel(writer, sheet_name=sheet_name,
                           startrow=len(cm_df) + 3, index=True)


# --------------------------------------------------------------------------- #
#                                   MAIN                                      #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    # 1️⃣  Load data
    try:
        df = pd.read_excel(args.file, sheet_name=args.sheet)
    except Exception as e:
        sys.exit(f"💥  Failed to read Excel file: {e}")

    if args.actual_col not in df.columns or args.pred_col not in df.columns:
        missing = {args.actual_col, args.pred_col} - set(df.columns)
        sys.exit(f"💥  Column(s) not found: {', '.join(missing)}")

    y_true = df[args.actual_col]
    y_pred = df[args.pred_col]

    # 2️⃣  Compute confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = normalise(cm.astype(float), args.normalize.lower())

    # 3️⃣  Metrics
    average_modes = ["micro", "macro", "weighted"]
    metrics = {
        "precision": {avg: precision_score(y_true, y_pred, average=avg)
                      for avg in average_modes},
        "recall":    {avg: recall_score(y_true, y_pred, average=avg)
                      for avg in average_modes},
        "f1":        {avg: f1_score(y_true, y_pred, average=avg)
                      for avg in average_modes},
    }

    # 4️⃣  Print results
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print("\nPer‑class metrics:")
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, digits=3
    )
    print(pd.DataFrame(report_dict).T.round(3))
    print("Aggregate metrics:")
    for m in metrics:
        print(f"  {m.capitalize():9s} "
              + " | ".join(f"{k}: {v:.3f}" for k, v in metrics[m].items()))
    print()

    # 5️⃣  Plot matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format=".2f")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # 6️⃣  Save / show plot
    if args.savefig:
        Path(args.savefig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.savefig, dpi=args.dpi)
        print(f"📊  Figure saved to: {args.savefig}")
    else:
        plt.show()

    # 7️⃣  Write results back to Excel
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    report_df = pd.DataFrame(report_dict).T.round(3)

    sheet_out = args.sheet_out or f"ConfusionMatrix_{datetime.now():%Y%m%d_%H%M%S}"
    try:
        save_to_excel(args.file, cm_df, report_df, sheet_out, args.overwrite)
        print(f"📝  Results written to sheet: '{sheet_out}' in {args.file}")
    except ValueError as e:  # e.g. sheet already exists & overwrite=False
        sys.exit(f"💥  {e}")


if __name__ == "__main__":
    main()
