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
            "Generate multiple confusion matrices + precision/recall/F1 from an Excel "
            "sheet, plot them, and append the results to new worksheets."
        )
    )
    p.add_argument("-f", "--file", required=True,
                   help="Path to the Excel workbook (.xlsx, .xlsm, …)")
    p.add_argument("-s", "--sheet", default=0,
                   help="Sheet name *or* index (default: first sheet)")
    p.add_argument("-actual-cols", required=True, nargs='+',
                   help="Column headers with the *true* labels (can specify multiple)")
    p.add_argument("-pred-cols", required=True, nargs='+',
                   help="Column headers with the *predicted* labels (can specify multiple)")

    p.add_argument("-normalize", choices=["true", "pred", "all", "none"],
                   default="none",
                   help="Normalise confusion matrix by rows (true), columns "
                        "(pred), all, or not at all (none). Default: none")

    p.add_argument("-savefig", nargs='+',
                   help="Save the confusion‑matrix plots with custom names (can specify multiple)")
    p.add_argument("-dpi", type=int, default=100,
                   help="DPI for saved figure (default: 100)")

    p.add_argument("-sheet-out-prefix", default="ConfusionMatrix",
                   help=("Prefix for worksheet names to write results to "
                         "(default: ConfusionMatrix_YYYYMMDD_HHMMSS_1, _2, etc.)"))
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


def generate_confusion_matrix(y_true, y_pred, labels, normalize_mode, savefig_path=None, 
                            dpi=100, matrix_index=1, total_matrices=1):
    """Generate a single confusion matrix with metrics and plotting."""
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = normalise(cm.astype(float), normalize_mode)

    # Metrics
    average_modes = ["micro", "macro", "weighted"]
    metrics = {
        "precision": {avg: precision_score(y_true, y_pred, average=avg)
                      for avg in average_modes},
        "recall":    {avg: recall_score(y_true, y_pred, average=avg)
                      for avg in average_modes},
        "f1":        {avg: f1_score(y_true, y_pred, average=avg)
                      for avg in average_modes},
    }

    # Print results
    print(f"\n=== Confusion Matrix {matrix_index}/{total_matrices} ===")
    print("Confusion Matrix (rows = true, cols = predicted):")
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

    # Plot matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format=".2f")
    plt.title(f"Confusion Matrix {matrix_index}/{total_matrices}")
    plt.tight_layout()

    # Save / show plot
    if savefig_path:
        # Create directory if it doesn't exist
        Path(savefig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savefig_path, dpi=dpi)
        print(f"📊  Figure saved to: {savefig_path}")
    else:
        plt.show()
    
    plt.close()  # Close the figure to free memory

    return cm, report_dict


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

    # Check if all columns exist
    all_required_cols = args.actual_cols + args.pred_cols
    missing_cols = set(all_required_cols) - set(df.columns)
    if missing_cols:
        sys.exit(f"💥  Column(s) not found: {', '.join(missing_cols)}")

    # Validate savefig arguments
    if args.savefig:
        if len(args.savefig) != len(args.actual_cols):
            sys.exit(f"💥  Number of savefig files ({len(args.savefig)}) must match "
                    f"number of column pairs ({len(args.actual_cols)})")

    # 2️⃣  Generate confusion matrices for each pair
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, (actual_col, pred_col) in enumerate(zip(args.actual_cols, args.pred_cols), 1):
        print(f"\n{'='*50}")
        print(f"Processing column pair {i}/{len(args.actual_cols)}: {actual_col} vs {pred_col}")
        print(f"{'='*50}")
        
        y_true = df[actual_col]
        y_pred = df[pred_col]
        
        # Get all unique labels
        labels = sorted(set(y_true) | set(y_pred))
        
        # Get savefig path for this matrix
        savefig_path = args.savefig[i-1] if args.savefig else None
        
        # Generate confusion matrix
        cm, report_dict = generate_confusion_matrix(
            y_true, y_pred, labels, args.normalize.lower(),
            savefig_path, args.dpi, i, len(args.actual_cols)
        )
        
        # 3️⃣  Write results back to Excel
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        report_df = pd.DataFrame(report_dict).T.round(3)
        
        sheet_out = f"{args.sheet_out_prefix}_{timestamp}_{i}"
        try:
            save_to_excel(args.file, cm_df, report_df, sheet_out, args.overwrite)
            print(f"📝  Results written to sheet: '{sheet_out}' in {args.file}")
        except ValueError as e:  # e.g. sheet already exists & overwrite=False
            sys.exit(f"💥  {e}")
    
    print(f"\n✅  Successfully generated {len(args.actual_cols)} confusion matrices!")


if __name__ == "__main__":
    main()
