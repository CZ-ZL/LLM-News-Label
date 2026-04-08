import argparse
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_CSV_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "latin1"]
UINT32_MOD = 2**32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a FinBERT-style classifier to label news sentiment across multiple runs with per-run/per-label seeds."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Path to a fine-tuned sequence-classification model directory.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV/Excel file containing the text column.")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path. If .xlsx, writes labels + run_summary + config sheets.")
    parser.add_argument("--text_column", type=str, default="News", help="Name of the text column.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for per_run mode.")
    parser.add_argument("--label_column", type=str, default="FinBERT_label", help="Base name for numeric label columns.")
    parser.add_argument("--label_name_column", type=str, default="FinBERT_label_name", help="Base name for string label columns.")
    parser.add_argument("--pred_id_column", type=str, default="FinBERT_pred_id", help="Base name for raw predicted class id columns.")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of full repeated runs.")
    parser.add_argument("--max_length", type=int, default=512, help="Tokenizer max_length.")
    parser.add_argument("--csv_encoding", type=str, default=None, help="Optional encoding override for CSV input/output.")
    parser.add_argument("--force_dropout_inference", action="store_true", help="Enable dropout during inference so seeds can affect outputs.")
    parser.add_argument("--seed_mode", type=str, choices=["per_label", "per_run"], default="per_label", help="Seed once per run or derive a new seed for each label generation.")
    parser.add_argument("--base_seed", type=int, default=None, help="Optional starting seed. If omitted, Unix time in seconds is used.")
    parser.add_argument("--save_per_run_files", action="store_true", help="Also save one separate file per run.")
    return parser.parse_args()


def load_dataframe(path: str, csv_encoding: Optional[str] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        encodings = [csv_encoding] if csv_encoding else DEFAULT_CSV_ENCODINGS
        last_error = None
        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Unable to read CSV with encodings {encodings}: {last_error}")
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file extension: {ext}. Use .csv or .xlsx/.xls.")


def save_dataframe(df: pd.DataFrame, path: str, csv_encoding: Optional[str] = None) -> str:
    ext = os.path.splitext(path)[1].lower()
    output_path = path
    outdir = os.path.dirname(output_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    if ext == ".csv":
        df.to_csv(output_path, index=False, encoding=csv_encoding or "utf-8-sig")
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(output_path, index=False)
    else:
        output_path = path + ".csv"
        df.to_csv(output_path, index=False, encoding=csv_encoding or "utf-8-sig")
    return output_path


def save_excel_workbook(path: str, labels_df: pd.DataFrame, run_summary_df: pd.DataFrame, config_df: pd.DataFrame) -> str:
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        labels_df.to_excel(writer, index=False, sheet_name="labels")
        run_summary_df.to_excel(writer, index=False, sheet_name="run_summary")
        config_df.to_excel(writer, index=False, sheet_name="config")
    return path


def add_suffix_to_filename(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return base + suffix + ext


def derive_output_paths(input_file: str, output_file: Optional[str]) -> Tuple[str, str, str]:
    if output_file:
        combined_path = output_file
    else:
        combined_path = add_suffix_to_filename(input_file, "_multirun_labeled")
    combined = Path(combined_path)
    runs_dir = str(combined.with_name(f"{combined.stem}_runs"))
    summary_path = str(combined.with_name(f"{combined.stem}_run_summary.csv"))
    return combined_path, runs_dir, summary_path


def get_unix_seed(offset: int = 0) -> int:
    return int(time.time()) + int(offset)


def generate_run_base_seeds(num_runs: int, base_seed: Optional[int] = None) -> List[int]:
    start_seed = int(base_seed) if base_seed is not None else get_unix_seed()
    return [start_seed + i for i in range(num_runs)]


def normalize_seed_for_rng(seed: int) -> int:
    return int(seed) % UINT32_MOD


def set_all_seeds(seed: int) -> int:
    actual_seed = normalize_seed_for_rng(seed)
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed)
    return actual_seed


def normalize_label_name(label_name: str) -> str:
    return label_name.strip().lower().replace("-", "_").replace(" ", "_")


def build_numeric_mapping(model) -> Tuple[Dict[int, str], Dict[int, Optional[int]]]:
    raw_id2label = getattr(model.config, "id2label", None) or {}
    id2label: Dict[int, str] = {}
    for k, v in raw_id2label.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            idx = k
        id2label[idx] = str(v)

    num_labels = getattr(model.config, "num_labels", len(id2label) or None)
    numeric_mapping: Dict[int, Optional[int]] = {}

    if id2label:
        for idx, label_name in id2label.items():
            normalized = normalize_label_name(label_name)
            if "negative" in normalized or normalized == "neg":
                numeric_mapping[idx] = -1
            elif "positive" in normalized or normalized == "pos":
                numeric_mapping[idx] = 1
            elif "neutral" in normalized or normalized == "neu":
                numeric_mapping[idx] = 0
            else:
                numeric_mapping[idx] = None
        if all(v is not None for v in numeric_mapping.values()):
            return id2label, numeric_mapping

    if num_labels == 2:
        return id2label or {0: "negative", 1: "positive"}, {0: -1, 1: 1}
    if num_labels == 3:
        return id2label or {0: "negative", 1: "neutral", 2: "positive"}, {0: -1, 1: 0, 2: 1}

    raise ValueError(
        "Could not infer a safe numeric label mapping from model.config.id2label. "
        "Please inspect the model labels before converting to numeric sentiment values."
    )


def looks_like_generic_classifier_labels(id2label: Dict[int, str]) -> bool:
    if not id2label:
        return True
    values = {str(v).strip().upper() for v in id2label.values()}
    return all(v.startswith("LABEL_") for v in values)


def prepare_model_mode(model, force_dropout_inference: bool) -> None:
    if force_dropout_inference:
        model.train()
    else:
        model.eval()


def predict_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    force_dropout_inference: bool,
) -> List[int]:
    prepare_model_mode(model, force_dropout_inference)
    all_pred_ids: List[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_pred_ids.extend(preds)
    return all_pred_ids


def predict_single(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    force_dropout_inference: bool,
) -> int:
    prepare_model_mode(model, force_dropout_inference)
    with torch.no_grad():
        enc = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        pred = int(torch.argmax(outputs.logits, dim=-1).cpu().item())
    return pred


def run_inference_for_one_run(
    texts: List[str],
    source_row_ids: List[int],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    force_dropout_inference: bool,
    seed_mode: str,
    base_seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if seed_mode == "per_run":
        actual_seed = set_all_seeds(base_seed)
        preds = predict_batch(texts, tokenizer, model, device, batch_size, max_length, force_dropout_inference)
        requested_seeds = [base_seed] * len(texts)
        used_seeds = [actual_seed] * len(texts)
        return preds, requested_seeds, used_seeds

    preds: List[int] = []
    requested_seeds: List[int] = []
    used_seeds: List[int] = []
    for text, source_row_id in zip(texts, source_row_ids):
        requested_seed = int(base_seed) + int(source_row_id)
        used_seed = set_all_seeds(requested_seed)
        pred = predict_single(text, tokenizer, model, device, max_length, force_dropout_inference)
        preds.append(pred)
        requested_seeds.append(requested_seed)
        used_seeds.append(used_seed)
    return preds, requested_seeds, used_seeds


def make_run_filename(input_file: str, run_idx: int, seed: int, runs_dir: str) -> str:
    input_path = Path(input_file)
    return str(Path(runs_dir) / f"{input_path.stem}_run_{run_idx:02d}_base_seed_{seed}{input_path.suffix}")


def make_config_df(args: argparse.Namespace, id2label: Dict[int, str], numeric_mapping: Dict[int, Optional[int]]) -> pd.DataFrame:
    rows = [
        ("model_dir", args.model_dir),
        ("input_file", args.input_file),
        ("output_file", args.output_file or ""),
        ("text_column", args.text_column),
        ("num_runs", args.num_runs),
        ("seed_mode", args.seed_mode),
        ("base_seed", args.base_seed if args.base_seed is not None else ""),
        ("force_dropout_inference", args.force_dropout_inference),
        ("batch_size", args.batch_size),
        ("max_length", args.max_length),
        ("id2label", str(id2label)),
        ("numeric_mapping", str(numeric_mapping)),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def main() -> None:
    args = parse_args()
    if args.num_runs <= 0:
        raise ValueError("--num_runs must be >= 1")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")
    if args.max_length <= 0:
        raise ValueError("--max_length must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_dataframe(args.input_file, csv_encoding=args.csv_encoding)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in file. Columns: {list(df.columns)}")

    combined_df = df.copy()
    combined_df["source_row_id"] = range(len(combined_df))
    texts = combined_df[args.text_column].fillna("").astype(str).tolist()
    source_row_ids = combined_df["source_row_id"].tolist()

    config = AutoConfig.from_pretrained(args.model_dir)
    print(f"Loaded config from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    id2label, numeric_mapping = build_numeric_mapping(model)
    print(f"Detected label mapping: {id2label}")
    print(f"Numeric sentiment mapping: {numeric_mapping}")

    if looks_like_generic_classifier_labels(id2label):
        warnings.warn(
            "The model labels look generic (e.g. LABEL_0/LABEL_1). This often means you loaded a base model or a poorly exported checkpoint. "
            "For real sentiment labeling, prefer your fine-tuned sequence-classification output directory, not a pretraining/base directory.",
            stacklevel=1,
        )

    combined_output_path, runs_dir, summary_path = derive_output_paths(args.input_file, args.output_file)
    if args.save_per_run_files:
        os.makedirs(runs_dir, exist_ok=True)

    print(
        "Important: standard classifier eval mode is usually deterministic. Different seeds only create actual output variation when some stochastic mechanism is active."
    )
    if args.seed_mode == "per_label":
        print("Seed mode is per_label: each label generation gets a derived seed based on run base seed + source_row_id.")
    else:
        print("Seed mode is per_run: each full run shares one seed.")
    if args.force_dropout_inference:
        print("Dropout is enabled during inference, so different seeds may produce different outputs.")
    else:
        print("Dropout is NOT enabled during inference, so repeated runs may remain identical even if seeds differ.")

    run_records = []
    base_seeds = generate_run_base_seeds(args.num_runs, base_seed=args.base_seed)

    for run_idx, base_seed in enumerate(base_seeds, start=1):
        print(f"Starting run {run_idx}/{args.num_runs} with base_seed={base_seed}")
        pred_ids, requested_seeds, used_seeds = run_inference_for_one_run(
            texts=texts,
            source_row_ids=source_row_ids,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            force_dropout_inference=args.force_dropout_inference,
            seed_mode=args.seed_mode,
            base_seed=base_seed,
        )

        label_names = [id2label.get(pred_id, str(pred_id)) for pred_id in pred_ids]
        numeric_labels = [numeric_mapping.get(pred_id) for pred_id in pred_ids]

        pred_id_col = f"{args.pred_id_column}_run_{run_idx:02d}"
        label_name_col = f"{args.label_name_column}_run_{run_idx:02d}"
        numeric_label_col = f"{args.label_column}_run_{run_idx:02d}"
        seed_requested_col = f"seed_requested_run_{run_idx:02d}"
        seed_used_col = f"seed_used_run_{run_idx:02d}"

        combined_df[pred_id_col] = pred_ids
        combined_df[label_name_col] = label_names
        combined_df[numeric_label_col] = numeric_labels
        combined_df[seed_requested_col] = requested_seeds
        combined_df[seed_used_col] = used_seeds

        label_counts = pd.Series(numeric_labels).value_counts(dropna=False).to_dict()
        run_records.append(
            {
                "run": run_idx,
                "base_seed": base_seed,
                "min_seed_requested": min(requested_seeds) if requested_seeds else None,
                "max_seed_requested": max(requested_seeds) if requested_seeds else None,
                "min_seed_used": min(used_seeds) if used_seeds else None,
                "max_seed_used": max(used_seeds) if used_seeds else None,
                "unique_numeric_labels": sorted(pd.Series(numeric_labels).dropna().unique().tolist()),
                "count_neg1": int(label_counts.get(-1, 0)),
                "count_0": int(label_counts.get(0, 0)),
                "count_1": int(label_counts.get(1, 0)),
            }
        )

        if args.save_per_run_files:
            run_df = df.copy()
            run_df["source_row_id"] = source_row_ids
            run_df[pred_id_col] = pred_ids
            run_df[label_name_col] = label_names
            run_df[numeric_label_col] = numeric_labels
            run_df[seed_requested_col] = requested_seeds
            run_df[seed_used_col] = used_seeds
            run_output_file = make_run_filename(args.input_file, run_idx, base_seed, runs_dir)
            save_dataframe(run_df, run_output_file, csv_encoding=args.csv_encoding)

    numeric_run_cols = [f"{args.label_column}_run_{i:02d}" for i in range(1, args.num_runs + 1)]
    combined_df["all_runs_identical"] = combined_df[numeric_run_cols].nunique(axis=1).eq(1)

    run_summary_df = pd.DataFrame(run_records)
    config_df = make_config_df(args, id2label, numeric_mapping)

    output_ext = os.path.splitext(combined_output_path)[1].lower()
    if output_ext in [".xlsx", ".xls"]:
        saved_combined_path = save_excel_workbook(combined_output_path, combined_df, run_summary_df, config_df)
        saved_summary_path = combined_output_path
        print(f"Saved labels, run_summary, and config sheets into one Excel workbook: {saved_combined_path}")
    else:
        saved_combined_path = save_dataframe(combined_df, combined_output_path, csv_encoding=args.csv_encoding)
        saved_summary_path = save_dataframe(run_summary_df, summary_path, csv_encoding=args.csv_encoding)
        print(f"Saved combined multirun file to: {saved_combined_path}")
        print(f"Saved run summary to: {saved_summary_path}")

    if args.save_per_run_files:
        print(f"Saved per-run files to directory: {runs_dir}")


if __name__ == "__main__":
    main()
