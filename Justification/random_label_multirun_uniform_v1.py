import argparse
import json
import os
import random
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


MODEL_NAME = "random-uniform-baseline"
LABEL_SPACE = [-1, 0, 1]
DEFAULT_CSV_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign random labels (-1, 0, 1) with equal probability, with multi-run output, "
            "checkpointing, resume support, and one combined Excel workbook."
        )
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV/XLSX/XLS file")
    parser.add_argument("--output_file", type=str, required=True, help="Output Excel workbook path")
    parser.add_argument("--text_column", type=str, default="News", help="Column used to decide which rows are labelable")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of full repeated runs")
    parser.add_argument(
        "--seed_mode",
        type=str,
        default="each_label",
        help=(
            "Seed granularity: each_label or each_run. "
            "Aliases per_label/per_run are also accepted."
        ),
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=None,
        help="Optional starting seed. If provided, run seeds advance deterministically from this value.",
    )
    parser.add_argument(
        "--input_encoding",
        type=str,
        default=None,
        help="Optional CSV encoding override",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="Optional checkpoint JSON path",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional JSONL event log path",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save lightweight checkpoint every N processed rows",
    )
    parser.add_argument(
        "--save_raw_text",
        action="store_true",
        help="Also store an audit string for each generated label in the Excel output.",
    )
    parser.add_argument(
        "--request_pause_seconds",
        type=float,
        default=0.0,
        help="Optional pause after each generated label",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from prior output/checkpoint if available",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap for testing/debugging",
    )
    parser.add_argument(
        "--stop_after_generated_labels",
        type=int,
        default=None,
        help="Optional safety cap on generated labels for one invocation",
    )
    parser.add_argument(
        "--drop_blank_text",
        action="store_true",
        help="Drop rows with blank text in the selected text column before running",
    )
    return parser.parse_args()


def default_checkpoint_path(output_file: str) -> Path:
    p = Path(output_file)
    return p.with_name(f"{p.stem}.checkpoint.json")


def default_log_path(output_file: str) -> Path:
    p = Path(output_file)
    return p.with_name(f"{p.stem}.events.jsonl")


def utc_now_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_unix_seed(offset: int = 0) -> int:
    return int(time.time() * 1000) + int(offset)


def normalize_seed_mode(seed_mode: str) -> str:
    normalized = str(seed_mode).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "each_label": "each_label",
        "per_label": "each_label",
        "label": "each_label",
        "each_run": "each_run",
        "per_run": "each_run",
        "run": "each_run",
    }
    if normalized not in mapping:
        raise ValueError("seed_mode must be one of: each_label, each_run (aliases: per_label, per_run)")
    return mapping[normalized]


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def read_input_table(path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=0)

    if suffix == ".csv":
        encodings = []
        for e in [encoding, *DEFAULT_CSV_ENCODINGS]:
            if e and e not in encodings:
                encodings.append(e)
        last_err = None
        for enc in encodings:
            for parser_kwargs in [{"engine": "c"}, {"engine": "python"}]:
                try:
                    return pd.read_csv(path, header=0, encoding=enc, **parser_kwargs)
                except Exception as exc:
                    last_err = exc
        raise ValueError(f"Failed to read CSV using encodings {encodings}: {last_err}")

    raise ValueError(f"Unsupported file type: {suffix}")


def find_text_column(df: pd.DataFrame, requested_name: str) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    key = requested_name.strip().lower()
    if key in normalized:
        return str(normalized[key])
    raise KeyError(f"Column '{requested_name}' not found. Available columns: {list(df.columns)}")


def sanitize_dataframe(
    df: pd.DataFrame,
    text_col: str,
    drop_blank_text: bool = False,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    out = df.copy()
    out[text_col] = out[text_col].astype("string")
    out["source_row_id"] = range(len(out))

    if drop_blank_text:
        mask = out[text_col].fillna("").str.strip().ne("")
        out = out[mask].copy()

    if max_rows is not None:
        out = out.head(max_rows).copy()

    out.reset_index(drop=True, inplace=True)
    return out


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def append_event(log_path: Path, payload: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "ts": utc_now_str()}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=make_json_safe) + "\n")


def load_existing_sheet(output_file: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    if not output_file.exists():
        return None
    try:
        return pd.read_excel(output_file, sheet_name=sheet_name)
    except Exception:
        return None


def initialize_output_df(base_df: pd.DataFrame, output_file: Path, resume: bool) -> pd.DataFrame:
    if not resume:
        return base_df.copy()
    existing = load_existing_sheet(output_file, "labels")
    if existing is None:
        return base_df.copy()
    if len(existing) != len(base_df):
        raise ValueError(
            f"Resume aborted: existing labels sheet has {len(existing)} rows but current input has {len(base_df)} rows. "
            "Use the exact same input/filter settings when resuming."
        )

    output_df = base_df.copy()
    for col in existing.columns:
        if col not in output_df.columns:
            output_df[col] = existing[col].values
    return output_df


def get_existing_run_metadata(output_file: Path, resume: bool) -> List[Dict[str, Any]]:
    if not resume:
        return []
    existing = load_existing_sheet(output_file, "run_metadata")
    if existing is None:
        return []
    return existing.to_dict(orient="records")


def get_or_create_run_metadata(
    existing_meta: List[Dict[str, Any]],
    num_runs: int,
    requested_seed: Optional[int],
    text_col: str,
    run_session_id: str,
    seed_mode: str,
) -> List[Dict[str, Any]]:
    by_run = {int(m["run_number"]): dict(m) for m in existing_meta if "run_number" in m}
    output: List[Dict[str, Any]] = []
    for run_number in range(1, num_runs + 1):
        if run_number in by_run and pd.notna(by_run[run_number].get("base_seed")):
            meta = by_run[run_number]
        else:
            base_seed = int(requested_seed) + (run_number - 1) if requested_seed is not None else get_unix_seed(run_number - 1)
            meta = {
                "run_number": run_number,
                "base_seed": int(base_seed),
                "status": "pending",
                "last_completed_row": -1,
                "rows_success": 0,
                "rows_error": 0,
            }
        meta["model"] = MODEL_NAME
        meta["text_column"] = text_col
        meta["seed_mode"] = seed_mode
        meta["run_session_id"] = run_session_id
        output.append(meta)
    return output


def ensure_run_columns(output_df: pd.DataFrame, run_number: int, save_raw_text: bool) -> Tuple[str, Optional[str], str]:
    suffix = f"{run_number:02d}"
    label_col = f"label_run_{suffix}"
    raw_col = f"raw_run_{suffix}" if save_raw_text else None
    seed_col = f"seed_run_{suffix}"
    cols = [label_col, seed_col]
    if raw_col is not None:
        cols.append(raw_col)
    for col in cols:
        if col not in output_df.columns:
            output_df[col] = pd.NA
    return label_col, raw_col, seed_col


def next_unfinished_row(output_df: pd.DataFrame, label_col: str, total_rows: int) -> int:
    if label_col not in output_df.columns:
        return 0
    for idx in range(total_rows):
        val = output_df.at[idx, label_col]
        if pd.isna(val) or str(val).strip() == "":
            return idx
    return total_rows


def build_summary_df(output_df: pd.DataFrame, num_runs: int) -> pd.DataFrame:
    summary = pd.DataFrame(index=output_df.index)
    label_cols = [f"label_run_{i:02d}" for i in range(1, num_runs + 1) if f"label_run_{i:02d}" in output_df.columns]

    def to_int_or_nan(x: Any) -> Optional[int]:
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s in {"-1", "0", "1"}:
            return int(s)
        return None

    label_matrix = pd.DataFrame({col: output_df[col].map(to_int_or_nan) for col in label_cols}) if label_cols else pd.DataFrame(index=output_df.index)

    if not label_matrix.empty:
        summary["valid_runs"] = label_matrix.notna().sum(axis=1)
        summary["count_neg1"] = (label_matrix == -1).sum(axis=1)
        summary["count_0"] = (label_matrix == 0).sum(axis=1)
        summary["count_1"] = (label_matrix == 1).sum(axis=1)
        summary["mean_label"] = label_matrix.mean(axis=1)
        summary["std_label"] = label_matrix.std(axis=1)
        summary["agreement_ratio"] = label_matrix.apply(
            lambda row: row.value_counts(dropna=True).max() / row.notna().sum() if row.notna().sum() else pd.NA,
            axis=1,
        )

        def majority_vote(row: pd.Series) -> str:
            counts = row.value_counts(dropna=True)
            if counts.empty:
                return ""
            top_count = counts.iloc[0]
            top_labels = sorted([int(idx) for idx, cnt in counts.items() if cnt == top_count])
            return str(top_labels[0])

        summary["majority_label"] = label_matrix.apply(majority_vote, axis=1)
    else:
        summary["valid_runs"] = 0
        summary["count_neg1"] = 0
        summary["count_0"] = 0
        summary["count_1"] = 0
        summary["mean_label"] = pd.NA
        summary["std_label"] = pd.NA
        summary["agreement_ratio"] = pd.NA
        summary["majority_label"] = ""

    return summary


def build_config_df(args: argparse.Namespace, text_col: str, df_rows: int) -> pd.DataFrame:
    distribution_text = "uniform over [-1, 0, 1] with probability 1/3 each"
    rows = [
        ("model", MODEL_NAME),
        ("label_distribution", distribution_text),
        ("label_distribution_sha256_16", prompt_hash(distribution_text)),
        ("num_runs", args.num_runs),
        ("seed_mode", args.seed_mode),
        ("base_seed", args.base_seed),
        ("text_column", text_col),
        ("input_file", args.input_file),
        ("output_file", args.output_file),
        ("input_rows", df_rows),
        ("save_every", args.save_every),
        ("save_raw_text", args.save_raw_text),
        ("request_pause_seconds", args.request_pause_seconds),
        ("resume", args.resume),
        ("max_rows", args.max_rows),
        ("drop_blank_text", args.drop_blank_text),
        ("stop_after_generated_labels", args.stop_after_generated_labels),
        ("run_session_id", os.environ.get("RUN_SESSION_ID", "")),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def atomic_write_excel(output_file: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_name(f"{output_file.stem}.tmp.xlsx")
    with pd.ExcelWriter(temp_file, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    os.replace(temp_file, output_file)


def save_progress(
    output_df: pd.DataFrame,
    run_metadata: List[Dict[str, Any]],
    output_file: Path,
    checkpoint_file: Path,
    log_file: Path,
    state_payload: Dict[str, Any],
    args: argparse.Namespace,
    text_col: str,
    write_excel: bool = False,
) -> None:
    if write_excel:
        summary_df = build_summary_df(output_df, args.num_runs)
        config_df = build_config_df(args, text_col, len(output_df))
        atomic_write_excel(
            output_file,
            {
                "labels": output_df,
                "summary": summary_df,
                "run_metadata": pd.DataFrame(run_metadata),
                "config": config_df,
            },
        )

    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_file.open("w", encoding="utf-8") as f:
        json.dump(state_payload, f, ensure_ascii=False, indent=2, default=make_json_safe)

    append_event(log_file, {"event": "checkpoint_saved", "write_excel": write_excel, **state_payload})


def build_state_payload(args: argparse.Namespace, run_number: int, row_index: int, status: str, message: str) -> Dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "current_run": run_number,
        "current_row": row_index,
        "input_file": args.input_file,
        "output_file": args.output_file,
        "text_column": args.text_column,
        "num_runs": args.num_runs,
        "seed_mode": args.seed_mode,
        "saved_at": utc_now_str(),
    }


def is_labelable_text(value: Any) -> bool:
    return not (pd.isna(value) or str(value).strip() == "")


def advance_rng_for_resume(rng: random.Random, df: pd.DataFrame, text_col: str, start_row: int) -> int:
    draws = 0
    for row_idx in range(start_row):
        if is_labelable_text(df.at[row_idx, text_col]):
            rng.choice(LABEL_SPACE)
            draws += 1
    return draws


def draw_random_label_each_label(call_seed: int) -> int:
    local_rng = random.Random(call_seed)
    return int(local_rng.choice(LABEL_SPACE))


def main() -> None:
    args = parse_args()
    args.seed_mode = normalize_seed_mode(args.seed_mode)

    output_file = Path(args.output_file)
    checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else default_checkpoint_path(args.output_file)
    log_file = Path(args.log_file) if args.log_file else default_log_path(args.output_file)
    run_session_id = f"session_{get_unix_seed()}"
    os.environ["RUN_SESSION_ID"] = run_session_id

    base_df = read_input_table(args.input_file, encoding=args.input_encoding)
    text_col = find_text_column(base_df, args.text_column)
    df = sanitize_dataframe(base_df, text_col, drop_blank_text=args.drop_blank_text, max_rows=args.max_rows)

    output_df = initialize_output_df(df, output_file, resume=args.resume)
    existing_meta = get_existing_run_metadata(output_file, resume=args.resume)
    run_metadata = get_or_create_run_metadata(
        existing_meta=existing_meta,
        num_runs=args.num_runs,
        requested_seed=args.base_seed,
        text_col=text_col,
        run_session_id=run_session_id,
        seed_mode=args.seed_mode,
    )

    append_event(
        log_file,
        {
            "event": "job_started",
            "run_session_id": run_session_id,
            "input_file": args.input_file,
            "output_file": args.output_file,
            "text_column": text_col,
            "num_runs": args.num_runs,
            "seed_mode": args.seed_mode,
            "resume": args.resume,
            "rows": len(df),
            "distribution": "uniform_equal_3way",
        },
    )

    generated_labels = 0
    total_rows = len(df)

    for run_idx in range(args.num_runs):
        run_number = run_idx + 1
        label_col, raw_col, seed_col = ensure_run_columns(output_df, run_number, args.save_raw_text)
        start_row = next_unfinished_row(output_df, label_col, total_rows) if args.resume else 0
        base_seed = int(run_metadata[run_idx]["base_seed"])
        run_metadata[run_idx]["status"] = "running"
        run_metadata[run_idx]["started_at"] = run_metadata[run_idx].get("started_at", utc_now_str())

        print(f"Starting run {run_number}/{args.num_runs} from row {start_row} with base_seed={base_seed}")
        append_event(log_file, {
            "event": "run_started",
            "run_number": run_number,
            "start_row": start_row,
            "base_seed": base_seed,
            "seed_mode": args.seed_mode,
            "run_session_id": run_session_id,
        })

        processed_since_save = 0
        draw_index = 0
        run_rng: Optional[random.Random] = None

        if args.seed_mode == "each_run":
            run_rng = random.Random(base_seed)
            draw_index = advance_rng_for_resume(run_rng, df, text_col, start_row)
            if draw_index > 0:
                append_event(log_file, {
                    "event": "run_rng_advanced_for_resume",
                    "run_number": run_number,
                    "base_seed": base_seed,
                    "advanced_draws": draw_index,
                })

        for row_idx in range(start_row, total_rows):
            if args.stop_after_generated_labels is not None and generated_labels >= args.stop_after_generated_labels:
                state_payload = build_state_payload(
                    args,
                    run_number,
                    row_idx,
                    "paused_manual_budget_cap",
                    f"Stopped after {generated_labels} generated labels",
                )
                run_metadata[run_idx]["status"] = "paused_manual_budget_cap"
                save_progress(output_df, run_metadata, output_file, checkpoint_file, log_file, state_payload, args, text_col, write_excel=True)
                print("Paused because stop_after_generated_labels was reached. Progress saved.")
                return

            text_item = df.at[row_idx, text_col]
            if not is_labelable_text(text_item):
                output_df.at[row_idx, label_col] = pd.NA
                if raw_col is not None:
                    output_df.at[row_idx, raw_col] = pd.NA
                output_df.at[row_idx, seed_col] = pd.NA
                run_metadata[run_idx]["last_completed_row"] = row_idx
                processed_since_save += 1
                continue

            if args.seed_mode == "each_label":
                call_seed = base_seed + int(df.at[row_idx, "source_row_id"])
                label = draw_random_label_each_label(call_seed)
                audit_text = f"mode=each_label;base_seed={base_seed};call_seed={call_seed};label={label}"
            else:
                if run_rng is None:
                    raise RuntimeError("run_rng was not initialized for each_run mode")
                call_seed = base_seed
                label = int(run_rng.choice(LABEL_SPACE))
                audit_text = f"mode=each_run;base_seed={base_seed};draw_index={draw_index};label={label}"
                draw_index += 1

            preview = str(text_item).replace("\n", " ")[:120]
            print(f"Run {run_number} | Row {row_idx} | seed={call_seed} | label={label} | {preview}...")

            try:
                output_df.at[row_idx, label_col] = label
                if raw_col is not None:
                    output_df.at[row_idx, raw_col] = audit_text
                output_df.at[row_idx, seed_col] = call_seed
                run_metadata[run_idx]["last_completed_row"] = row_idx
                run_metadata[run_idx]["rows_success"] = int(run_metadata[run_idx].get("rows_success", 0)) + 1
                generated_labels += 1
                append_event(log_file, {
                    "event": "row_success",
                    "run_number": run_number,
                    "row_index": row_idx,
                    "source_row_id": int(df.at[row_idx, "source_row_id"]),
                    "seed": call_seed,
                    "label": label,
                    "audit_text": audit_text,
                })
            except Exception as exc:
                output_df.at[row_idx, label_col] = pd.NA
                if raw_col is not None:
                    output_df.at[row_idx, raw_col] = f"ERROR: {exc}"
                output_df.at[row_idx, seed_col] = call_seed
                run_metadata[run_idx]["last_completed_row"] = row_idx
                run_metadata[run_idx]["rows_error"] = int(run_metadata[run_idx].get("rows_error", 0)) + 1
                append_event(log_file, {
                    "event": "row_error",
                    "run_number": run_number,
                    "row_index": row_idx,
                    "source_row_id": int(df.at[row_idx, "source_row_id"]),
                    "seed": call_seed,
                    "error": str(exc),
                })
                print(f"Run {run_number} | Error processing row {row_idx}: {exc}")

            processed_since_save += 1
            if processed_since_save >= args.save_every:
                state_payload = build_state_payload(
                    args,
                    run_number,
                    row_idx,
                    "running",
                    f"Checkpoint after processing row {row_idx} in run {run_number}",
                )
                save_progress(output_df, run_metadata, output_file, checkpoint_file, log_file, state_payload, args, text_col, write_excel=False)
                processed_since_save = 0

            if args.request_pause_seconds > 0:
                time.sleep(args.request_pause_seconds)

        run_metadata[run_idx]["status"] = "completed"
        run_metadata[run_idx]["completed_at"] = utc_now_str()
        state_payload = build_state_payload(args, run_number, total_rows - 1 if total_rows else -1, "running", f"Completed run {run_number}")
        save_progress(output_df, run_metadata, output_file, checkpoint_file, log_file, state_payload, args, text_col, write_excel=True)

    final_state = build_state_payload(
        args,
        args.num_runs,
        total_rows - 1 if total_rows else -1,
        "completed",
        "All runs completed successfully",
    )
    save_progress(output_df, run_metadata, output_file, checkpoint_file, log_file, final_state, args, text_col, write_excel=True)
    append_event(log_file, {
        "event": "job_completed",
        "run_session_id": run_session_id,
        "generated_labels": generated_labels,
    })
    print(f"Exported {args.num_runs} runs to {args.output_file}")


if __name__ == "__main__":
    main()
