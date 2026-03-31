import os
import re
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docx
import openai
import pandas as pd

openai.api_key = os.getenv("deepseek_key")
openai.api_base = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"


def build_openai_client():
    """Return a chat client compatible with either legacy or modern OpenAI-style SDKs."""
    if hasattr(openai, "OpenAI"):
        return openai.OpenAI(api_key=openai.api_key, base_url=openai.api_base)
    return None


class PauseForCredits(Exception):
    """Raised when the API appears to have insufficient balance or quota."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label news with DeepSeek (CSV/XLSX, resumable, checkpointed, summary-ready)"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV/XLSX/XLS file")
    parser.add_argument("--prompt_file", type=str, required=True, help="Prompt DOCX file")
    parser.add_argument("--output_file", type=str, required=True, help="Output Excel workbook")
    parser.add_argument("--text_column", type=str, default="headline", help="Column sent to the model")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed when num_runs=1")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of full repeated runs")
    parser.add_argument("--input_encoding", type=str, default=None, help="Optional CSV encoding override")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Optional checkpoint JSON path")
    parser.add_argument("--log_file", type=str, default=None, help="Optional JSONL event log path")
    parser.add_argument("--save_every", type=int, default=10, help="Save progress every N processed rows")
    parser.add_argument("--max_retries", type=int, default=4, help="Retries for transient API errors")
    parser.add_argument("--retry_wait_seconds", type=float, default=3.0, help="Initial retry backoff")
    parser.add_argument("--request_pause_seconds", type=float, default=0.0, help="Pause after each successful request")
    parser.add_argument("--resume", action="store_true", help="Resume from prior output/checkpoint if available")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap for testing/debugging")
    parser.add_argument(
        "--stop_after_api_calls",
        type=int,
        default=None,
        help="Optional safety cap on successful API calls for one invocation",
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


def load_prompt_template(filepath: str) -> str:
    doc = docx.Document(filepath)
    text = "\n".join(para.text for para in doc.paragraphs)
    return text.strip()


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def read_input_table(path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=0)

    if suffix == ".csv":
        import io

        encodings = []
        for e in [encoding, "utf-8", "utf-8-sig", "cp1252", "latin1"]:
            if e and e not in encodings:
                encodings.append(e)

        last_err = None

        # First, try reading the file directly with pandas.
        for enc in encodings:
            for parser_kwargs in [
                {"engine": "c"},
                {"engine": "python"},
            ]:
                try:
                    return pd.read_csv(
                        path,
                        header=0,
                        encoding=enc,
                        sep=",",
                        quotechar='"',
                        **parser_kwargs,
                    )
                except Exception as exc:
                    last_err = exc

        # If direct parsing fails, clean raw bytes and retry from memory.
        raw = p.read_bytes()
        if b"\x00" in raw:
            raw = raw.replace(b"\x00", b"")

        for enc in encodings:
            try:
                text = raw.decode(enc)
            except Exception as exc:
                last_err = exc
                continue

            for parser_kwargs in [
                {"engine": "python"},
                {"engine": "python", "on_bad_lines": "warn"},
                {"engine": "python", "on_bad_lines": "skip"},
            ]:
                try:
                    return pd.read_csv(
                        io.StringIO(text),
                        header=0,
                        sep=",",
                        quotechar='"',
                        **parser_kwargs,
                    )
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


def sanitize_dataframe(df: pd.DataFrame, text_col: str, drop_blank_text: bool = False, max_rows: Optional[int] = None) -> pd.DataFrame:
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


def extract_impact(label_response: Any) -> str:
    text = str(label_response).strip()
    patterns = [
        r'"impact"\s*:\s*"?([^"}\n]+)"?',
        r'"label"\s*:\s*"?([^"}\n]+)"?',
        r'\b(positive|neutral|negative)\b',
        r'\b(-1|0|1)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text


def normalize_label(value: Any) -> str:
    cleaned = str(value).strip().lower()
    mapping = {
        "1": "1",
        "0": "0",
        "-1": "-1",
        "positive": "1",
        "neutral": "0",
        "negative": "-1",
        "+1": "1",
    }
    return mapping.get(cleaned, "0")


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    insufficient = [
        "insufficient balance",
        "insufficient quota",
        "quota exceeded",
        "not enough balance",
        "insufficient credits",
        "payment required",
        "billing",
        "credit",
        "余额不足",
    ]
    transient = [
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "server error",
        "bad gateway",
        "gateway timeout",
        "service unavailable",
    ]
    if any(k in text for k in insufficient) or "402" in text:
        return "insufficient_funds"
    if any(k in text for k in transient) or "429" in text or "5xx" in text:
        return "transient"
    return "other"


def label_news_item(text_item: str, prompt_template: str, temperature: float, seed: int) -> str:
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": str(text_item)},
    ]

    if hasattr(openai, "ChatCompletion"):
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            temperature=float(temperature),
            seed=int(seed),
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content

    client = build_openai_client()
    if client is not None:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=float(temperature),
            seed=int(seed),
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content

    raise RuntimeError(
        "No compatible OpenAI/DeepSeek client interface found. Install a supported SDK or update the script client call."
    )


def label_news_item_with_retry(
    text_item: str,
    prompt_template: str,
    temperature: float,
    seed: int,
    max_retries: int,
    retry_wait_seconds: float,
) -> str:
    attempt = 0
    while True:
        try:
            return label_news_item(text_item, prompt_template, temperature, seed)
        except Exception as exc:
            kind = classify_error(exc)
            if kind == "insufficient_funds":
                raise PauseForCredits(str(exc)) from exc
            if kind == "transient" and attempt < max_retries:
                sleep_s = retry_wait_seconds * (2 ** attempt)
                print(f"Transient API error. seed={seed}, retry {attempt + 1}/{max_retries} in {sleep_s:.1f}s: {exc}")
                time.sleep(sleep_s)
                attempt += 1
                continue
            raise


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
    temperature: float,
    text_col: str,
    run_session_id: str,
) -> List[Dict[str, Any]]:
    by_run = {int(m["run_number"]): dict(m) for m in existing_meta if "run_number" in m}
    output: List[Dict[str, Any]] = []
    for run_number in range(1, num_runs + 1):
        if run_number in by_run and pd.notna(by_run[run_number].get("base_seed")):
            meta = by_run[run_number]
        else:
            base_seed = requested_seed if requested_seed is not None and num_runs == 1 else get_unix_seed(run_number - 1)
            meta = {
                "run_number": run_number,
                "base_seed": int(base_seed),
                "status": "pending",
                "last_completed_row": -1,
                "rows_success": 0,
                "rows_error": 0,
            }
        meta["temperature"] = float(temperature)
        meta["model"] = MODEL_NAME
        meta["text_column"] = text_col
        meta["run_session_id"] = run_session_id
        output.append(meta)
    return output


def ensure_run_columns(output_df: pd.DataFrame, run_number: int) -> Tuple[str, str, str]:
    suffix = f"{run_number:02d}"
    label_col = f"label_run_{suffix}"
    raw_col = f"raw_run_{suffix}"
    seed_col = f"seed_run_{suffix}"
    for col in (label_col, raw_col, seed_col):
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


def build_config_df(args: argparse.Namespace, prompt_text: str, text_col: str, df_rows: int) -> pd.DataFrame:
    rows = [
        ("model", MODEL_NAME),
        ("temperature", args.temperature),
        ("num_runs", args.num_runs),
        ("text_column", text_col),
        ("input_file", args.input_file),
        ("prompt_file", args.prompt_file),
        ("output_file", args.output_file),
        ("prompt_sha256_16", prompt_hash(prompt_text)),
        ("prompt_preview", prompt_text[:500]),
        ("input_rows", df_rows),
        ("save_every", args.save_every),
        ("max_retries", args.max_retries),
        ("retry_wait_seconds", args.retry_wait_seconds),
        ("request_pause_seconds", args.request_pause_seconds),
        ("resume", args.resume),
        ("max_rows", args.max_rows),
        ("drop_blank_text", args.drop_blank_text),
        ("stop_after_api_calls", args.stop_after_api_calls),
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
    prompt_text: str,
    text_col: str,
) -> None:
    summary_df = build_summary_df(output_df, args.num_runs)
    config_df = build_config_df(args, prompt_text, text_col, len(output_df))
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

    append_event(log_file, {"event": "checkpoint_saved", **state_payload})


def build_state_payload(args: argparse.Namespace, run_number: int, row_index: int, status: str, message: str) -> Dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "current_run": run_number,
        "current_row": row_index,
        "input_file": args.input_file,
        "prompt_file": args.prompt_file,
        "output_file": args.output_file,
        "text_column": args.text_column,
        "temperature": args.temperature,
        "num_runs": args.num_runs,
        "saved_at": utc_now_str(),
    }


def main() -> None:
    args = parse_args()

    if not openai.api_key:
        raise EnvironmentError("Environment variable 'deepseek_key' is not set.")

    if not (hasattr(openai, "ChatCompletion") or hasattr(openai, "OpenAI")):
        raise RuntimeError(
            "Installed openai/deepseek client is not compatible with this script. Need legacy ChatCompletion or modern OpenAI client."
        )

    output_file = Path(args.output_file)
    checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else default_checkpoint_path(args.output_file)
    log_file = Path(args.log_file) if args.log_file else default_log_path(args.output_file)
    run_session_id = f"session_{get_unix_seed()}"
    os.environ["RUN_SESSION_ID"] = run_session_id

    prompt_text = load_prompt_template(args.prompt_file)
    base_df = read_input_table(args.input_file, encoding=args.input_encoding)
    text_col = find_text_column(base_df, args.text_column)
    df = sanitize_dataframe(base_df, text_col, drop_blank_text=args.drop_blank_text, max_rows=args.max_rows)

    output_df = initialize_output_df(df, output_file, resume=args.resume)
    existing_meta = get_existing_run_metadata(output_file, resume=args.resume)
    run_metadata = get_or_create_run_metadata(
        existing_meta=existing_meta,
        num_runs=args.num_runs,
        requested_seed=args.seed,
        temperature=args.temperature,
        text_col=text_col,
        run_session_id=run_session_id,
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
            "temperature": args.temperature,
            "resume": args.resume,
            "rows": len(df),
            "prompt_sha": prompt_hash(prompt_text),
        },
    )

    successful_api_calls = 0
    total_rows = len(df)

    for run_idx in range(args.num_runs):
        run_number = run_idx + 1
        label_col, raw_col, seed_col = ensure_run_columns(output_df, run_number)
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
            "run_session_id": run_session_id,
        })

        processed_since_save = 0

        for row_idx in range(start_row, total_rows):
            if args.stop_after_api_calls is not None and successful_api_calls >= args.stop_after_api_calls:
                state_payload = build_state_payload(
                    args, run_number, row_idx, "paused_manual_budget_cap", f"Stopped after {successful_api_calls} successful API calls"
                )
                run_metadata[run_idx]["status"] = "paused_manual_budget_cap"
                save_progress(output_df, run_metadata, output_file, checkpoint_file, log_file, state_payload, args, prompt_text, text_col)
                print("Paused because stop_after_api_calls was reached. Progress saved.")
                return

            text_item = df.at[row_idx, text_col]
            if pd.isna(text_item) or str(text_item).strip() == "":
                output_df.at[row_idx, label_col] = pd.NA
                output_df.at[row_idx, raw_col] = pd.NA
                output_df.at[row_idx, seed_col] = pd.NA
                run_metadata[run_idx]["last_completed_row"] = row_idx
                processed_since_save += 1
                continue

            call_seed = base_seed + int(df.at[row_idx, "source_row_id"])
            preview = str(text_item).replace("\n", " ")[:120]
            print(f"Run {run_number} | Row {row_idx} | seed={call_seed} | {preview}...")

            try:
                raw_text = label_news_item_with_retry(
                    text_item=str(text_item),
                    prompt_template=prompt_text,
                    temperature=args.temperature,
                    seed=call_seed,
                    max_retries=args.max_retries,
                    retry_wait_seconds=args.retry_wait_seconds,
                )
                label = normalize_label(extract_impact(raw_text))
                output_df.at[row_idx, label_col] = label
                output_df.at[row_idx, raw_col] = raw_text
                output_df.at[row_idx, seed_col] = call_seed
                run_metadata[run_idx]["last_completed_row"] = row_idx
                run_metadata[run_idx]["rows_success"] = int(run_metadata[run_idx].get("rows_success", 0)) + 1
                successful_api_calls += 1
                append_event(log_file, {
                    "event": "row_success",
                    "run_number": run_number,
                    "row_index": row_idx,
                    "source_row_id": int(df.at[row_idx, "source_row_id"]),
                    "seed": call_seed,
                    "label": label,
                })
            except PauseForCredits as exc:
                state_payload = build_state_payload(args, run_number, row_idx, "paused_insufficient_funds", str(exc))
                run_metadata[run_idx]["status"] = "paused_insufficient_funds"
                run_metadata[run_idx]["paused_at"] = utc_now_str()
                save_progress(output_df, run_metadata, output_file, checkpoint_file, log_file, state_payload, args, prompt_text, text_col)
                print("Paused because API balance/quota appears insufficient. Progress saved.")
                return
            except Exception as exc:
                output_df.at[row_idx, label_col] = pd.NA
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
                save_progress(output_df, run_metadata, output_file, checkpoint_file, log_file, state_payload, args, prompt_text, text_col)
                processed_since_save = 0

            if args.request_pause_seconds > 0:
                time.sleep(args.request_pause_seconds)

        run_metadata[run_idx]["status"] = "completed"
        run_metadata[run_idx]["completed_at"] = utc_now_str()
        state_payload = build_state_payload(args, run_number, total_rows - 1 if total_rows else -1, "running", f"Completed run {run_number}")
        save_progress(output_df, run_metadata, output_file, checkpoint_file, log_file, state_payload, args, prompt_text, text_col)

    final_state = build_state_payload(
        args,
        args.num_runs,
        total_rows - 1 if total_rows else -1,
        "completed",
        "All runs completed successfully",
    )
    save_progress(output_df, run_metadata, output_file, checkpoint_file, log_file, final_state, args, prompt_text, text_col)
    append_event(log_file, {
        "event": "job_completed",
        "run_session_id": run_session_id,
        "successful_api_calls": successful_api_calls,
    })
    print(f"Exported {args.num_runs} runs to {args.output_file}")


if __name__ == "__main__":
    main()
