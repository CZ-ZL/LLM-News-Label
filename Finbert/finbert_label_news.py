import argparse
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Use fine-tuned FinBERT to label news sentiment.")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to your fine-tuned FinBERT sentiment model directory.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input CSV/Excel file containing a 'News' column.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save labeled file. If not provided, will add '_labeled' before extension.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="News",
        help="Name of the column containing news text.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="FinBERT_label",
        help="Name of the numeric label column to add (1=positive, -1=negative).",
    )
    return parser.parse_args()


def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .csv or .xlsx/.xls.")
    return df


def save_dataframe(df: pd.DataFrame, path: str):
    ext = os.path.splitext(path)[1].lower()
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
        # 默认存成 csv
        df.to_csv(path + ".csv", index=False)


def add_suffix_to_filename(path: str, suffix: str = "_labeled") -> str:
    base, ext = os.path.splitext(path)
    return base + suffix + ext


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 读数据
    df = load_dataframe(args.input_file)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in file. Columns: {list(df.columns)}")

    texts = df[args.text_column].astype(str).tolist()

    # 2. 加载模型和 tokenizer（从 fine-tune 后的目录）
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # 3. 批量推理
    all_pred_ids = []
    batch_size = args.batch_size

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()  # 0/1

            all_pred_ids.extend(preds)

    # 4. 把 0/1 映射到 -1 / 1
    # 按你训练时的定义：0 = 负面, 1 = 正面
    numeric_labels = [1 if p == 1 else -1 for p in all_pred_ids]

    df[args.label_column] = numeric_labels  # 只这一列，纯数字

    # 5. 保存结果
    output_path = args.output_file or add_suffix_to_filename(args.input_file, "_labeled")
    save_dataframe(df, output_path)
    print(f"Saved labeled file to: {output_path}")


if __name__ == "__main__":
    main()

