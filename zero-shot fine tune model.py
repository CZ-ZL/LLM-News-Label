import argparse
import os
import time
import pandas as pd
import openai
from tqdm import tqdm
from openai.error import RateLimitError, OpenAIError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-label news articles using a fine-tuned OpenAI model."
    )
    parser.add_argument(
        "-i", "--input-file", required=True,
        help="Path to the input Excel file containing news articles."
    )
    parser.add_argument(
        "-o", "--output-file", required=True,
        help="Path to the output Excel file to save labeled data."
    )
    parser.add_argument(
        "-n", "--news-col", default=None,
        help="Name of the column containing the news text. Default: column name News."
    )
    parser.add_argument(
        "-l", "--label-col", default="label",
        help="Name of the new column for labels. Default: 'label'."
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="OpenAI fine-tuned model ID (e.g., gpt-3.5-turbo:ft-...)."
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="Max retries on rate limits."
    )
    parser.add_argument(
        "--backoff-seconds", type=float, default=2.0,
        help="Initial backoff delay in seconds."
    )
    return parser.parse_args()


def init_openai_api():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = key


def call_with_backoff(func, max_retries, backoff):
    delay = backoff
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            print(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
            delay *= 2
        except OpenAIError as oe:
            print(f"OpenAI error: {oe}")
            break
    raise RuntimeError("Failed after retries.")


def label_article(text: str, model: str) -> str:
    """
    Send a single news article to the OpenAI ChatCompletion endpoint and return the model's response as the label.
    """
    def send_request():
        return openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": text}
            ],
            max_tokens=10,
            temperature=0
        )

    response = call_with_backoff(send_request, args.max_retries, args.backoff_seconds)
    return response.choices[0].message.content.strip()


def main():
    global args
    args = parse_args()
    init_openai_api()

    df = pd.read_excel(args.input_file)
    news_col = args.news_col or "News"
    if news_col not in df.columns:
        raise ValueError(f"Column '{news_col}' not found in the Excel file. Available columns: {list(df.columns)}")
    df[args.label_col] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling articles"):
        text = row[news_col]
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            continue
        try:
            label = label_article(text, args.model)
            df.at[idx, args.label_col] = label
        except Exception as e:
            print(f"Error labeling row {idx}: {e}")

    df.to_excel(args.output_file, index=False)
    print(f"Saved labeled data to {args.output_file}")


if __name__ == "__main__":
    main()
