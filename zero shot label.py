import pandas as pd
import os
import openai 
import re
import docx
import argparse
import time
from openai.error import RateLimitError
# Set your DeepSeek API key and base URL
#openai.api_key = "sk-fcab93dba39946c2b213778dca7646ae"  # Replace with your actual API key
#openai.api_base = "https://api.deepseek.com"
openai.api_key = os.getenv("OPENAI_API_KEY")
parser = argparse.ArgumentParser(description='label news')
parser.add_argument('--input_file', type=str, required=True, help='input file')
parser.add_argument('--prompt_file', type=str, required=True, help='zero shot label file')
parser.add_argument('--output_file', type=str, required=True, help='output file')
parser.add_argument('--Label_column', type=str, required=True, help='column name')

# 解析命令行参数
args = parser.parse_args()

# 使用命令行参数替代硬编码的文件路径
input_file = args.input_file
prompt_file = args.prompt_file
output_file = args.output_file



def safe_label(item):
    delay = 2
    for _ in range(5):
        try:
            return label_news_item(item)
        except RateLimitError:
            print(f"wait {delay}s re try…")
            time.sleep(delay)
            delay *= 2
    raise Exception("fail")


def build_prompt_from_excel(filepath):
    """
    Reads an Excel file containing training examples with columns "News" and "label",
    then builds a prompt string where each example is formatted as:
    
        News: <news text>
        Label: <label>
    
    Training examples are separated by a blank line.
    
    Parameters:
        filepath (str): Full path to the Excel file.
        
    Returns:
        str: A prompt string with all training examples.
    """
    # Read the Excel file (assuming headers are present).
    df = pd.read_excel(filepath, header=0)
    news_idx = df.columns.get_loc("News")
    prompt_parts = []
    for _, row in df.iterrows():
        news_text = row["News"]
        label_text = row.iloc[news_idx + 1]
        # Skip rows with missing news or label.
        if pd.isna(news_text) or pd.isna(label_text):
            continue
        prompt_parts.append(f"====\n{news_text}\n@@@@\n{label_text}\n")
    
    prompt = "\n\n".join(prompt_parts)
    return prompt

# Example usage:
prompt_text = build_prompt_from_excel(prompt_file)


def label_news_item(news_item):
    """
    Sends a news item to the DeepSeek API using a prompt template and returns the numeric label.
    Assumes that the API returns a plain text output that is one of "Positive", "Neutral", or "Negative".
    """
    # Insert the news text into the prompt template.
    
    response =  openai.ChatCompletion.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"====\n{news_item}\n@@@@\n"},
        ],
            temperature=0,
            max_tokens=1,
        stream=False
    )
    
    # Get the raw output (plain text) and print it for debugging.
    result = response.choices[0].message.content

    return result

import re

def clean_label_response(response_text):
    """
    Extracts and returns only the numeric label (1, 0, or -1) from the model's response.
    The response is expected to contain a signed number, for example "+1(label)".
    """
    # Use regex to capture a signed or unsigned integer.
    match = re.search(r'([+-]?\d+)', response_text)
    if match:
        number_str = match.group(1).lstrip('+')  # Remove a plus if present.
        if number_str in ['1', '0', '-1']:
            return int(number_str)
    # If not found, return a default (you may choose to raise an error instead)
    return 0


def main():
    # Read the input Excel file (news items assumed to be in the first column).
    df = pd.read_excel(input_file, header=0)
    
    # List to store the numeric labels.
    numeric_labels = []
    
    # Process each news item.
    for index, row in df.iterrows():
        news_item = row["News"]
        if pd.isna(news_item):
            print(f"Row {index} is empty. Skipping...")
            numeric_labels.append(None)
            continue
        
        print(f"\nProcessing row {index}: {str(news_item)[:60]}...")
        try:
            # 使用safe_label函数来获取标签，它会自动处理API限制
            label_text_LLM = safe_label(str(news_item))
            # Map the label text to its numeric equivalent.
            numeric_label = clean_label_response(label_text_LLM)
            numeric_labels.append(numeric_label)
            print(f"Row {index} label: {label_text_LLM} ")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            numeric_labels.append(None)
    
    # Create a new DataFrame for output.
    output_df = df.copy()
    output_df[args.Label_column] = numeric_labels
    
    # Export the new DataFrame to a separate Excel file.
    output_df.to_excel(output_file, index=False)
    print(f"\nExported labeled news to {output_file}")

if __name__ == "__main__":
    main()
