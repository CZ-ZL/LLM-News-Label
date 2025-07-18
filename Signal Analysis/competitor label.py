import os
import pandas as pd
import openai
import re
import argparse

# Set your DeepSeek API key and base URL
openai.api_key = os.getenv("deepseek_key") # Replace with your actual API key
openai.api_base = "https://api.deepseek.com"

parser = argparse.ArgumentParser(description='label news')
parser.add_argument('--input_file', type=str, required=True, help='input file')
parser.add_argument('--output_file', type=str, required=True, help='output file')

# Parse command line arguments
args = parser.parse_args()

# Use command line arguments instead of hardcoded file paths
input_file = args.input_file
output_file = args.output_file

# New prompt template
PROMPT_TEMPLATE = """Act as an expert of forex trading holding US/CHN. Based on the article {article}, will you buy, sell or hold US/CHN in the short term? Answer in one token: positive for a buy, negative for a sell, or neutral for a hold position."""

def map_label_to_numeric(label_str):
    """
    Map the label string to numeric values:
      - "positive" -> 1
      - "negative" -> -1
      - "neutral" -> 0
    If the label is not recognized, defaults to 0.
    """
    # Normalize the string: remove extra spaces and convert to lowercase
    label_str = label_str.strip().lower()
    
    mapping = {
        "positive": 1,
        "negative": -1,
        "neutral": 0
    }
    
    numeric_value = mapping.get(label_str, 0)
    
    # Ensure we only return valid numeric values
    if numeric_value not in [0, -1, 1]:
        print(f"Warning: Unexpected label '{label_str}' mapped to {numeric_value}, defaulting to 0")
        return 0
    
    return numeric_value

def label_news_item(news_item):
    """
    Sends a news item to the DeepSeek API using the prompt template 
    and returns the label (positive/negative/neutral).
    """
    # Insert the news text into the prompt template
    prompt = PROMPT_TEMPLATE.format(article=news_item)
    
    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=1
    )
    
    # Get the raw output and clean it
    result = response.choices[0].message.content.strip()
    return result

def main():
    # Read the input Excel file
    df = pd.read_excel(input_file, header=0)
    
    # Lists to store both text and numeric labels
    text_labels = []
    numeric_labels = []
    
    # Process each news item
    for index, row in df.iterrows():
        news_item = row["News"]
        if pd.isna(news_item):
            print(f"Row {index} is empty. Skipping...")
            text_labels.append(None)
            numeric_labels.append(None)
            continue
        
        print(f"\nProcessing row {index}: {str(news_item)[:60]}...")
        try:
            # Get the text label from the API
            text_label = label_news_item(str(news_item))
            # Convert to numeric label
            numeric_label = map_label_to_numeric(text_label)
            
            text_labels.append(text_label)
            numeric_labels.append(numeric_label)
            
            print(f"Row {index} label: {text_label} -> numeric label: {numeric_label}")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            text_labels.append(None)
            numeric_labels.append(None)
    
    # Create a new DataFrame for output
    output_df = df.copy()
    output_df["Competitor Label"] = numeric_labels  # Only numeric values: 0, -1, 1
    output_df["Text Label"] = text_labels  # Keep the original text labels for reference
    
    # Export the new DataFrame to a separate Excel file
    output_df.to_excel(output_file, index=False, sheet_name='Sheet1')
    print(f"\nExported labeled news to {output_file}")
    
    # Print summary statistics
    print(f"\nLabeling Summary:")
    print(f"Total rows processed: {len(df)}")
    print(f"Successful labels: {len([x for x in numeric_labels if x is not None])}")
    print(f"Positive (1): {numeric_labels.count(1)}")
    print(f"Negative (-1): {numeric_labels.count(-1)}")
    print(f"Neutral (0): {numeric_labels.count(0)}")
    
    # Verify that all numeric labels are valid
    invalid_labels = [x for x in numeric_labels if x is not None and x not in [0, -1, 1]]
    if invalid_labels:
        print(f"Warning: Found invalid numeric labels: {invalid_labels}")
    else:
        print("✓ All numeric labels are valid (0, -1, 1)")

if __name__ == "__main__":
    main()
