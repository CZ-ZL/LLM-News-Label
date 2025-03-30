import os
import pandas as pd
import openai
import re
import docx

# Set your DeepSeek API key and base URL
openai.api_key = "API Key"  # Replace with your actual API key
openai.api_base = "https://api.deepseek.com"

# File paths: update as needed.
input_file = "C:\\Users\\29806\\Desktop\\test data.xlsx"
prompt_file = "C:\\Users\\29806\\Desktop\\Prompt 3.docx"
output_file = "C:\\Users\\29806\\Desktop\\Labeled_News.xlsx"   

def load_prompt_template(filepath):
    """
    Load and return the prompt template from a DOCX file.
    The template should contain a placeholder `{news}` where the news text will be inserted.
    """
    doc = docx.Document(filepath)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return full_text

# Load the prompt template from the DOCX file.
prompt_template = load_prompt_template(prompt_file)

def map_impact_to_numeric(impact_str):
    """
    Map the plain text impact value to a numeric label:
      - "Positive" -> 1
      - "Neutral"  -> 0
      - "Negative" -> -1
    If the impact is not recognized, defaults to 0.
    """
    mapping = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }
    # Normalize the string: remove extra spaces and capitalize first letter.
    impact_str = impact_str.strip().capitalize()
    return mapping.get(impact_str, 0)

def label_news_item(news_item):
    """
    Sends a news item to the DeepSeek API using a prompt template and returns the numeric label.
    Assumes that the API returns a plain text output that is one of "Positive", "Neutral", or "Negative".
    """
    # Insert the news text into the prompt template.
    
    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": news_item},
        ],
        stream=False
    )
    
    # Get the raw output (plain text) and print it for debugging.
    result = response.choices[0].message.content

    return result

def extract_impact(label_response):
    """
    Extracts the impact value from the API response string.
    It looks for a pattern like '"impact": "Value"' and returns the Value.
    """
    match = re.search(r'"impact":\s*"([^"]+)"', label_response)
    if match:
        return match.group(1)
    else:
        # Fallback: return the original response stripped
        return label_response.strip()

def main():
    # Read the input Excel file (news items assumed to be in the first column).
    df = pd.read_excel(input_file, header=0)
    
    # List to store the numeric labels.
    numeric_labels = []
    
    # Process each news item.
    for index, row in df.iterrows():
        news_item = row.iloc[1]
        if pd.isna(news_item):
            print(f"Row {index} is empty. Skipping...")
            numeric_labels.append(None)
            continue
        
        print(f"\nProcessing row {index}: {str(news_item)[:60]}...")
        try:
            # Retrieve the label text from the API.
            label_text = label_news_item(str(news_item))
            impact_str = extract_impact(label_text)
            # Map the label text to its numeric equivalent.
            numeric_label = map_impact_to_numeric(impact_str)
            numeric_labels.append(numeric_label)
            print(f"Row {index} label: {label_text} -> numeric label: {numeric_label}")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            numeric_labels.append(None)
    
    # Create a new DataFrame for output.
    output_df = df.copy()
    output_df["Numeric Label"] = numeric_labels
    
    # Export the new DataFrame to a separate Excel file.
    output_df.to_excel(output_file, index=False)
    print(f"\nExported labeled news to {output_file}")

if __name__ == "__main__":
    main()
