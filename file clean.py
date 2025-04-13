import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
# 解析命令行参数
args = parser.parse_args()

# Load the Excel file into a DataFrame.
# Replace 'input.xlsx' with the path to your Excel file.
df = pd.read_excel(args.input_file)

# Optional: Remove leading and trailing whitespace from string values.
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Replace any occurrence of the string "non-data" with NaN.
df.replace("no_data", pd.NA, inplace=True)

# Drop any rows where all the cells are either NaN or empty.
df.dropna(how='any', inplace=True)

# Drop any columns where all the cells are either NaN or empty.
df.dropna(axis=1, how='all', inplace=True)

# Save the cleaned DataFrame back to an Excel file.
# Change 'output.xlsx' to your desired output filename.
df.to_excel(args.output_file, index=False)

print("Excel file cleaned: empty rows/columns, extraneous whitespace, and 'non-data' values removed.")
