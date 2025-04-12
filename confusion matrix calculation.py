import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import argparse

parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--input_file', type=str, required=True, help='input file')
parser.add_argument('--data_sheet', type=str, required=True, help='data_sheet')
args = parser.parse_args()
# Specify the path to your Excel file and the sheet name where your data is stored.
file_path = args.input_file
data_sheet = args.data_sheet

# Read the data from the Excel file.
df = pd.read_excel(file_path, sheet_name=data_sheet)

# Column with the true labels.
true_labels = df['Market Direction']

# List of predicted label columns. Adjust these names as needed.
predicted_columns = ['LLM Prompt Label', 'Expert Label', 'Market Label']

class_labels = [1, -1]

# Set the output sheet name (keep it short to avoid the 31-character limit).
output_sheet = 'Results'

# Open the Excel writer; using mode 'a' appends to the file.
# if_sheet_exists='replace' ensures that any existing sheet with that name is overwritten.
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    # We'll keep track of the current starting row for writing each table.
    startrow = 0

    # Loop through each predicted column and append its results.
    for col in predicted_columns:
        predicted = df[col]

        # Compute the confusion matrix using the specified class label order.
        cm = confusion_matrix(true_labels, predicted, labels=class_labels)
        # Build the confusion matrix DataFrame with the specified format.
        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        cm_df.index.name = "true/label"
        cm_df = cm_df.reset_index()

        # Compute the metrics (pass zero_division=0 to avoid division warnings).
        precision = precision_score(true_labels, predicted, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted, average='weighted', zero_division=0)
        accuracy = accuracy_score(true_labels, predicted)
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'Accuracy'],
            'Score': [precision, recall, accuracy]
        })

        # Write a header indicating which predicted column the results are for.
        header_df = pd.DataFrame([[f"Results for {col}"]], columns=[f"Results for {col}"])
        header_df.to_excel(writer, sheet_name=output_sheet, startrow=startrow, index=False)
        startrow += header_df.shape[0] + 1

        # Write a sub-header for the confusion matrix.
        cm_caption = pd.DataFrame([["Confusion Matrix"]])
        cm_caption.to_excel(writer, sheet_name=output_sheet, startrow=startrow, index=False, header=False)
        startrow += 1

        # Write the formatted confusion matrix.
        cm_df.to_excel(writer, sheet_name=output_sheet, startrow=startrow, index=False)
        startrow += cm_df.shape[0] + 2  # leave a couple of rows space

        # Write a sub-header for the metrics.
        metrics_caption = pd.DataFrame([["Metrics (Precision, Recall, Accuracy)"]])
        metrics_caption.to_excel(writer, sheet_name=output_sheet, startrow=startrow, index=False, header=False)
        startrow += 1

        # Write the metrics DataFrame.
        metrics_df.to_excel(writer, sheet_name=output_sheet, startrow=startrow, index=False)
        startrow += metrics_df.shape[0] + 3  # space before the next prediction block

    # No need to explicitly call writer.save() as the context manager handles it.
    
print("All results have been written to the 'Results' sheet in your Excel file.")