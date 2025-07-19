# LLM News Label

This repository contains scripts and data for labeling foreign-exchange news with large language models (LLMs).  
The project experiments with both zero-shot prompts and fine-tuned models for sentiment classification.

## Repository Structure

- **`zero shot train data/`** – example news and price files used to build a zero-shot training dataset.
- **`testing data/`** – spreadsheets and prompts for evaluating the models.
- **`Signal Analysis/`** – additional datasets, prompts and generated confusion matrix images.
- **`*.py` scripts** – utilities for labeling news, cleaning results and computing metrics:
  - `news label.py` – joins news with FX data and labels each article based on a price threshold.
  - `LLM_Deepseek_labeling.py` – labels news using the DeepSeek API.
  - `zero shot label.py` – applies a zero-shot prompt to label news articles.
  - `zero-shot fine tune model.py` – run a fine‑tuned OpenAI model to label batches of articles.
  - `confusion matrix calculation.py` – append confusion matrices and precision/recall metrics to an Excel file.
  - `file clean.py` – remove empty rows/columns from Excel outputs.

Several text files such as `Zero shot command`, `Fine Tune Model Command` and `Testing Command` contain example command lines for running these scripts.  
Images of confusion matrices are stored at the repository root and under `Signal Analysis/cm`.

## Setup

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Some scripts require additional packages listed in `freeze.txt` as well as API keys for OpenAI or DeepSeek.

## Example Workflow

1. Generate a labeled training set by combining exchange-rate data with news:

```bash
python "news label.py" --fx_file "zero shot train data/Zero Shot Training Currency File.xlsx" \
       --news_file "zero shot train data/Zero Shot News File.xlsx" \
       --spread 0.0005 --output_file "zero shot train data/Zero Shot News With Label.xlsx"
```

2. Label new articles using a prompt or fine‑tuned model (see `Testing Command` for full steps):

```bash
python "LLM_Deepseek_labeling.py" --input_file "testing data/test news file.xlsx" \
       --prompt_file "testing data/LLM_Prompt.docx" --output_file "testing data/test news file.xlsx"
```

3. Clean the output, then compute confusion matrices and precision/recall metrics:

```bash
python "file clean.py" --input_file "testing data/test news file.xlsx" --output_file "testing data/test news result.xlsx"
python "confusion matrix calculation.py" --input_file "testing data/test news result.xlsx" \
       --data_sheet "Sheet1" --output_file "testing data/test news result.xlsx"
```

The resulting Excel file will contain the labeled data along with evaluation metrics.

## License

This repository does not include a license file. If you plan to use or share the code, please consult the project owner.
