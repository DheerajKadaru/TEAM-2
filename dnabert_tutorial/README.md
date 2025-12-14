# DNABERT Implementation Guide

A complete, end-to-end guide to implementing, training, and testing a DNABERT model for DNA sequence classification.

## Project Structure

```
dnabert_tutorial/
├── generate_data.py    # Generates synthetic multi-domain DNA dataset
├── train.py            # Fine-tunes DNABERT using PyTorch
├── app.py              # Streamlit Web UI for inference
├── requirements.txt    # Project dependencies
└── dnabertmodel/       # (Created after training) Saved model files
```

## 1. Installation

Ensure you have Python installed. Install the required libraries:

```bash
pip install -r requirements.txt
```

*(Note: PyTorch installation may vary based on your OS/CUDA support. Visit [pytorch.org](https://pytorch.org) for specific commands if needed).*

## 2. Dataset Generation

We use a synthetic dataset generator to simulate DNA from four domains: **Bacteria, Archaea, Eukaryotes, and Viruses**.

Run the generator:

```bash
python generate_data.py
```

This will create `synthetic_dna_data.csv` with 2000 sampled sequences.

## 3. Model Training

We fine-tune the pre-trained `armheb/DNA_bert_6` model. This script:
1.  Tokenizes sequences into 6-mers.
2.  Creates a PyTorch Dataset.
3.  Trains for 3 epochs.
4.  Saves the model to `dnabertmodel/`.

Run training:

```bash
python train.py
```

*Note: Training on CPU may take a few minutes. GPU is recommended but not required for this small dataset.*

## 4. Analysis & Inference

Launch the web interface to test the model:

```bash
streamlit run app.py
```

### Usage
1.  Open the URL provided (e.g., `http://localhost:8501`).
2.  Upload a `.fasta` file containing DNA sequences.
3.  Click **Run Analysis**.
4.  View the predicted Class and Confidence score for each sequence.

## Technical Details

### Architecture
-   **Model**: DNABERT (BERT architecture adapted for DNA).
-   **Tokenization**: K-mer tokenization ($k=6$). A sequence like `ATGC...` becomes `ATGCGT TGCGTA ...`.
-   **Loss Function**: CrossEntropyLoss.
-   **Optimizer**: AdamW.

### Customization
To use your own data:
1.  Format your data as a CSV with `sequence` and `label` columns.
2.  Update `DATA_FILE` in `train.py`.
3.  Retrain the model.
