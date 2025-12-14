import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

# Configuration
MODEL_DIR = "dnabertmodel"

st.set_page_config(page_title="DNABERT Analysis", layout="wide")

def predict_dna(sequences):
    """
    Run inference on a list of sequences.
    """
    if not os.path.exists(MODEL_DIR):
        st.error("Model not found! Run train.py first.")
        return []

    # Load Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    # Load Labels
    with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
        labels = json.load(f)

    results = []
    
    # Simple loop for inference (can be batched for speed)
    progress_bar = st.progress(0)
    
    for i, seq in enumerate(sequences):
        # K-mer tokenization
        k = 6
        if len(seq) < k:
            results.append({"Sequence": seq, "Prediction": "Too Short", "Confidence": 0.0})
            continue
            
        kmers = " ".join([seq[j:j+k] for j in range(len(seq) - k + 1)])
        
        inputs = tokenizer(
            kmers, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        results.append({
            "Sequence Snippet": seq[:20] + "...",
            "Prediction": labels[pred_idx.item()],
            "Confidence": f"{conf.item():.4f}"
        })
        progress_bar.progress((i + 1) / len(sequences))
        
    return results

def parse_fasta(file_content):
    """Simple FASTA parser"""
    sequences = []
    current_seq = []
    
    for line in file_content.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if current_seq:
                sequences.append("".join(current_seq))
                current_seq = []
        else:
            current_seq.append(line)
            
    if current_seq:
        sequences.append("".join(current_seq))
        
    return sequences

# UI Layout
st.title("🧬 DNABERT Sequence Analysis")
st.markdown("Upload a FASTA file to classify DNA sequences into **Bacteria, Archaea, Eukaryotes, or Viruses**.")

uploaded_file = st.file_uploader("Upload FASTA File", type=["fasta", "fa", "txt"])

if uploaded_file is not None:
    stringio = uploaded_file.getvalue().decode("utf-8")
    sequences = parse_fasta(stringio)
    
    st.info(f"Loaded {len(sequences)} sequences from file.")
    
    if st.button("Run Analysis"):
        with st.spinner("Analyzing sequences with DNABERT..."):
            results = predict_dna(sequences)
            
        st.success("Analysis Complete!")
        st.table(results)

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This application uses a fine-tuned DNABERT model to classify DNA sequences. "
    "Ensure `train.py` has been run to generate the model files."
)
