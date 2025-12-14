import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Configuration
MODEL_NAME = "armheb/DNA_bert_6"
DATA_FILE = "synthetic_dna_data.csv"
OUTPUT_DIR = "dnabertmodel"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {label: i for i, label in enumerate(sorted(set(labels)))}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # K-mer tokenization (k=6)
        # DNABERT expects space-separated k-mers: "ATGCGT TGCGTA ..."
        k = 6
        kmers = " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])
        
        encoding = self.tokenizer.encode_plus(
            kmers,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long)
        }

def train():
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run generate_data.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    seqs = df['sequence'].tolist()
    labels = df['label'].tolist()
    
    unique_labels = sorted(list(set(labels)))
    num_labels = len(unique_labels)
    print(f"Classes: {unique_labels}")
    
    # Split
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(seqs, labels, test_size=0.2, random_state=42)
    
    print("Initializing Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels, 
        trust_remote_code=True
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on {device}")
    model.to(device)
    
    train_dataset = DNADataset(train_seqs, train_labels, tokenizer, MAX_LEN)
    val_dataset = DNADataset(val_seqs, val_labels, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_true, val_preds)  
        print(f"Validation Accuracy: {val_acc:.4f}")
        
    print("\nSaving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save label map for inference
    import json
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(unique_labels, f)
        
    print(f"Model saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    train()
