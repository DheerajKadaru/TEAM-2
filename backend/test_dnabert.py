import os
import sys
import logging
from models.dnabert_model import DNABERT_SpeciesIdentifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_fasta(filename="test_input.fasta"):
    """Create a dummy FASTA file with synthetic DNA sequences."""
    sequences = [
        # Species A (Cluster 1-ish)
        "ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGCGTACGT",
        "ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGCGTACGA",
        "ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGCGTACGC",
        # Species B (Cluster 2-ish)
        "GGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCC",
        "GGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCA",
        # Unknown (Random)
        "AAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAA", 
        "AAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGATA"
    ]
    
    with open(filename, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")
            
    return filename

def main():
    logger.info("Starting DNABERT Model Test...")
    
    # 1. Initialize
    try:
        # Use a small cache dir to avoid global pollution
        # armheb/DNA_bert_6 is a standard BERT model (k=6)
        model = DNABERT_SpeciesIdentifier(model_name="armheb/DNA_bert_6")
        logger.info("Model Initialized Successfully.")
    except Exception as e:
        logger.error(f"Initialization Failed: {e}")
        return

    # 2. Mock Training
    # Note: Sequences here will be converted to k-mers by preprocess, 
    # but since train_known expects 'sequences' (which could be preprocessed), 
    # we should handle that. In real flow, input is fastq path or raw strings.
    # The current 'train_known' assumes preprocessed inputs if called directly?
    # Actually, train_known calls get_embeddings. preprocess returns k-mer strings.
    # So we should pass RAW sequences if we want robust testing or simulate preprocessed.
    # Let's pass pre-kmerized strings for simplicity in this unit test or modify the class to be flexible.
    # Since DNABERT_SpeciesIdentifier.preprocess does the k-mer conversion, 
    # and train_known takes the output of preprocess...
    # Let's just create k-mer strings here:
    
    def to_kmer(seq):
        return " ".join([seq[i:i+6] for i in range(len(seq) - 6 + 1)])

    train_seqs = [
        to_kmer("ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGCGTACGT"),
        to_kmer("ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGCGTACGA"),
        to_kmer("ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGCGTACGC"),
        to_kmer("GGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCC"),
        to_kmer("GGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCA")
    ]
    train_labels = ["Species_A", "Species_A", "Species_A", "Species_B", "Species_B"]
    
    logger.info("Training on dummy data...")
    model.train_known(train_seqs, train_labels)
    
    # 3. Predict
    test_file = create_dummy_fasta()
    logger.info(f"Predicting on {test_file}...")
    results = model.predict(test_file)
    
    import json
    print("\n--- INFERENCE RESULTS ---")
    print(json.dumps(results, indent=2))
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    main()
