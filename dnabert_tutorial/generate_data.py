import random
import pandas as pd
import numpy as np

def generate_dna(length, gc_content):
    """
    Generate a random DNA sequence with a target GC content.
    """
    g_or_c = int(length * gc_content)
    a_or_t = length - g_or_c
    
    bases = ['G'] * (g_or_c // 2) + ['C'] * (g_or_c - g_or_c // 2) + \
            ['A'] * (a_or_t // 2) + ['T'] * (a_or_t - a_or_t // 2)
            
    random.shuffle(bases)
    return "".join(bases)

def generate_dataset(num_samples_per_class=500, seq_length=100):
    data = []
    
    # Define classes with distinct GC content signatures for synthetic separation
    domains = {
        "Bacteria": 0.50,    # Balanced
        "Archaea": 0.40,     # Generally lower GC in some contexts (simplified)
        "Eukaryotes": 0.45,  # Varied
        "Viruses": 0.55      # Varied
    }
    
    # To ensure model learns motifs and not just GC content, 
    # we would ideally insert specific motifs, but for a basic tutorial,
    # GC content + random k-mer patterns is sufficient for clear convergence.
    
    print(f"Generating {num_samples_per_class} sequences per domain...")
    
    for label, gc in domains.items():
        for _ in range(num_samples_per_class):
            # Add some variance to GC content
            local_gc = np.clip(np.random.normal(gc, 0.05), 0.2, 0.8)
            seq = generate_dna(seq_length, local_gc)
            data.append({"sequence": seq, "label": label})
            
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    output_file = "synthetic_dna_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    generate_dataset()
