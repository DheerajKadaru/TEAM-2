import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from io import StringIO
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DNABERT_SpeciesIdentifier:
    def __init__(self, model_name: str = "armheb/DNA_bert_6", cache_dir: str = "./backend/models/cache"):
        """
        Initialize DNABERT model (v1, k=6) and classifiers.
        Using armheb/DNA_bert_6 for Windows compatibility (no Triton dependency).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.k = 6 # Default for this model
        logger.info(f"Initializing DNABERT on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load DNABERT model: {e}")
            raise e

        # Ensemble Classifiers (Placeholders - would be loaded from files in real prod)
        self.rf_model = None
        self.xgb_model = None
        self.mlp_model = None
        
        # Unsupervised Models
        self.umap_reducer = None
        self.clusterer = None
        
        # Temporary Known Species Registry for demo training
        self.known_species_map = {}
        
        # Attempt load
        self.load_models()

    def _seq_to_kmers(self, seq: str, k: int = 6) -> str:
        """Convert sequence to space-separated k-mers."""
        return " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])

    def preprocess(self, fastq_path: str, k: int = 6) -> List[str]:
        """
        Read FASTQ/FASTA, filter quality, and return k-mer strings.
        """
        sequences = []
        try:
            # Simple file read handling both FASTA and FASTQ
            file_format = "fasta"
            if fastq_path.endswith(".fastq") or fastq_path.endswith(".fq"):
                file_format = "fastq"
                
            for record in SeqIO.parse(fastq_path, file_format):
                seq_str = str(record.seq).upper()
                # Basic Quality Filtering (Length > k)
                if len(seq_str) > k: 
                    # Convert to k-mer string
                    kmer_str = self._seq_to_kmers(seq_str, k=self.k)
                    sequences.append(kmer_str)
                    
            logger.info(f"Loaded {len(sequences)} sequences from {fastq_path}")
            return sequences
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return []

    def get_embeddings(self, sequences: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings using DNABERT-2.
        """
        all_embeddings = []
        
        # Generator for batches
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            inputs = self.tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token (or mean pooling) - DNABERT-2 often uses mean or CLS
                # Hidden states: [batch, len, dim]
                # Let's take the mean of the last hidden state for robust representation
                embeddings = outputs.last_hidden_state.mean(dim=1) 
                
            all_embeddings.append(embeddings.cpu().numpy())
            
        if not all_embeddings:
            return np.array([])
            
        return np.vstack(all_embeddings)

    def save_models(self, directory: str = "./backend/models/data"):
        """Save trained models and encoders to disk."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        try:
            if self.rf_model:
                joblib.dump(self.rf_model, os.path.join(directory, "rf_model.joblib"))
            if self.xgb_model:
                joblib.dump(self.xgb_model, os.path.join(directory, "xgb_model.joblib"))
            if self.mlp_model:
                joblib.dump(self.mlp_model, os.path.join(directory, "mlp_model.joblib"))
            if hasattr(self, 'label_encoder'):
                joblib.dump(self.label_encoder, os.path.join(directory, "label_encoder.joblib"))
                
            # Also save species map just in case
            joblib.dump(self.known_species_map, os.path.join(directory, "species_map.joblib"))
            
            logger.info(f"Models saved to {directory}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self, directory: str = "./backend/models/data"):
        """Load trained models from disk."""
        try:
            rf_path = os.path.join(directory, "rf_model.joblib")
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                logger.info("Loaded Random Forest model.")
                
            xgb_path = os.path.join(directory, "xgb_model.joblib")
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
                logger.info("Loaded XGBoost model.")
                
            mlp_path = os.path.join(directory, "mlp_model.joblib")
            if os.path.exists(mlp_path):
                self.mlp_model = joblib.load(mlp_path)
                logger.info("Loaded MLP model.")
                
            le_path = os.path.join(directory, "label_encoder.joblib")
            if os.path.exists(le_path):
                self.label_encoder = joblib.load(le_path)
                logger.info("Loaded Label Encoder.")
                
            map_path = os.path.join(directory, "species_map.joblib")
            if os.path.exists(map_path):
                self.known_species_map = joblib.load(map_path)
                logger.info(f"Loaded Species Map with {len(self.known_species_map)} classes.")
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    def train_known(self, sequences: List[str], labels: List[str]):
        """
        Train ensemble classifiers on provided data.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        
        logger.info("Generating embeddings for training...")
        X = self.get_embeddings(sequences)
        
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.known_species_map = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        
        logger.info("Training Ensemble...")
        # 1. Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        
        # 2. XGBoost
        self.xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.xgb_model.fit(X, y)
        
        # 3. MLP
        self.mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        self.mlp_model.fit(X, y)
        
        logger.info("Training complete.")
        self.save_models()

    def discover_novel(self, embeddings: np.ndarray):
        """
        Clustering for novel species detection.
        """
        import umap
        import hdbscan
        
        if len(embeddings) < 5:
            logger.warning("Not enough data for clustering.")
            return [-1] * len(embeddings) # Noise

        logger.info("Running UMAP dimensionality reduction...")
        self.umap_reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
        u_emb = self.umap_reducer.fit_transform(embeddings)
        
        logger.info("Running HDBSCAN clustering...")
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        cluster_labels = self.clusterer.fit_predict(u_emb)
        
        return cluster_labels

    def predict(self, fastq_path: str) -> Dict:
        """
        Main pipeline: Identify known species and cluster unknown ones.
        """
        sequences = self.preprocess(fastq_path)
        if not sequences:
            return {"error": "No valid sequences found."}
            
        embeddings = self.get_embeddings(sequences)
        
        # KNOWN SPECIES PREDICTION (Ensemble Voting)
        known_results = []
        unknown_indices = []
        
        if self.rf_model is None:
             logger.warning("Models not trained. treating all as unknown.")
             unknown_indices = list(range(len(sequences)))
        else:
            rf_probs = self.rf_model.predict_proba(embeddings)
            xgb_probs = self.xgb_model.predict_proba(embeddings)
            mlp_probs = self.mlp_model.predict_proba(embeddings)
            
            # Average probabilities
            avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3
            predictions = np.argmax(avg_probs, axis=1)
            confidences = np.max(avg_probs, axis=1)
            
            novelty_threshold = 0.6 # If confidence < 0.6, treat as unknown
            
            species_counts = {}
            
            for i, (pred_idx, conf) in enumerate(zip(predictions, confidences)):
                if conf >= novelty_threshold:
                    species_name = self.known_species_map.get(pred_idx, "Unknown")
                    if species_name not in species_counts:
                         species_counts[species_name] = {"count": 0, "conf_sum": 0}
                    species_counts[species_name]["count"] += 1
                    species_counts[species_name]["conf_sum"] += conf
                else:
                    unknown_indices.append(i)
            
            for name, stats in species_counts.items():
                known_results.append({
                    "name": name,
                    "confidence": round(stats["conf_sum"] / stats["count"], 2),
                    "abundance": stats["count"]
                })

        # UNKNOWN SPECIES DISCOVERY
        unknown_clusters_res = []
        if unknown_indices:
            unknown_embeddings = embeddings[unknown_indices]
            cluster_labels = self.discover_novel(unknown_embeddings)
            
            cluster_counts = {}
            for label in cluster_labels:
                if label == -1: continue # Noise
                cid = f"Novel_Cluster_{label}"
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            
            unknown_clusters_res = [{"cluster_id": k, "reads": v} for k, v in cluster_counts.items()]

        return {
            "known_species": known_results,
            "unknown_clusters": unknown_clusters_res
        }
