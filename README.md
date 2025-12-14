# 🌊 **S.A.G.A.R**
Smart Aquatic Genomic Analysis & Research

S.A.G.A.R is an AI-powered marine eDNA analysis platform designed to uncover hidden biodiversity in deep-ocean ecosystems by combining environmental DNA (eDNA), Generative AI–based representation learning, and intelligent machine learning algorithms for species discovery and conservation.

_**Problem Statement :**_

The deep ocean remains one of the least explored regions on Earth, with more than 80% still undiscovered. These ecosystems host a wide range of organisms, including novel, rare, endangered, and invasive species, yet traditional biodiversity monitoring is limited due to inaccessibility, high operational costs, and ecological sensitivity.

Environmental DNA (eDNA) allows non-invasive biodiversity monitoring by analyzing genetic material present in water and sediment samples. However, existing eDNA analysis approaches suffer from:

Strong dependence on incomplete reference databases

Inability to generalize to deep-sea organisms

Poor handling of unknown and novel species

Time-consuming and manual workflows

Limited interpretability for conservation decision-making

As a result, a significant portion of marine biodiversity remains undetected.

_**Solution Overview**_

S.A.G.A.R addresses these challenges by integrating DNABERT-based GenAI embeddings, supervised learning for known species, and unsupervised clustering for novel species discovery, along with an AI-powered chatbot for interpretation.

_**Detailed Pipeline:**_

1️. eDNA Data Collection

Input: Water and sediment samples

Output: Raw FASTQ sequences

Ensures non-invasive biodiversity monitoring.

2️. FASTQ Upload & Parsing

FASTQ parsing using Biopython

Why used:
Efficient extraction of DNA reads without information loss.

3️. DNA Sequence Representation

here we do 

k-mer tokenization (k = 6)

Optional reverse-complement augmentation

Why used:

Preserves local genomic signatures

Alignment-free and scalable

Matches DNABERT’s pretraining assumptions

4️. GenAI-Based DNA Representation Learning

_Core Algorithm:_

DNABERT (Transformer-based DNA Language Model)

Why used:

Learns contextual patterns in DNA sequences

Captures evolutionary and functional signals

Generates dense embeddings even for unknown organisms

Reduces dependence on reference databases

* Role in S.A.G.A.R:

DNABERT is used only for feature extraction, not classification.

5️. Known Species Discovery

Algorithm Used:

RandomForestClassifier (Supervised Learning)

Why Random Forest?

Handles high-dimensional DNABERT embeddings effectively

Robust to noise common in eDNA data

Works well with limited and imbalanced training data

Provides feature importance and interpretability

Reduces overfitting compared to deep classifiers

Output:

Known species label

Confidence score (probability estimate)

6️. Unknown Species Discovery

Algorithm Used:

HDBSCAN (Hierarchical Density-Based Spatial Clustering)

Why HDBSCAN?

Does not require predefining number of clusters

Automatically identifies noise and outliers

Ideal for biological data with uneven density

Discovers natural groupings corresponding to potential novel species

Output:

Clusters of unknown genetic sequences

Each cluster represents a candidate novel species or lineage

7️. Endangered Species Detection

Algorithms Used:

Relative abundance estimation

Low-frequency statistical filtering

Why used:
Rare but consistent detections may indicate endangered species, enabling conservation prioritization.

8️. Invasive Species Detection

Algorithms Used:

Temporal abundance change detection

Spatial distribution comparison

Why used:
Sudden appearance or rapid growth patterns signal potential invasiveness.

9️. Biodiversity Metrics & Ecosystem Insights

Algorithms Used:

Species richness estimation

Shannon and Simpson diversity indices

Why used:
Provides quantitative ecosystem health indicators for scientists and policymakers.

10. AI Chatbot for Explanation & Decision Support

_Important_: The chatbot does not analyze DNA.

Algorithms Used:

Large Language Models (LLMs)

Intent detection

Context-grounded prompt engineering

Why used:

Translates complex AI outputs into human-understandable insights

Answers general questions and result-specific queries

Prevents hallucinations by grounding responses in analysis outputs

Chatbot LLM	Natural language explanation & assistance

DNABERT and the chatbot are independent GenAI systems, each serving a different purpose.

_**Technology Stack**_

**AI & ML**

DNABERT (Transformer-based DNA Language Model)

RandomForestClassifier (known species)

HDBSCAN (unknown species)

Scikit-learn

PyTorch

**Bioinformatics**

Biopython

FASTQ / FASTA processing

k-mer encoding

**Backend**

FastAPI

Uvicorn

Python

**Frontend**

HTML / CSS / JavaScript / Bootstrap (production UI)

**Chatbot**

LLM (OpenAI / Gemini)

Prompt engineering

Context injection

**Final Outcome:**

By combining DNABERT-based GenAI embeddings, RandomForest classification, HDBSCAN-based novel species discovery, and an AI-powered explanation assistant, S.A.G.A.R enables:

Accurate identification of known marine species

Discovery of previously unrecorded deep-sea organisms

Early detection of endangered and invasive species

Reduced dependence on reference databases

Faster, scalable, and interpretable biodiversity analysis
