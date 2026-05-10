# AI Job Market Persona Discovery - Deep Learning

This project applies deep learning text representations to discover AI job market 
personas from job description text across six European countries. Building directly
on the previous project, which clustered DACH job postings using keyword-extracted skill
profiles - this project replaces hand-crafted features with neural text
representations and compares a trained PyTorch autoencoder (baseline) against
pre-trained DistilBERT embeddings (experimental). The core question is whether
Transformer-based semantic representations produce more coherent role archetypes
than classical TF-IDF vectorisation.

**Dataset:** Custom collection via the Adzuna Jobs API - 1,328 job postings across
Germany, United Kingdom, France, Netherlands, Austria and Switzerland, enriched
to 1,088 modeling-ready entries with full descriptions  
**Source:** Collected using `fetch_ai_jobs_personas.py`, enriched using
`enrich_descriptions.py`

---

## How to Run the Project

**1. Clone the repository**

    git clone https://github.com/n1n4ch/deep-learning-systems.git
    cd deep-learning

**2. Install dependencies**

    pip install -r requirements.txt

**3. Collect the dataset** *(optional — CSVs already included)*

    python fetch_ai_jobs_personas.py
    python enrich_descriptions.py

**4. Open and run the notebook**

    jupyter notebook deep_learning.ipynb

Run all cells in order via **Kernel → Restart & Run All**

> DistilBERT encoding runs on CPU and takes approximately 10–20 minutes for
> 1,088 job descriptions. An internet connection is required on first run to
> download the pre-trained model weights from HuggingFace (~250MB).

---

## Project Structure

| File | Description |
|------|-------------|
| `deep_learning.ipynb` | Main notebook — data loading, baseline, experimental model, evaluation |
| `fetch_ai_jobs_personas.py` | Adzuna API collection script (DACH + GB, FR, NL) |
| `enrich_descriptions.py` | Description enrichment script — fetches full text from source pages |
| `adzuna_ai_jobs_europe.csv` | Raw job postings (1,328 rows, 17 columns) |
| `adzuna_ai_jobs_europe_enriched.csv` | Enriched job postings with full descriptions (1,328 rows, 19 columns) |
| `Deep_Learning_Systems_Analysis_Report.pdf` | Written report with citations |
| `requirements.txt` | Python dependencies (generated via `pip freeze`) |

---

## Approach

**Data collection** — The Adzuna Jobs API was queried across 30 search terms
spanning five role categories (AI Engineering, Data Analytics, AI Product and
Project Management, AI Integration, AI Consulting) in six countries. English-language
queries run across all countries; German-language queries are restricted to DACH.
A secondary enrichment step fetches full job pages from source URLs, increasing
mean description length from 500 to ~4,000 characters.

**Baseline model** — TF-IDF vectors (500 features) are compressed into a
32-dimensional latent space using a PyTorch autoencoder trained for 50 epochs
with MSE loss and Adam optimiser. K-Means clustering (k=5) is applied to the
latent vectors.

**Experimental model** — Pre-trained DistilBERT encodes each job description
into a 768-dimensional semantic embedding using the [CLS] token representation.
No training is required. K-Means clustering (k=5) is applied identically to the
baseline, isolating the effect of the representation method.

**Evaluation** — Both models are compared using the silhouette score across
k = 2 to k = 8, side-by-side PCA cluster visualisations, and cross-tabulation
of cluster assignments against human-assigned role categories.

---

## Key Findings

Five role archetypes were identified from the DistilBERT experimental model,
reflecting the semantic structure of the European AI job market:

| Cluster | Archetype |
|---------|-----------|
| 0 | Data & Analytics Generalist |
| 1 | Cloud ML Engineer |
| 2 | MLOps & GenAI Engineer |
| 3 | AI Automation & Integration |
| 4 | Deep Learning & AI Research |

The silhouette score comparison between baseline and experimental models
demonstrates whether semantic Transformer embeddings produce more coherent
role archetypes than TF-IDF-based compression. Results are reported in the
notebook and the written report.

---

## Data Bias and Responsible Use

The United Kingdom accounts for 51% of postings (675 of 1,328), meaning the
discovered personas primarily reflect the British AI job market rather than a
balanced European view. Austria and Netherlands are underrepresented with fewer
than 35 postings each. The skill and role patterns identified here should not
be used to filter, rank or screen job applicants - applying cluster labels to
individual candidates would risk encoding labour market biases present in the
training data and falls under high-risk AI system classification under the EU
AI Act. This project is intended for descriptive market analysis only.

---

## Requirements

Regenerate with:

    pip freeze > requirements.txt
