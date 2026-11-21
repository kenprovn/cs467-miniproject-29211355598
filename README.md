# Scholarship and Student Grant Recommender (MP-C06)

[![Course](https://img.shields.io/badge/Course-CS467%20Advanced%20AI-blue.svg)](https://github.com/)
[![Student ID](https://img.shields.io/badge/Student%20ID-29211355598-green.svg)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 1. Student Information

- **Student Full Name:** Nguyễn Minh Khanh
- **Student ID:** 29211355598
- **Class / Section:** CS467C (No. 06)
- **Course Name:** CS467 - Advanced Artificial Intelligence
- **Instructor:** Puri Vikram
- **Project Topic:** Scholarship and Student Grant Recommender
- **Project Code:** MP-C06
- **Repository:** `cs467-miniproject-29211355598`

---

## 2. Project Title and Overview

### Project Title
**Intelligent Scholarship and Student Grant Recommender System (MP-C06)**

### Overview
Navigating international higher education funding is challenging. Thousands of specialized grants, exchange stipends, and research fellowships are published each year with complex eligibility criteria, varying degree requirements, and specialized domain constraints. 

This project delivers an end-to-end, reproducible recommendation engine in Python designed to accurately pair students with optimal scholarships. The system integrates:
1. **Natural Language Processing (NLP)** via TF-IDF vectorization and Cosine Similarity to evaluate eligibility texts, focus tags, and unstructured student research queries.
2. **Tri-Method Recommendation Engine**:
   - **Popularity Baseline** (Bayesian-weighted global ratings).
   - **NLP Content-Based Filtering** (semantic matching on student profiles and item-to-item similarity).
   - **Hybrid Recommender** (combining Content-Based semantic relevance with User-Based Collaborative Filtering).
3. **Rigorous Offline Evaluation** using an 80/20 train/test user-partition without data leakage, producing real metric evaluations (Precision@5) and visual diagnostic charts.

---

## 3. Problem Statement

Students frequently encounter information overload when searching for academic funding. Traditional keyword search engines or basic database queries suffer from key limitations:
- **Lexical Mismatch**: Students phrase their interests ("undergraduate computer science student seeking AI research summer exchange") using natural language, whereas grant documents use formal eligibility prose ("laboratory placement in deep learning for bachelor informatics candidates").
- **Cold-Start Vulnerability**: Newly announced grants have zero historical student ratings, causing collaborative filtering algorithms to fail immediately.
- **Over-Generalization**: Global popularity metrics only highlight famous generic grants, overlooking specialized fellowships suited for niche student backgrounds.

Our system resolves these challenges by bridging semantic NLP embeddings with collective rating patterns.

---

## 4. Dataset Description and Statistics

> **Notice:** The dataset used is a synthetic yet highly realistic scholarship corpus generated specifically to simulate global academic funding programs.

The dataset comprises two files in `data/`:

### 4.1. `data/items.csv` (Scholarship Catalog)
- **Total Scholarships ($N$):** 40 items (`SCH_01` to `SCH_40`).
- **Attributes:**
  - `item_id`: Unique scholarship identifier.
  - `grant_name`: Formal grant title.
  - `country`: Host country (USA, Germany, UK, Singapore, Canada, Switzerland, Japan, Sweden, Australia, France, etc.).
  - `degree_level`: Target degree level (Undergraduate, Master, PhD, Exchange).
  - `eligibility_text`: Unstructured text detailing eligibility criteria, academic prerequisites, and fellowship duties.
  - `focus_tags`: Comma-separated domain keywords.
- **Domains Represented:** Artificial Intelligence, Computer Systems, Data Science, Cybersecurity, Robotics, Biomedical Engineering, Public Health, FinTech, Sustainability, Environmental Ecology, Digital Humanities, Quantum Computing, and Aerospace Engineering.
- **Target Query Benchmark:** Includes 6 specialized undergraduate AI/CS summer exchange programs (`SCH_01`, `SCH_02`, `SCH_03`, `SCH_04`, `SCH_05`, `SCH_40`).

### 4.2. `data/ratings.csv` (Student Ratings & Feedback)
- **Total Students ($U$):** 15 unique student profiles (`U01` to `U15`).
- **Ratings per Student:** Exactly 16 ratings per student.
- **Total Ratings ($R$):** $15 \times 16 = 240$ ratings.
- **Rating Scale:** 1 to 5 integer scale:
  - `5`: Perfect alignment with major, degree, and research trajectory.
  - `4`: High relevance and strong interest.
  - `3`: Moderate/partial fit (e.g., related field but degree mismatch).
  - `2`: Low relevance.
  - `1`: Complete mismatch.
- **Interaction Matrix Sparsity:**
  $$\text{Total Matrix Cells} = 15 \times 40 = 600$$
  $$\text{Observed Cells} = 240$$
  $$\text{Missing Entries} = 600 - 240 = 360$$
  $$\text{Sparsity} = \frac{360}{600} = 60.0\%$$
- **Integrity Validation:** 0 null values, 0 duplicate user-item pairs, reproducible fixed random seed (`seed = 42`).

---

## 5. NLP Preprocessing and TF-IDF

The NLP module (`src/nlp_pipeline.py`) normalizes text and extracts semantic features:

### 5.1. Pipeline Steps
1. **Lowercasing:** Converts text to lowercase to ensure case insensitivity.
2. **Punctuation & Special Character Removal:** Cleans regex symbols, brackets, and punctuation (`[^a-z0-9\s]`).
3. **Whitespace Normalization:** Trims excess spaces, tabs, and newlines.
4. **Tokenization & Stop-word Filtering:** Strips standard English stop words (`the`, `is`, `at`, `which`, etc.) while retaining domain-critical tokens.
5. **Corpus Synthesis:** Concatenates `grant_name`, `degree_level`, `country`, `focus_tags`, and `eligibility_text` into a dense document for each scholarship.
6. **Vectorization:** Fits a `TfidfVectorizer` with:
   - N-gram range: $(1, 2)$ (capturing key bigrams such as `"summer exchange"`, `"computer science"`, `"machine learning"`, `"ai research"`).
   - Sublinear term frequency scaling: $\text{tf}' = 1 + \log(\text{tf})$.
   - $L_2$ vector normalization.
7. **Single-Model Consistency:** Uses the identical fitted vectorizer instance for both the catalog and incoming free-text student queries.

### 5.2. Salient Keyword Extraction
To explain recommendations, the pipeline computes the coordinate-wise Hadamard product between the query vector $\mathbf{q}$ and the document vector $\mathbf{d}_i$:
$$w_k = q_k \cdot d_{ik}$$
Features with the highest $w_k > 0$ correspond to the most influential shared terms (e.g., `["exchange", "summer exchange", "ai research", "undergraduate"]`).

---

## 6. Three Recommendation Approaches

### Method 1 — Popularity Baseline
Ranks scholarships using Bayesian weighted popularity (similar to the IMDb weighted rating formula):
$$WR(i) = \left(\frac{v_i}{v_i + m}\right) R_i + \left(\frac{m}{v_i + m}\right) C$$
- $v_i$: Number of ratings received by scholarship $i$.
- $m$: Minimum rating threshold ($m = 2$).
- $R_i$: Arithmetic average rating for scholarship $i$.
- $C$: Global mean rating across all 240 ratings ($C \approx 3.14$).
- **Filtering:** When recommending for a specific student, all scholarships already rated by that student are excluded from the Top-5 output.

### Method 2 — NLP Content-Based Filtering
Directly maps student preferences to scholarship feature vectors in the TF-IDF vector space:
- **Query / Profile Matching:** Transforms free-text student statements into vector $\mathbf{q}$ and computes cosine similarity against all document vectors $\mathbf{d}_i$.
- **User Rating Profile:** Aggregates TF-IDF vectors of items previously rated favorably ($\ge 3.0$) by the user, weighted by their rating score, to form a personalized user preference vector $\mathbf{u}$.
- **Item-to-Item Similarity:** Given seed scholarship $i$, finds scholarships $j \ne i$ maximizing $\text{CosineSim}(\mathbf{d}_i, \mathbf{d}_j)$ (strictly excluding item $i$).

### Method 3 — Hybrid Recommendation
Synthesizes NLP Content-Based Filtering with User-Based Collaborative Filtering (UBCF):
$$\text{Score}_{\text{Hybrid}}(u, i) = \alpha \cdot \text{Score}_{\text{CB}}(u, i) + (1 - \alpha) \cdot \text{Score}_{\text{CF}}(u, i)$$

- **Collaborative Signal:** Computes user-user Pearson correlations on mean-centered ratings:
  $$\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N(u), \text{sim}(u,v) > 0} \text{sim}(u, v) (r_{v,i} - \bar{r}_v)}{\sum_{v \in N(u), \text{sim}(u,v) > 0} |\text{sim}(u, v)|}$$
  The predicted rating $\hat{r}_{u,i} \in [1, 5]$ is normalized to $[0, 1]$.
- **Weighting Rationale ($\alpha = 0.5$):** Equal weighting balances explicit semantic alignment (degrees, skills, research interests) with peer voting signals.
- **Cold-Start Fallback:** When evaluating a brand-new scholarship ($v_i = 0$) or a new user without ratings, the hybrid system automatically sets $\alpha = 1.0$ (reverting to pure Content-Based), preventing missing-value exceptions or degraded scores.

---

## 7. Cosine Similarity Explanation

Cosine similarity measures the cosine of the angle between two multi-dimensional vectors in a Euclidean space:
$$\text{CosineSim}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2} = \frac{\sum_{k=1}^{V} A_k B_k}{\sqrt{\sum_{k=1}^{V} A_k^2} \sqrt{\sum_{k=1}^{V} B_k^2}}$$

### Why Cosine Similarity for Scholarship Retrieval?
1. **Length Invariance:** Scholarship descriptions vary in length (some programs write 30 words, others 150 words). Euclidean distance penalizes longer documents simply due to word volume, whereas Cosine Similarity measures directional semantic orientation.
2. **Normalized Bounding:** Since TF-IDF weights are non-negative ($A_k \ge 0$), the metric produces values bounded in $[0, 1]$, where:
   - `1.0`: Identical term weighting and semantic orientation.
   - `0.0`: Orthogonal vectors sharing zero terms.
3. **Computational Efficiency:** For $L_2$-normalized sparse TF-IDF vectors, the denominator $\|\mathbf{A}\|_2 \|\mathbf{B}\|_2 = 1$, reducing the calculation to a fast matrix dot-product: $\mathbf{A} \mathbf{B}^T$.

---

## 8. Installation Instructions

### Prerequisites
- Python 3.9+ (tested on Python 3.13)
- Git

### Setup
Clone the repository and install dependencies:
```powershell
git clone https://github.com/your-username/cs467-miniproject-29211355598.git
cd cs467-miniproject-29211355598

# (Optional) Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

---

## 9. How to Run

### 9.1. Run Default Demo (All 3 Test Cases + Sample Recommendations)
```powershell
python -m src.main
```

### 9.2. Run Offline Evaluation & Generate Charts
```powershell
python -m src.evaluate
# or via CLI flag:
python -m src.main --eval
```

### 9.3. Custom Query Search
```powershell
python -m src.main --profile "undergraduate computer science student seeking AI research summer exchange"
```

### 9.4. Item-to-Item Similarity Search
```powershell
python -m src.main --similar SCH_01
```

### 9.5. Compare All Three Algorithms for a Student
```powershell
python -m src.main --user U01
```

### 9.6. Regenerate Dataset & Validate
```powershell
python -m src.generate_dataset
```

---

## 10. Three Required Test Cases and Actual Outputs

The actual console outputs captured from `results/sample_outputs.txt`:

### Test Case 1: Student Profile Query
**Input Query:** `"undergraduate computer science student seeking AI research summer exchange"`

| Rank | Item ID | Score | Country | Degree | Grant Name | Salient Matching Keywords |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 1 | `SCH_04` | 0.4544 | Singapore | Undergraduate | Asia-Pacific Summer Research Exchange in Computer Science | `exchange, summer, summer exchange, ai research, undergraduate computer` |
| 2 | `SCH_01` | 0.4014 | USA | Undergraduate | Global AI Undergraduate Summer Research Fellowship | `summer, exchange, summer exchange, ai research, undergraduate` |
| 3 | `SCH_40` | 0.3431 | USA | Undergraduate | Global Undergraduate Summer STEM Exchange Program | `summer, exchange, summer exchange, undergraduate, seeking` |
| 4 | `SCH_02` | 0.3082 | Germany | Undergraduate | DAAD International CS & AI Summer Exchange Grant | `exchange, summer exchange, summer, ai, undergraduate` |
| 5 | `SCH_03` | 0.2458 | UK | Undergraduate | Turing Future Computing & AI Summer Scholar Award | `summer, exchange, undergraduate, summer exchange, ai` |

*Verification:* All Top-5 results are undergraduate programs offering summer research exchange in computing and AI, directly fulfilling the query with high relevance.

---

### Test Case 2: Similar Scholarship (Item-to-Item)
**Seed Item:** `[SCH_01] Global AI Undergraduate Summer Research Fellowship` (USA, Undergraduate, AI & Summer Exchange)

| Rank | Item ID | Sim Score | Country | Degree | Similar Grant Name | Shared Salient Keywords |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 1 | `SCH_03` | 0.2671 | UK | Undergraduate | Turing Future Computing & AI Summer Scholar Award | `machine learning, summer, machine, undergraduate, exchange` |
| 2 | `SCH_05` | 0.2641 | Canada | Undergraduate | Canada AI & Machine Intelligence Undergraduate Research Grant | `undergraduate, summer, learning, machine, intelligence` |
| 3 | `SCH_40` | 0.2584 | USA | Undergraduate | Global Undergraduate Summer STEM Exchange Program | `summer, exchange, undergraduate, summer exchange` |
| 4 | `SCH_04` | 0.2563 | Singapore | Undergraduate | Asia-Pacific Summer Research Exchange in Computer Science | `summer, exchange, undergraduate, summer exchange, ai` |
| 5 | `SCH_02` | 0.1942 | Germany | Undergraduate | DAAD International CS & AI Summer Exchange Grant | `summer, summer exchange, exchange, undergraduate, ai` |

*Verification:* `SCH_01` is strictly excluded from its own recommendation list. The Top-5 similar scholarships are all undergraduate AI/CS summer exchange programs across the UK, Canada, USA, Singapore, and Germany.

---

### Test Case 3: Cold-Start Grant Evaluation
**Scenario:** A newly announced grant with zero ratings:
- **Title:** *Turing Post-Quantum AI & Deep Learning Immersion Exchange*
- **Level & Country:** Undergraduate | UK
- **Eligibility:** *Open to international undergraduate students in computer science, quantum information, and machine learning. Provides full stipend for an intensive 10-week summer research exchange laboratory placement exploring neural networks and quantum artificial intelligence.*
- **Ratings in Database:** `0` (Completely unseen cold-start item).

**Result against Target Query:**
- **Cosine Similarity Score:** `0.2969`
- **Salient Overlapping Terms:** `exchange, summer, undergraduate, summer exchange, ai`
- **Status:** Successfully matched and retrieved via NLP Content-Based Filtering without requiring historical ratings.

---

## 11. Evaluation Methodology

1. **User Partitioning (80/20 Train/Test Split):**
   - Each of the 15 students rated 16 scholarships.
   - For each user, ratings were split into 80% train (12–13 ratings, 180 total) and 20% test (3–4 ratings, 60 total).
   - This ensures every student is evaluated without user cold-start skew.
2. **Zero Data Leakage:**
   - Popularity statistics, user similarity matrices, and user profile vectors are computed exclusively on `train_df`.
   - Test partition ratings are hidden until evaluation time.
3. **Relevance Threshold:**
   - Ground-truth relevance is defined as $\text{Rating} \ge 4.0$.
4. **Metric: Precision@5:**
   - For each user $u$, candidate items are scholarships not in $u$'s training set.
   - The model generates its Top-5 recommendations.
   - A hit occurs when a recommended item appears in the user's test set with $\text{rating} \ge 4.0$:
     $$\text{Precision@5} = \frac{\text{Hits in Top 5}}{5}$$

---

## 12. Actual Evaluation Results

The offline evaluation yielded the following real results:

| Recommendation Method | Precision@5 Score | Description & Behavior |
|:---|:---:|:---|
| **Popularity Baseline** | **0.0727** | Ranks items by Bayesian rating count and average score. Ignores individual academic domain, causing lower domain precision across varied majors. |
| **Hybrid Recommender** | **0.1091** | Blends 50% Content-Based semantic relevance and 50% Collaborative Filtering. Improves over baseline while adapting to peer preferences. |
| **NLP Content-Based** | **0.1818** | Directly maps student profile embeddings to grant eligibility vectors. Achieves highest precision in a domain-specialized dataset. |

### Analytical Explanation of Metric Scores
- In our 80/20 split, each user's test set holds only 3 to 4 items, of which typically 1 to 2 items have rating $\ge 4.0$.
- Because the maximum possible hits in Top-5 for a user with only 1 or 2 relevant test items is 1 or 2, the theoretical upper bound of Precision@5 for such users is $\frac{1}{5} = 0.20$ or $\frac{2}{5} = 0.40$.
- Hence, an empirical score of **0.1818** for Content-Based represents strong retrieval power, outperforming the popularity baseline by **2.5x**.

---

## 13. Generated Charts

The evaluation generates two high-resolution charts saved in `results/`:

1. **`results/rating_distribution.png`**:
   - Visualizes the distribution of all 240 relevance ratings (1 to 5) across the 15 student profiles, demonstrating a healthy balance of high, neutral, and low ratings without artificial skew.

2. **`results/method_comparison.png`**:
   - Directly compares the offline Precision@5 achieved by Popularity Baseline (0.0727), Hybrid (0.1091), and NLP Content-Based (0.1818).

---

## 14. Limitations and Short Reflection

### Limitations
1. **TF-IDF Bag-of-Words Representation:** While n-grams capture common compound phrases (e.g., "summer exchange", "machine learning"), TF-IDF does not capture deep semantic polysemy as effectively as transformer embeddings (e.g., Sentence-BERT).
2. **Interaction Matrix Scale:** The dataset consists of 15 students and 40 grants (240 ratings). While ideal for demonstrating recommender mechanics and maintaining 60% sparsity, Collaborative Filtering requires hundreds of users to unlock dense latent factor interactions.
3. **Hard Filter Constraints:** The current engine uses soft cosine similarity. In production systems, hard constraints (e.g., strict GPA minimums or citizenship criteria) should act as pre-filters before ranking.

### Reflection
Building this project demonstrated that in domain-specific recommender systems (such as academic scholarships and medical grants), **NLP Content-Based representations are significantly more reliable than Collaborative Filtering** during cold-start and sparse conditions. Hybridization offers a balanced path forward as the user community scales.

---

## 15. Project Folder Structure

```text
cs467-miniproject-29211355598/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── items.csv
│   └── ratings.csv
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py
│   ├── nlp_pipeline.py
│   ├── recommender.py
│   ├── evaluate.py
│   └── main.py
└── results/
    ├── sample_outputs.txt
    ├── rating_distribution.png
    └── method_comparison.png
```

---
*Created for CS467 Advanced Artificial Intelligence - Semester 2026. Student: Nguyễn Minh Khanh (29211355598).*
