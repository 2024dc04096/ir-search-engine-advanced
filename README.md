# Advanced Information Retrieval Search Engine

An end-to-end **Information Retrieval (IR) system** developed using **Python and Streamlit** for Information Retrieval Assignment 2.

The application demonstrates the complete IR lifecycle through a Streamlit-based interface, including:

- Web crawling
- Duplicate detection
- Text preprocessing and mining
- Inverted indexing
- TF-IDF based retrieval
- PageRank and HITS based ranking
- Content-based, collaborative, and hybrid recommendation
- IR evaluation metrics
- Performance analytics and visualizations

---

## 1. Project Objective

The objective of this project is to implement a cohesive Information Retrieval system that integrates information acquisition, preprocessing, indexing, searching, ranking, recommendation, evaluation, and performance analysis.

The complete workflow is accessible through the **Streamlit front end**.

---

## 2. Features

### Dashboard

Provides an overview of the current IR corpus and system state, including corpus and index statistics.

### Crawling

The crawler supports:

- Multiple seed URLs
- Configurable crawl depth
- URL normalization
- Duplicate URL detection
- Duplicate document detection using content hashing
- HTML text extraction
- Metadata extraction
- Outgoing hyperlink extraction
- Storage of document contents and metadata

Outgoing links are retained so that the crawled document collection can be represented as a directed graph for PageRank and HITS.

### Text Preprocessing and Mining

The preprocessing and text-mining module supports:

- Tokenization
- Lowercasing
- Stop-word removal
- Stemming
- Lemmatization strategy where configured
- Keyword extraction
- Document profiling
- Vocabulary and feature analysis
- KMeans document clustering
- LDA topic modelling
- Comparison of preprocessing strategies

> **Note:** KMeans is used for unsupervised document clustering. It is not presented as supervised document classification.

### Index Management

The application provides an explicit **Build / Rebuild Index** workflow.

The index contains:

- TF-IDF document representation
- Explicit inverted index/postings structure
- Document hyperlink graph
- PageRank scores
- HITS hub scores
- HITS authority scores

The inverted index conceptually follows:

```text
term -> [(document_id, term_frequency), ...]
```

TF-IDF vectors are separately used for query-document similarity calculations.

### Search and Ranking

The search module supports ranked retrieval using:

1. **TF-IDF + Cosine Similarity**
2. **TF-IDF + PageRank**
3. **TF-IDF + HITS Authority**

Queries are preprocessed using the same preprocessing strategy used to build the document index.

For link-aware ranking, textual relevance is combined with normalized PageRank or HITS authority scores.

This allows the effect of link-analysis ranking on retrieval order to be compared with pure TF-IDF retrieval.

### Recommendation System

The recommendation module supports:

- Content-based recommendation
- Collaborative recommendation
- Hybrid recommendation
- Configurable Top-K recommendations
- Similarity/recommendation scores

Content-based recommendations use document similarity.

Collaborative recommendation can use uploaded user-document interaction data. If real interaction data is unavailable, synthetic interactions may be used strictly for algorithm demonstration and are identified as such.

### Evaluation

The evaluation module supports:

- Precision
- Recall
- F1-score
- Precision@K
- Recall@K
- Average Precision (AP)
- Reciprocal Rank (RR)
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- NDCG@K

AP and RR are calculated at the individual-query level.

MAP and MRR are calculated across multiple saved evaluation queries.

The Streamlit interface allows relevance judgments to be recorded and retrieval approaches to be compared using tables and visualizations.

### Performance Analytics

The analytics section provides information and visualizations related to:

- Corpus characteristics
- Index statistics
- Search performance
- Ranking behaviour
- Evaluation results
- Retrieval timing
- Recommendation behaviour

---

## 3. System Architecture

The overall workflow is:

```text
Seed URLs
    |
    v
Web Crawler
    |
    +--> URL Deduplication
    +--> Content Deduplication
    +--> Metadata Extraction
    +--> Hyperlink Extraction
    |
    v
Document Collection
    |
    v
Text Preprocessing
    |
    +-----------------------+
    |                       |
    v                       v
Inverted Index         TF-IDF Matrix
    |                       |
    |                       v
    |                 Cosine Similarity
    |                       |
    v                       |
Document Link Graph         |
    |                       |
    +--> PageRank ----------+
    |
    +--> HITS --------------+
                            |
                            v
                     Ranked Retrieval
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          Recommendation          Evaluation
                 |                     |
                 +----------+----------+
                            |
                            v
                  Performance Analytics
```

---

## 4. Project Structure

```text
ir-search-engine-advanced/
|
|-- app.py
|-- ir_system.py
|-- requirements.txt
|-- README.md
|-- report.md
|-- experiment_summary.csv
|
|-- data/
|   |-- D0001.txt
|   |-- D0002.txt
|   |-- ...
|   `-- metadata.json
|
`-- assignment/reference files
```

### Main Files

**`app.py`**

Contains the Streamlit user interface and integrates the different IR modules.

**`ir_system.py`**

Contains the core IR functionality, including:

- Crawler
- Document representation
- Text preprocessing/mining
- Inverted index
- TF-IDF indexing
- Search
- PageRank
- HITS
- Recommendation
- Evaluation metrics

**`report.md`**

Contains implementation details, experimental analysis, observations, inference, and discussion.

**`experiment_summary.csv`**

Contains results from the controlled validation experiment used during implementation verification.

> The controlled validation experiment is not a substitute for the final experimental results obtained from the BITS Virtual Lab corpus.

**`data/`**

Contains the final crawled/document collection and associated metadata when generated.

---

## 5. Installation

### Prerequisites

Recommended:

- Python 3.10 or later
- `pip`

Clone the repository:

```bash
git clone https://github.com/2024dc04096/ir-search-engine-advanced.git
cd ir-search-engine-advanced
```

Create a virtual environment if required:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## 6. Dependencies

The project uses:

```text
streamlit
requests
beautifulsoup4
nltk
numpy
pandas
scikit-learn
scipy
plotly
networkx
lxml
```

These dependencies are listed in `requirements.txt`.

---

## 7. Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

If the `streamlit` executable is not directly available, use:

```bash
python -m streamlit run app.py
```

Streamlit will display the local application URL in the terminal.

---

## 8. Recommended Application Workflow

For a complete end-to-end demonstration, use the application in the following order.

### Step 1 — Crawling

Open **Crawling**.

1. Enter one or more seed URLs.
2. Configure the crawl depth.
3. Configure the maximum number of documents if required.
4. Start crawling.
5. Inspect the collected documents and crawl statistics.
6. Verify duplicate handling and extracted metadata.

### Step 2 — Preprocessing

Open **Preprocessing**.

1. Select the preprocessing strategy.
2. Inspect processed corpus statistics.
3. Review extracted keywords and document profiles.
4. Compare preprocessing/feature representations.
5. Inspect document clustering and topic-modelling results where applicable.

Preprocessing is reflected interactively in the displayed mining results; there is no separate **Run Preprocessing** button required.

### Step 3 — Index Management

Open **Index Management**.

Click:

**Build / Rebuild Index**

Inspect:

- Number of indexed documents
- Vocabulary/index statistics
- Inverted-index information
- Graph nodes and edges
- PageRank values
- HITS hub/authority values

A crawled corpus containing hyperlinks between collected documents should produce graph edges. If the graph has no internal links, PageRank/HITS cannot provide meaningful structural differentiation.

### Step 4 — Search

Open **Search**.

1. Enter a query.
2. Select the ranking method.
3. Select the number of results.
4. Execute the search.
5. Inspect ranked documents and their scores.

Compare the same query using:

```text
TF-IDF only
TF-IDF + PageRank
TF-IDF + HITS Authority
```

This demonstrates how textual relevance and link-analysis signals affect ranking.

### Step 5 — Recommendations

Open **Recommendations**.

Select a document or the relevant recommendation input and compare:

- Content-based recommendations
- Collaborative recommendations
- Hybrid recommendations

Inspect the **Top-K recommendation scores**.

If collaborative interaction data is uploaded, the application uses those interactions. Synthetic interactions should only be used as demonstration data when real user interaction data is unavailable.

### Step 6 — Evaluation

Open **Evaluation**.

1. Execute an evaluation query.
2. Identify/mark relevant documents.
3. Save the query evaluation.
4. Repeat the process for multiple queries.
5. Inspect individual-query metrics.
6. Inspect aggregate MAP and MRR.

Evaluation should be performed across several queries rather than relying on a single query.

### Step 7 — Performance Analytics

Open **Performance Analytics**.

Review the available performance measurements and visualizations and use them as experimental evidence for the final report.

---

## 9. Crawling and Duplicate Handling

Two forms of duplication are considered.

### Duplicate URLs

Normalized/visited URLs are tracked so that the same URL is not repeatedly crawled.

### Duplicate Documents

Document contents are hashed to identify identical content available through different URLs.

This prevents duplicate documents from unnecessarily affecting:

- Corpus size
- Index statistics
- Search ranking
- Recommendations
- Evaluation results

---

## 10. Search Ranking

### TF-IDF Ranking

Documents and queries are represented using TF-IDF features.

Cosine similarity measures textual similarity between the query and each indexed document.

### PageRank

The crawler retains outgoing links discovered in crawled documents.

A directed graph is created where:

```text
node = crawled document
edge = hyperlink from one crawled document to another
```

PageRank estimates structural importance based on this graph.

Normalized PageRank values can then be combined with textual similarity.

Conceptually:

```text
Combined Score =
(1 - alpha) * Text Similarity
+ alpha * Normalized PageRank
```

### HITS

HITS calculates:

- Hub scores
- Authority scores

The search module can combine normalized HITS authority scores with TF-IDF similarity to provide an alternative link-aware ranking experiment.

---

## 11. Recommendation Approaches

### Content-Based

Uses similarity between document representations to recommend documents similar to a selected document.

### Collaborative

Uses user-document interaction information.

When real interaction data is unavailable, synthetic interaction data may be used only for demonstration purposes.

### Hybrid

Combines content-based and collaborative recommendation signals.

Top-K recommendations are displayed together with recommendation/similarity scores.

---

## 12. Evaluation Methodology

For each query, the system compares retrieved documents against relevance judgments.

Single-query metrics include:

```text
Precision
Recall
F1
Precision@K
Recall@K
AP
RR
NDCG@K
```

Across multiple evaluated queries:

```text
MAP = Mean of Average Precision values

MRR = Mean of Reciprocal Rank values
```

The final assignment experiment should evaluate several representative queries and compare ranking approaches using tables and visualizations.

---

## 13. Validation Experiment

A controlled validation corpus was used during development to verify that the corrected IR pipeline behaves as expected.

The validation experiment exercises:

- Crawling
- Duplicate detection
- Preprocessing
- Inverted indexing
- Link-graph construction
- PageRank
- HITS
- Search
- Recommendation
- Evaluation metrics

Its summarized results are available in:

```text
experiment_summary.csv
```

These results are **implementation-validation results**.

They must not be represented as the final experimental results obtained in the BITS Virtual Lab.

The final report should use the actual corpus, screenshots, measurements, and relevance judgments generated during the final Virtual Lab execution.

---

## 14. Final BITS Virtual Lab Evidence

Before final submission, the project should be executed in the BITS Virtual Lab and evidence should be collected for:

- Dashboard
- Crawling interface
- Crawled corpus
- Duplicate handling
- Preprocessing/text mining
- Index Management
- Inverted index
- Link graph
- PageRank/HITS
- Search results
- Ranking comparison
- Recommendation results
- Evaluation results
- Performance analytics

The final report should contain screenshots and experimental measurements from this execution.

---

## 15. Important Limitations

1. PageRank and HITS depend on hyperlinks between documents that are actually present in the crawled corpus.
2. A corpus with few internal links may produce limited PageRank/HITS differentiation.
3. Collaborative recommendation quality depends on the availability and quality of user-document interaction data.
4. Synthetic collaborative interactions are intended only for algorithm demonstration.
5. KMeans in this implementation is an unsupervised clustering technique and should not be interpreted as a supervised classifier.
6. Retrieval evaluation depends on the quality of the relevance judgments.
7. Controlled validation results should not be confused with final Virtual Lab experimental results.

---

## 16. Submission Checklist

Before final submission, verify that the submission contains:

- [ ] `app.py`
- [ ] `ir_system.py`
- [ ] `requirements.txt`
- [ ] `README.md`
- [ ] Final dataset/document collection
- [ ] `metadata.json`
- [ ] Final report
- [ ] Actual experimental results
- [ ] Evaluation tables and visualizations
- [ ] Streamlit screenshots from the Virtual Lab
- [ ] Demo evidence/screen recording if required
- [ ] Compulsory inference and discussion answers
- [ ] Virtual Lab execution evidence

---

## 17. Repository

GitHub repository:

`https://github.com/2024dc04096/ir-search-engine-advanced`

---

## 18. Academic Note

The application is intended to demonstrate Information Retrieval concepts through an integrated end-to-end workflow.

Experimental results reported for the final assignment should be generated from the actual submitted corpus and final Virtual Lab execution so that the implementation, evidence, and conclusions remain consistent.