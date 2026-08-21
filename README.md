# IR Assignment 2 - End-to-End Information Retrieval System


## Setup

Create and activate the project virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```


```bash
pip install -r requirements.txt
```

The application stores crawled document text in `data/*.txt` and crawl metadata separately in `data/metadata.json`.


## Run


```bash
streamlit run app.py
```


## Features


1. **Dashboard** - Corpus overview and statistics
2. **Crawling** - Configurable web crawler with depth control, deduplication
3. **Preprocessing** - Stemming, lemmatization, stopword removal, keyword extraction, clustering, topic modeling
4. **Index Management** - Inverted index, PageRank, HITS scores
5. **Search** - TF-IDF + PageRank ranked retrieval with visualizations
6. **Recommendations** - Content-based, collaborative, and hybrid
7. **Evaluation** - Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG
8. **Performance Analytics** - System-wide metrics and analysis


## Workflow


1. Go to **Crawling** → enter seed URLs → click "Start Crawling"
2. Go to **Preprocessing** → configure preprocessing → click "Run Preprocessing"
3. Go to **Index Management** → click "Build / Rebuild Index"
4. Go to **Search** → enter queries and view ranked results
5. Go to **Recommendations** → explore content-based and collaborative recommendations
6. Go to **Evaluation** → mark relevant documents and compute IR metrics
7. Go to **Dashboard** / **Performance Analytics** → view overall statistics
