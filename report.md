# IR Assignment 2 - Report
## End-to-End Information Retrieval System

## Important reproducibility note

This report has been synchronized with the **corrected implementation** and with a set of **controlled validation experiments executed against that implementation**. The validation corpus contains 12 small IR-themed documents with an explicit hyperlink graph so that crawling, indexing, PageRank, HITS, retrieval, recommendation, and evaluation can all be exercised deterministically.

The numerical results in Sections B-F below are therefore **validation results for the corrected code**, not results from the student's final BITS Virtual Lab crawl. They should be retained as implementation evidence only where appropriate; before final submission, the same experiments should be rerun in the BITS Virtual Lab on the actual submitted corpus and the corresponding screenshots/results should replace the validation figures.

The assignment requires the complete workflow to be executable through the Streamlit front end and requires experimental results, tables, visualizations, and inferences. The final submission must therefore include the actual BITS Lab execution evidence in addition to this synchronized report.

---

## A. System Overview

A Streamlit-based end-to-end Information Retrieval system was developed with the following modules:

- **Dashboard** for corpus, graph, and index statistics.
- **Web Crawler** with configurable crawl depth, multiple seed URLs, URL deduplication, exact-content deduplication, metadata storage, and preservation of outgoing links for graph-based ranking.
- **Text Preprocessing & Mining** with stopword removal, Porter stemming, rule-based normalization, TF-IDF keyword extraction, preprocessing-strategy comparison, document profiling, KMeans document clustering, and LDA topic modeling.
- **Index Management** with an explicit postings-list inverted index, TF-IDF representation, and a document hyperlink graph.
- **Search Engine** with TF-IDF cosine similarity, PageRank, HITS authority scores, and configurable combined ranking.
- **Recommender System** supporting content-based, collaborative, and hybrid recommendation with Top-K results and similarity scores. Collaborative mode can use an uploaded `user_id,doc_id` interaction CSV; deterministic synthetic demo interactions are used only when no interaction file is supplied.
- **Evaluation Dashboard** computing Precision, Recall, F1, P@K, R@K, AP/RR for individual queries, and true multi-query MAP/MRR through aggregation across saved relevance judgments.
- **Performance Analytics** for crawl, index, search, and recommendation timings.

**Technology stack:** Python, Streamlit, scikit-learn, NetworkX, BeautifulSoup, pandas, NumPy, SciPy, Plotly.

---

## B. Data Acquisition and Crawling

### B.1 Crawl configuration

The crawler supports:

- Multiple seed URLs.
- Configurable maximum crawl depth from 0 to 3 in the Streamlit interface.
- Maximum number of stored pages.
- Same-domain traversal.
- URL normalization that removes fragments and normalizes scheme/host/path formatting.
- URL deduplication using a `visited` set.
- Exact-content deduplication using SHA-256 content hashes.
- Separate storage of document text and metadata.
- Storage of discovered outgoing links for construction of the PageRank/HITS graph.

### B.2 Controlled crawler validation

A local six-page HTML test site was used to verify the crawler behavior. With one seed, maximum depth 2, and maximum 20 pages, the crawler produced the following result:

| Statistic | Validation result |
|---|---:|
| Requested URLs | 6 |
| Stored documents | 6 |
| Visited URLs | 6 |
| Links discovered | 12 |
| Exact duplicate content | 0 |
| Non-HTML pages skipped | 0 |
| Request errors | 0 |
|

A separate duplicate-content test used two seed URLs containing identical page content. The crawler requested 2 URLs, stored 1 document, and detected **1 exact duplicate**. This confirms that content-level deduplication is functioning independently of URL deduplication.

### B.3 Metadata/content separation

Each stored document has metadata fields such as document ID, URL, title, crawl depth, timestamp, content length, content hash, domain, and outgoing links. The document text is stored separately in `data/Dxxxx.txt`, while metadata is stored in `data/metadata.json`.

**Screenshots to add in the final BITS Lab version:**

1. Crawling page showing multiple seeds and crawl depth.
2. Crawl result statistics.
3. Saved corpus and metadata.

---

## C. Text Preprocessing and Mining

### C.1 Preprocessing strategies

The implementation compares four strategies:

1. `none`
2. `stopwords`
3. `stem_stopwords`
4. `lemma_stopwords`

The implementation uses a Porter stemmer for stemming. The `lemma_stopwords` option is a lightweight rule-based normalization method; it is **not** a full linguistic lemmatizer and is described as such in the code/UI.

### C.2 Validation experiment: preprocessing comparison

The 12-document controlled corpus produced the following statistics:

| Strategy | Vocabulary size | Average document length |
|---|---:|---:|
| None | 94 | 11.83 |
| Stopword removal | 91 | 11.50 |
| Stemming + stopword removal | 79 | 11.50 |
| Rule-based normalization + stopword removal | 82 | 11.50 |

The stemmed strategy produced the smallest vocabulary, reducing the feature space from 94 to 79 terms. The rule-based normalization strategy reduced the vocabulary to 82 terms.

### C.3 Keyword extraction

Top keywords are extracted using TF-IDF scores for each document. The Streamlit interface displays the selected document's top-N terms together with their TF-IDF weights.

### C.4 Document profiling

The profiling view reports document ID, title, domain, word count, and unique processed terms. This provides a basic corpus-characteristic view before retrieval experiments.

### C.5 Document clustering

KMeans is used for **unsupervised document clustering**. The UI and report use the term clustering rather than supervised classification because no labeled training set is assumed.

In the 12-document validation corpus, a 3-cluster run produced three coherent groups broadly corresponding to:

- retrieval/indexing/ranking concepts,
- web/link-analysis concepts,
- recommendation concepts.

Because KMeans labels are numeric and arbitrary, the exact cluster number is not itself a semantic category; the content of each cluster determines its interpretation.

### C.6 Topic modeling

LDA is applied to a CountVectorizer representation. On the validation corpus, the three-topic run produced representative term groups around retrieval/ranking, web crawling/link analysis, and recommendation/evaluation concepts.

**Screenshots to add in the final BITS Lab version:**

1. Preprocessing strategy comparison table/chart.
2. Keyword extraction for a selected document.
3. Document clustering output.
4. LDA topic table.
5. Document profiling table.

---

## D. Web Searching and Ranking

### D.1 Index construction

The corrected implementation uses two complementary structures:

- **Explicit inverted index:** `term -> [(doc_id, term_frequency), ...]`.
- **TF-IDF matrix:** used for cosine-similarity retrieval.

For the 12-document validation corpus:

| Index statistic | Result |
|---|---:|
| Documents | 12 |
| Indexed vocabulary terms | 79 |
| Hyperlink graph edges | 30 |
| Index build time | 0.00973 s |

The explicit inverted index makes the retrieval architecture easier to explain as a conventional Information Retrieval system rather than treating the TF-IDF matrix itself as the inverted index.

### D.2 Query preprocessing

Search queries are passed through the same preprocessing strategy used for the corpus before TF-IDF transformation. This avoids a tokenization/stemming mismatch between indexed documents and user queries.

### D.3 PageRank and HITS

Outgoing links captured during crawling are resolved to document IDs and used to create a directed document graph. PageRank and HITS are then computed on that actual graph.

For the validation corpus, the graph contained **12 nodes and 30 directed edges**, so PageRank/HITS were based on a non-empty link structure rather than isolated document nodes.

The highest PageRank nodes in the validation run were D0005, D0008, and D0003 after normalization. The highest HITS authority score was also assigned to D0005.

### D.4 Ranking modes

The search interface supports:

- **TF-IDF only**
- **TF-IDF + PageRank**
- **TF-IDF + HITS authority**

For PageRank/HITS modes, the link-ranking signal is normalized and combined with cosine similarity using a configurable weight:

`combined_score = (1 - w) * TF-IDF_similarity + w * link_score`

This normalization is important because raw PageRank values are much smaller than cosine-similarity values.

### D.5 Validation ranking experiment

Five queries were evaluated against the 12-document validation corpus. Relevance sets were manually defined before evaluating the ranking methods. Evaluation was performed at **K = 5**.

| Ranking method | Precision | Recall | F1 | P@5 | R@5 | MAP | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF only | 0.6200 | 0.7667 | 0.6700 | 0.4000 | 0.7667 | 0.7278 | 1.0000 | 0.8265 |
| TF-IDF + PageRank | 0.6200 | 0.7667 | 0.6700 | 0.4000 | 0.7667 | **0.7400** | 1.0000 | **0.8346** |
| TF-IDF + HITS authority | 0.6200 | 0.7667 | 0.6700 | 0.4000 | 0.7667 | 0.7178 | 1.0000 | 0.8224 |

### D.6 Interpretation of ranking experiment

On this controlled corpus, adding PageRank produced a small improvement in MAP from **0.7278 to 0.7400** and NDCG@5 from **0.8265 to 0.8346**. HITS produced slightly lower MAP and NDCG than the TF-IDF baseline for these particular relevance judgments.

The improvement is deliberately modest: TF-IDF already captures strong lexical relevance, while link-based ranking provides an additional authority signal. The experiment therefore supports the conclusion that PageRank can improve the order of results when the link structure contains useful information, but it should not dominate textual relevance.

Average measured search time on this small validation corpus was approximately:

| Ranking method | Mean query time |
|---|---:|
| TF-IDF only | 0.00186 s |
| TF-IDF + PageRank | 0.00157 s |
| TF-IDF + HITS | 0.00148 s |

These timings are for a 12-document in-memory corpus and should **not** be interpreted as scalable production performance measurements. Their main purpose is to verify that ranking remains fast after index construction.

**Screenshots to add in the final BITS Lab version:**

1. Search results for the same query under TF-IDF, PageRank, and HITS.
2. Ranking-score visualization.
3. Index Management page showing inverted-index terms and graph edges.
4. PageRank/HITS score table.

---

## E. Recommender System

### E.1 Content-based recommendation

Content-based recommendation uses cosine similarity between TF-IDF document vectors. It is therefore based only on document content and works without user history.

### E.2 Collaborative recommendation

Collaborative recommendation uses user-item interactions. The application accepts an interaction CSV with the columns:

`user_id, doc_id`

When no interaction file is supplied, deterministic synthetic demo interactions are used and explicitly labeled as synthetic. They are included only to demonstrate the collaborative algorithm; they should not be described as real user behavior in the final report.

### E.3 Hybrid recommendation

The hybrid model combines content and collaborative scores using an adjustable content weight (`alpha`). This allows the user to control how strongly content similarity influences the final recommendation.

### E.4 Validation examples

For document D0009 (Content Based Recommendation), the validation run produced the following content-based recommendations:

| Rank | Document | Similarity |
|---|---|---:|
| 1 | D0003 - TF IDF Ranking | 0.2515 |
| 2 | D0011 - Hybrid Recommendation | 0.2311 |
| 3 | D0010 - Collaborative Filtering | 0.2051 |
| 4 | D0008 - Learning to Rank | 0.0388 |
| 5 | D0001 - Information Retrieval Overview | 0.0272 |

For the same document, the hybrid recommender ranked D0011 first with a combined score of **0.6156**, followed by D0005 and D0007 at **0.5000** each under the synthetic demonstration interactions.

These numbers demonstrate that the recommendation module produces Top-K results with scores and that the hybrid approach can change the ordering when collaborative evidence is introduced.

**Screenshots to add in the final BITS Lab version:**

1. Content-based Top-K recommendations.
2. Collaborative Top-K recommendations with uploaded interactions, if available.
3. Hybrid recommendations and alpha setting.

---

## F. Evaluation Metrics

The system supports the following metrics:

| Metric | Interpretation |
|---|---|
| Precision | Fraction of retrieved documents that are relevant |
| Recall | Fraction of relevant documents that are retrieved |
| F1 | Harmonic mean of Precision and Recall |
| P@K | Precision within the first K results |
| R@K | Recall within the first K results |
| AP | Average Precision for one query |
| RR | Reciprocal Rank for one query |
| MAP | Mean AP across saved queries |
| MRR | Mean RR across saved queries |
| NDCG@K | Ranking quality with logarithmic discounting |

The corrected implementation distinguishes **single-query AP/RR** from **multi-query MAP/MRR**. The aggregate values in Section D are macro-averages across five saved validation queries.

### F.1 Query-level validation examples

For the query **"pagerank ranking"** at K=5:

| Method | Precision | Recall | F1 | AP | RR | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.6000 | 1.0000 | 0.7500 | 0.8056 | 1.0000 | 0.9060 |
| TF-IDF + PageRank | 0.6000 | 1.0000 | 0.7500 | **0.8667** | 1.0000 | **0.9469** |
| TF-IDF + HITS | 0.6000 | 1.0000 | 0.7500 | 0.7556 | 1.0000 | 0.8855 |

This query illustrates the purpose of link-based ranking particularly well: PageRank changed the ordering of relevant documents enough to improve AP and NDCG while maintaining the same Precision and Recall at K.

### F.2 K-wise evaluation

The evaluation page also compares P@K, R@K, and NDCG@K across different K values for a selected query. This provides a direct visual analysis of how ranking quality changes as more results are inspected.

**Screenshots to add in the final BITS Lab version:**

1. Current-query metric table.
2. P@K / R@K / NDCG@K chart.
3. Multi-query MAP/MRR table.
4. Comparative ranking-method chart.

---

## G. Inference and Discussion

### 1. Suppose the system retrieves highly relevant documents but ranks them poorly. Identify the possible causes and propose improvements.

The validation experiments show why multiple ranking signals are useful. Possible causes include:

- TF-IDF depends heavily on lexical overlap and can under-score semantically similar documents that use different terms.
- A document may be relevant but have weak link authority, so PageRank can under-rank it if the link graph is sparse or biased.
- A high link-ranking weight can promote popular or highly linked documents even when their textual relevance is lower.
- Query and document vocabularies can still differ even after basic stemming and stopword removal.

Improvements include query expansion, semantic representations, learned ranking, weight tuning using validation metrics, and user-feedback-based reranking.

In the validation run, a moderate PageRank weight improved MAP and NDCG slightly, while an alternative HITS combination did not improve the baseline. This supports tuning the weight and evaluating several ranking signals rather than assuming that any additional signal will improve retrieval.

---

### 2. If duplicate or near-duplicate documents exist in the corpus, how would they affect indexing, ranking, recommendation, and evaluation?

Duplicates can:

- inflate index size and alter term/document-frequency statistics;
- occupy multiple Top-K positions and reduce result diversity;
- produce overly similar recommendations;
- distort evaluation by making repeated versions of the same information appear as separate relevant hits.

The implementation mitigates exact duplicates using SHA-256 content hashing. The crawler validation test confirmed that two URLs with identical page content produced **one stored document and one duplicate-content event**.

For near-duplicates, additional techniques such as shingling with MinHash/LSH, canonical URL handling, or post-retrieval result collapse would be needed.

---

### 3. Compare content-based and collaborative recommendation. Under what scenarios would each be preferable?

| Aspect | Content-based | Collaborative |
|---|---|---|
| Basis | Item/document features | User-item interaction patterns |
| New item | Works immediately if content exists | Requires interactions |
| New user | Needs at least one known preference | Needs user history and population interactions |
| Data requirement | Document content | Interaction matrix |
| Diversity | Can over-specialize around similar content | Can introduce less-obvious items |
| Explainability | Strong; shared terms/features can be inspected | More indirect |

Content-based recommendation is preferable for small systems with rich document text and sparse user activity. Collaborative filtering becomes more useful when many users generate sufficient interactions. A hybrid model is useful when both content and interaction signals are available.

The validation recommender also demonstrates an important limitation: synthetic interactions are suitable for testing the algorithm, but real collaborative claims require real interaction data.

---

### 4. Discuss how crawling, text mining, indexing, search, ranking, and recommendation contribute to an end-to-end Information Retrieval system.

The components form a dependency chain:

1. **Crawling** acquires the raw information and the document-link structure.
2. **Preprocessing and text mining** convert raw text into normalized features and corpus statistics.
3. **Indexing** makes those features efficiently searchable.
4. **Search** retrieves candidates using query-document similarity.
5. **Ranking** combines textual relevance with authority signals such as PageRank/HITS.
6. **Recommendation** extends discovery beyond an explicit query.
7. **Evaluation** measures the resulting quality and provides feedback for tuning earlier stages.

The validation experiments also showed that crawling and graph construction directly affect ranking: if outgoing links are not retained, PageRank/HITS cannot represent actual document relationships. Similarly, inconsistent preprocessing between documents and queries can reduce lexical matching.

---

### 5. Based on the results obtained, provide the learnings clearly.

The main learnings from the corrected implementation validation are:

- **Preprocessing materially changes feature space.** Stemming reduced the vocabulary from 94 to 79 terms on the validation corpus, while the rule-based normalization option reduced it to 82.
- **A real link graph is necessary for PageRank/HITS.** The corrected crawler preserved outgoing links, producing 30 graph edges in the 12-document validation corpus.
- **PageRank provided a small but measurable retrieval improvement.** MAP increased from 0.7278 for TF-IDF alone to 0.7400 for TF-IDF + PageRank, while NDCG@5 increased from 0.8265 to 0.8346.
- **HITS is not automatically better than PageRank.** The HITS combination produced MAP 0.7178 and NDCG@5 0.8224 in this validation experiment.
- **TF-IDF remains a strong baseline for small keyword-driven collections.** It achieved Precision 0.6200 and Recall 0.7667 across the five validation queries.
- **Recommendation quality depends strongly on data availability.** Content-based recommendation works directly from document vectors, while collaborative recommendation requires meaningful user-item interactions.
- **Evaluation depends on relevance judgments.** MAP/MRR become meaningful only when multiple queries have consistent relevance labels.
- **Performance should be interpreted in context.** Search operations were around 1.5-1.9 ms on the 12-document in-memory validation corpus, but these numbers are not a substitute for large-corpus scalability testing.

---

## H. Final Submission Preparation

Before the final BITS Virtual Lab submission, the following items must be replaced or completed with actual execution evidence:

- [ ] Run the corrected application in the BITS Virtual Lab.
- [ ] Crawl the final chosen external/public dataset using the actual seed URLs.
- [ ] Save the final `data/` corpus and metadata.
- [ ] Re-run preprocessing comparison on the final corpus.
- [ ] Rebuild the actual inverted index and link graph.
- [ ] Evaluate at least 4-5 representative queries with manually checked relevance judgments.
- [ ] Save at least two or more queries so MAP/MRR are true multi-query measurements.
- [ ] Compare TF-IDF, TF-IDF + PageRank, and TF-IDF + HITS on the final corpus.
- [ ] Capture Streamlit screenshots for Crawling, Preprocessing, Index Management, Search, Recommendations, Evaluation, and Performance Analytics.
- [ ] Replace the validation numbers in this report with the final BITS Lab numbers if the final corpus differs from the controlled validation corpus.
- [ ] Add the required screenshots/screen recording and final experimental charts.
- [ ] Verify that README commands and application workflow match exactly.

---

## I. Submission Checklist

- [ ] Streamlit application code (`app.py` + supporting module(s))
- [ ] Dataset (`data/` folder with final crawled documents and metadata)
- [ ] Synchronized report with final screenshots and actual BITS Lab results
- [ ] Screenshots or short screen recording of the application running
- [ ] README with dependency installation and `streamlit run app.py`
- [ ] Evidence of execution on the BITS Virtual Lab
