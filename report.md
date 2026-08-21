

# IR Assignment 2 - Report
## End-to-End Information Retrieval System


---


## A. System Overview


A Streamlit-based end-to-end Information Retrieval system was developed with the following modules:
- **Web Crawler** with configurable depth, multiple seed URLs, URL/content deduplication, and separate metadata storage
- **Text Preprocessing & Mining** with stemming, lemmatization, keyword extraction, document clustering (KMeans), topic modeling (LDA), and strategy comparison
- **Search Engine** with TF-IDF inverted index, PageRank, HITS, and combined ranking
- **Recommender System** supporting content-based, collaborative, and hybrid recommendations with Top-K display
- **Evaluation Dashboard** computing Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG


**Tech Stack:** Python, Streamlit, scikit-learn, NetworkX, BeautifulSoup, Plotly


---


## B. Data Acquisition (Crawling)


- Seed URLs: Wikipedia articles on Information Retrieval, Search Engines, Web Crawlers
- Configurable crawl depth (0-3) and max pages
- **Duplicate handling:** URL deduplication via visited set; content deduplication via MD5 hashing
- **Metadata** (URL, domain, depth, timestamp, content length) stored separately from document content


*(Insert screenshots of Crawling page here)*


---


## C. Text Preprocessing & Mining


- **Preprocessing options:** Stemming (Porter), Lemmatization (rule-based), Stopword removal
- **Keyword extraction:** TF-IDF based, top-N keywords visualized
- **Document classification:** KMeans clustering with configurable cluster count
- **Topic modeling:** LDA with configurable topic count
- **Strategy comparison:** Vocab size and avg document length compared across 4 preprocessing strategies


*(Insert screenshots of Preprocessing tabs here)*


---


## D. Web Searching


- **Inverted index** built from TF-IDF vectorizer
- **Ranking:** Combined score = (1 - w) × TF-IDF cosine similarity + w × PageRank
- **PageRank** computed on the document link graph using NetworkX
- **HITS** (Hub/Authority scores) also computed and displayed
- Adjustable PageRank weight and Top-K via UI


*(Insert screenshots of Search page and ranking visualization here)*


---


## E. Recommender System


- **Content-based:** Cosine similarity on TF-IDF vectors between documents
- **Collaborative:** User-based filtering using interaction overlap; synthetic users added for demonstration
- **Hybrid:** Weighted combination of content-based and collaborative scores (alpha parameter)
- Top-K recommendations displayed with similarity scores


*(Insert screenshots of Recommendations tabs here)*


---


## F. Evaluation Metrics


| Metric | Description |
|--------|-------------|
| Precision | Fraction of retrieved docs that are relevant |
| Recall | Fraction of relevant docs that are retrieved |
| F1-Score | Harmonic mean of Precision and Recall |
| P@K | Precision at rank K |
| R@K | Recall at rank K |
| MAP | Mean Average Precision across queries |
| MRR | Mean Reciprocal Rank across queries |
| NDCG@K | Normalized Discounted Cumulative Gain at rank K |


*(Insert evaluation results table and charts here)*


---


## G. Inference and Discussion


### 1. Highly relevant documents ranked poorly — causes and improvements


**Possible causes:**
- **TF-IDF limitations:** TF-IDF relies on term overlap. A relevant document using synonyms or paraphrased language will score low despite being highly relevant.
- **Sparse link graph:** If relevant documents have few inbound links, PageRank assigns them low scores, dragging down the combined ranking.
- **Query-document vocabulary mismatch:** The user's query terms may not match the document's vocabulary even though the topic is the same.
- **Equal PageRank weighting:** A high PageRank weight can boost popular but less relevant pages above niche but highly relevant ones.


**Proposed improvements:**
- Use **query expansion** (adding synonyms or related terms) to bridge vocabulary gaps.
- Apply **semantic embeddings** (e.g., Word2Vec, BERT) instead of or alongside TF-IDF for similarity.
- Use **learning-to-rank** models that combine multiple signals (content, links, user behavior) with learned weights.
- Tune the **PageRank weight parameter** based on evaluation metrics to find the optimal balance.
- Incorporate **user feedback** (click-through data) to re-rank results over time.


---


### 2. Impact of duplicate/near-duplicate documents


**Effects on each component:**
- **Indexing:** Duplicates inflate term frequencies and document frequencies, distorting TF-IDF weights. Index size increases unnecessarily.
- **Ranking:** Duplicate documents consume multiple top-K slots, reducing result diversity. PageRank can be diluted or artificially concentrated if duplicates link to each other.
- **Recommendation:** Content-based similarity between duplicates will be ~1.0, causing the recommender to suggest copies instead of diverse content.
- **Evaluation:** Precision appears inflated (multiple duplicates counted as relevant), while recall may drop if duplicates crowd out genuinely different relevant documents.


**Mitigation methods:**
- **Content hashing** (MD5/SHA) to detect exact duplicates at crawl time (implemented in our system).
- **Shingling + MinHash/LSH** for near-duplicate detection based on overlapping n-gram sets.
- **Canonical URL resolution** to merge documents from different URLs pointing to the same content.
- **Post-retrieval deduplication** to collapse near-duplicate results before displaying to the user.


---


### 3. Content-based vs. Collaborative recommendation


| Aspect | Content-Based | Collaborative |
|--------|--------------|---------------|
| **Mechanism** | Recommends items similar to what a user liked, based on item features (TF-IDF vectors) | Recommends items liked by similar users, based on interaction patterns |
| **Cold start (new item)** | Can recommend new items immediately if content is available | Cannot recommend new items until users interact with them |
| **Cold start (new user)** | Needs at least one liked item to start | Needs sufficient interaction history from the user |
| **Diversity** | Tends to recommend very similar items (filter bubble) | Can surface unexpected items that dissimilar content wouldn't suggest |
| **Data requirement** | Only needs item content | Needs a user-item interaction matrix |
| **Scalability** | Scales with corpus size | Scales with number of users × items |


**When to prefer content-based:**
- New systems with few users, sparse interaction data
- Domains where item content is rich and descriptive (e.g., academic papers, news articles)
- When explainability is important (can point to shared features)


**When to prefer collaborative:**
- Mature systems with many active users and dense interaction data
- When content features are poor or unavailable (e.g., music, movies)
- When discovering diverse or serendipitous recommendations is valued


**Hybrid** approaches combine both to mitigate individual weaknesses.


---


### 4. Integration of IR components — contribution to overall effectiveness


The end-to-end pipeline creates a synergistic system where each component enhances the others:


- **Crawling** provides the raw material. Configurable depth and seed diversity ensure broad coverage. Deduplication at this stage prevents downstream issues.
- **Text mining** transforms raw HTML into clean, structured representations. Preprocessing quality directly affects index quality — poor tokenization or missed stopwords degrade search precision.
- **Indexing** (inverted index + TF-IDF) enables efficient retrieval. Without indexing, every query would require a linear scan of all documents.
- **Ranking** (PageRank/HITS + TF-IDF) ensures that among matching documents, the most authoritative and relevant appear first. PageRank captures the web's link structure as a quality signal independent of query terms.
- **Recommendation** extends retrieval beyond explicit queries. Users discover relevant documents they wouldn't have searched for, increasing engagement and information coverage.
- **Evaluation** closes the feedback loop. Metrics like MAP and NDCG quantify system quality and guide improvements to each upstream component.


The key insight is that **errors compound across the pipeline** — a poor crawler produces noisy text, which creates a bad index, which returns poor search results. Conversely, improvements at any stage propagate downstream.


---


### 5. Learnings from experimental results


- **Preprocessing matters significantly:** The strategy comparison showed that stemming reduces vocabulary size more aggressively than lemmatization, but can merge semantically distinct words. Lemmatization preserves more meaning while still reducing dimensionality.
- **PageRank adds value only with sufficient link structure:** On small crawled corpora with few inter-document links, PageRank scores are nearly uniform. The benefit becomes apparent as corpus size and link density grow.
- **TF-IDF is a strong baseline:** Despite its simplicity, TF-IDF cosine similarity produces reasonable rankings for keyword queries. The gap between TF-IDF alone and TF-IDF+PageRank is modest on small collections.
- **Collaborative filtering requires sufficient data:** With few simulated users, collaborative recommendations are sparse. Content-based recommendations are more reliable for small-scale systems.
- **Evaluation requires careful ground truth:** IR metrics are only as meaningful as the relevance judgments. Manual annotation is time-consuming but essential for meaningful evaluation.
- **End-to-end integration reveals bottlenecks:** Building the full pipeline exposed that crawling speed and preprocessing quality are the primary bottlenecks — search and recommendation are fast once the index is built.


---


## Submission Checklist


- [ ] Streamlit application code (`app.py` + modules)
- [ ] Dataset (`data/` folder with crawled documents)
- [ ] This report with screenshots and results
- [ ] Screenshots / screen recording of the application
- [ ] README with setup instructions
- [ ] Executed on BITS Virtual Lab


