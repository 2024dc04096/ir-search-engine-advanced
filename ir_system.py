"""Core components for the end-to-end Information Retrieval assignment."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import networkx as nx
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    doc_id: str
    url: str
    title: str
    text: str
    depth: int = 0
    timestamp: str = ""
    content_length: int = 0
    content_hash: str = ""
    domain: str = ""
    out_links: list[str] = field(default_factory=list)


def document_frame(documents: list[Document]) -> pd.DataFrame:
    return pd.DataFrame([asdict(document) for document in documents])


class WebCrawler:
    """Breadth-first web crawler with URL, content and link-graph tracking."""

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout
        self.last_stats = {
            "requested": 0,
            "stored": 0,
            "duplicate_content": 0,
            "skipped_non_html": 0,
            "errors": 0,
            "links_discovered": 0,
        }

    def crawl(
        self,
        seeds: list[str],
        max_depth: int = 1,
        max_pages: int = 20,
    ) -> list[Document]:
        self.last_stats = {
            "requested": 0,
            "stored": 0,
            "duplicate_content": 0,
            "skipped_non_html": 0,
            "errors": 0,
            "links_discovered": 0,
        }

        queue: deque[tuple[str, int]] = deque()
        for seed in seeds:
            normalised = self._normalise_url(seed)
            if normalised:
                queue.append((normalised, 0))

        visited: set[str] = set()
        hashes: dict[str, str] = {}
        url_to_doc: dict[str, str] = {}
        documents: list[Document] = []
        session = requests.Session()
        session.headers.update({"User-Agent": "IR-Assignment-Crawler/1.0"})

        while queue and len(documents) < max_pages:
            url, depth = queue.popleft()
            if not url or url in visited or depth > max_depth:
                continue
            visited.add(url)
            self.last_stats["requested"] += 1

            try:
                response = session.get(url, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException:
                self.last_stats["errors"] += 1
                continue

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                self.last_stats["skipped_non_html"] += 1
                continue

            soup = BeautifulSoup(response.text, "lxml")
            for element in soup(["script", "style", "noscript"]):
                element.decompose()

            text = " ".join(soup.get_text(" ").split())
            source_domain = urlparse(url).netloc.lower()
            discovered_links: list[str] = []

            for link in soup.select("a[href]"):
                child = self._normalise_url(urljoin(url, link.get("href", "")))
                if not child:
                    continue
                child_domain = urlparse(child).netloc.lower()
                if child_domain != source_domain:
                    continue
                if child not in discovered_links:
                    discovered_links.append(child)
                    self.last_stats["links_discovered"] += 1
                    if depth < max_depth and child not in visited:
                        queue.append((child, depth + 1))

            if not text:
                continue

            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            existing_doc_id = hashes.get(content_hash)
            if existing_doc_id:
                # Keep a URL -> canonical document mapping so graph edges can
                # point to the canonical node even when another URL has the
                # exact same content.
                url_to_doc[url] = existing_doc_id
                self.last_stats["duplicate_content"] += 1
                continue

            doc_id = f"D{len(documents) + 1:04d}"
            document = Document(
                doc_id=doc_id,
                url=url,
                title=soup.title.get_text(" ", strip=True) if soup.title else url,
                text=text,
                depth=depth,
                timestamp=pd.Timestamp.now(tz="UTC").isoformat(),
                content_length=len(text),
                content_hash=content_hash,
                domain=source_domain,
                out_links=discovered_links,
            )
            documents.append(document)
            hashes[content_hash] = doc_id
            url_to_doc[url] = doc_id
            self.last_stats["stored"] += 1
            time.sleep(0.05)

        # Resolve every stored document's URL links to the canonical crawled
        # document IDs later in SearchIndex. The raw URLs are persisted here.
        self.last_stats["visited"] = len(visited)
        return documents

    @staticmethod
    def _normalise_url(url: str) -> str:
        url, _ = urldefrag(url.strip())
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ""

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return ""

        netloc = hostname
        if parsed.port:
            netloc = f"{hostname}:{parsed.port}"

        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")

        return urlunparse(
            (
                parsed.scheme.lower(),
                netloc,
                path,
                "",
                parsed.query,
                "",
            )
        )


class TextMiner:
    def __init__(self) -> None:
        self.stemmer = PorterStemmer()
        self.stopwords = set(ENGLISH_STOP_WORDS) | {
            "can",
            "could",
            "would",
            "should",
            "also",
            "using",
            "used",
            "use",
        }

    def tokens(self, text: str, strategy: str = "stem_stopwords") -> list[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{1,}", text.lower())
        if "stopwords" in strategy:
            words = [word for word in words if word not in self.stopwords]
        if "stem" in strategy:
            words = [self.stemmer.stem(word) for word in words]
        elif "lemma" in strategy:
            words = [self._lemma(word) for word in words]
        return words

    def preprocess(self, documents: list[Document], strategy: str = "stem_stopwords") -> list[str]:
        return [" ".join(self.tokens(document.text, strategy)) for document in documents]

    def compare_strategies(self, documents: list[Document]) -> pd.DataFrame:
        strategies = ["none", "stopwords", "stem_stopwords", "lemma_stopwords"]
        rows = []
        for strategy in strategies:
            texts = self.preprocess(documents, strategy)
            vocabulary = set(" ".join(texts).split())
            lengths = [len(text.split()) for text in texts]
            rows.append(
                {
                    "strategy": strategy,
                    "vocabulary_size": len(vocabulary),
                    "avg_document_length": round(float(np.mean(lengths)) if lengths else 0.0, 2),
                }
            )
        return pd.DataFrame(rows)

    def keywords(
        self,
        documents: list[Document],
        top_n: int = 10,
        strategy: str = "stem_stopwords",
    ) -> dict[str, list[tuple[str, float]]]:
        texts = self.preprocess(documents, strategy)
        if not any(texts):
            return {document.doc_id: [] for document in documents}

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        terms = np.asarray(vectorizer.get_feature_names_out())
        result: dict[str, list[tuple[str, float]]] = {}
        for row, document in enumerate(documents):
            scores = matrix[row].toarray().ravel()
            indices = scores.argsort()[-top_n:][::-1]
            result[document.doc_id] = [
                (terms[index], round(float(scores[index]), 4))
                for index in indices
                if scores[index] > 0
            ]
        return result

    @staticmethod
    def _lemma(word: str) -> str:
        """Lightweight rule-based normalization, not a linguistic lemmatizer."""
        replacements = (
            ("ies", "y"),
            ("ing", ""),
            ("ed", ""),
            ("es", ""),
            ("s", ""),
        )
        for suffix, replacement in replacements:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)] + replacement
        return word

    def cluster(self, matrix, clusters: int = 3) -> np.ndarray:
        if matrix.shape[0] == 0:
            return np.array([], dtype=int)
        if matrix.shape[1] == 0:
            return np.zeros(matrix.shape[0], dtype=int)
        count = max(1, min(clusters, matrix.shape[0]))
        if count == 1:
            return np.zeros(matrix.shape[0], dtype=int)
        return KMeans(n_clusters=count, random_state=42, n_init=10).fit_predict(matrix)

    def topics(
        self,
        texts: list[str],
        topics: int = 3,
        top_n: int = 8,
    ) -> list[list[tuple[str, float]]]:
        if not texts or not any(texts):
            return []

        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(texts)
        if matrix.shape[1] == 0:
            return []

        count = max(1, min(topics, matrix.shape[1], matrix.shape[0]))
        model = LatentDirichletAllocation(
            n_components=count,
            random_state=42,
            learning_method="batch",
        ).fit(matrix)
        terms = np.asarray(vectorizer.get_feature_names_out())

        output = []
        for row in range(count):
            indices = model.components_[row].argsort()[-top_n:][::-1]
            output.append([(terms[i], float(model.components_[row][i])) for i in indices])
        return output


class SearchIndex:
    def __init__(
        self,
        documents: list[Document],
        texts: list[str],
        strategy: str = "stem_stopwords",
    ) -> None:
        self.documents = documents
        self.strategy = strategy
        self.processed_texts = texts
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = sparse.csr_matrix((len(documents), 0), dtype=float)
        self.inverted_index: dict[str, list[tuple[str, int]]] = {}

        if texts and any(text.strip() for text in texts):
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            self.matrix = self.vectorizer.fit_transform(texts)

        self._build_inverted_index()
        self.graph = self._build_graph()
        self.pagerank = nx.pagerank(self.graph) if documents else {}

        try:
            if self.graph.number_of_edges():
                self.hubs, self.authorities = nx.hits(
                    self.graph,
                    max_iter=1000,
                    normalized=True,
                )
            else:
                raise nx.PowerIterationFailedConvergence(1000)
        except nx.PowerIterationFailedConvergence:
            uniform = 1 / len(documents) if documents else 0.0
            self.hubs = {document.doc_id: uniform for document in documents}
            self.authorities = dict(self.hubs)

        self.pagerank_normalized = self._normalize(self.pagerank)
        self.authorities_normalized = self._normalize(self.authorities)
        self.hubs_normalized = self._normalize(self.hubs)

    def _build_inverted_index(self) -> None:
        postings: dict[str, list[tuple[str, int]]] = {}
        for document, text in zip(self.documents, self.processed_texts):
            counts = Counter(text.split())
            for term, frequency in counts.items():
                postings.setdefault(term, []).append((document.doc_id, int(frequency)))
        self.inverted_index = dict(sorted(postings.items()))

    def _build_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(document.doc_id for document in self.documents)
        url_to_doc = {document.url: document.doc_id for document in self.documents}

        for document in self.documents:
            for link in document.out_links:
                child_id = url_to_doc.get(WebCrawler._normalise_url(link), url_to_doc.get(link))
                if child_id and child_id != document.doc_id:
                    graph.add_edge(document.doc_id, child_id)
        return graph

    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        maximum = max(values.values())
        if maximum <= 0:
            return {key: 0.0 for key in values}
        return {key: float(value / maximum) for key, value in values.items()}

    def search(
        self,
        query: str,
        top_k: int = 10,
        pagerank_weight: float = 0.2,
        ranking_mode: str = "tfidf_pagerank",
    ) -> pd.DataFrame:
        columns = [
            "rank",
            "doc_id",
            "title",
            "url",
            "tfidf_similarity",
            "pagerank",
            "authority",
            "combined_score",
        ]
        if not query.strip() or not self.documents or self.vectorizer is None:
            return pd.DataFrame(columns=columns)

        query_tokens = TextMiner().tokens(query, self.strategy)
        if not query_tokens:
            return pd.DataFrame(columns=columns)

        query_text = " ".join(query_tokens)
        similarities = cosine_similarity(
            self.vectorizer.transform([query_text]), self.matrix
        ).ravel()

        matched_indices = np.flatnonzero(similarities > 0)
        if matched_indices.size == 0:
            return pd.DataFrame(columns=columns)

        rows = []
        weight = min(max(float(pagerank_weight), 0.0), 1.0)
        for index in matched_indices:
            document = self.documents[int(index)]
            similarity = float(similarities[index])
            page_rank = float(self.pagerank_normalized.get(document.doc_id, 0.0))
            authority = float(self.authorities_normalized.get(document.doc_id, 0.0))

            if ranking_mode == "tfidf":
                score = similarity
            elif ranking_mode == "hits":
                score = (1 - weight) * similarity + weight * authority
            else:
                score = (1 - weight) * similarity + weight * page_rank

            rows.append(
                {
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "url": document.url,
                    "tfidf_similarity": similarity,
                    "pagerank": page_rank,
                    "authority": authority,
                    "combined_score": score,
                }
            )

        result = pd.DataFrame(rows).sort_values(
            ["combined_score", "tfidf_similarity"],
            ascending=False,
        ).head(top_k).reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result[columns]


class Recommender:
    """Content-based and item-co-occurrence collaborative recommendations."""

    def __init__(
        self,
        index: SearchIndex,
        interactions: dict[str, set[str]] | None = None,
    ) -> None:
        self.index = index
        ids = [document.doc_id for document in index.documents]
        if interactions is not None:
            self.interactions = interactions
            self.interaction_source = "uploaded"
        else:
            self.interactions = self._synthetic_interactions(ids)
            self.interaction_source = "synthetic_demo"

    @staticmethod
    def _synthetic_interactions(ids: list[str]) -> dict[str, set[str]]:
        if not ids:
            return {}
        n = len(ids)
        return {
            "demo_user_1": set(ids[: max(1, n // 2)]),
            "demo_user_2": set(ids[max(0, n // 3) :]),
            "demo_user_3": set(ids[::2]),
        }

    def content(self, doc_id: str, top_k: int = 5) -> pd.DataFrame:
        ids = [document.doc_id for document in self.index.documents]
        if doc_id not in ids or self.index.matrix.shape[1] == 0:
            return pd.DataFrame(columns=["doc_id", "similarity", "title"])

        scores = cosine_similarity(
            self.index.matrix[ids.index(doc_id)], self.index.matrix
        ).ravel()
        rows = [
            {"doc_id": item, "similarity": float(score)}
            for item, score in zip(ids, scores)
            if item != doc_id
        ]
        return self._details(
            sorted(rows, key=lambda row: row["similarity"], reverse=True)[:top_k]
        )

    def collaborative(self, doc_id: str, top_k: int = 5) -> pd.DataFrame:
        users = [items for items in self.interactions.values() if doc_id in items]
        counts: Counter[str] = Counter()
        for items in users:
            for item in items - {doc_id}:
                counts[item] += 1
        denominator = max(len(users), 1)
        rows = [
            {"doc_id": item, "similarity": count / denominator}
            for item, count in counts.items()
        ]
        return self._details(
            sorted(rows, key=lambda row: row["similarity"], reverse=True)[:top_k]
        )

    def hybrid(self, doc_id: str, alpha: float = 0.5, top_k: int = 5) -> pd.DataFrame:
        content = self.content(doc_id, len(self.index.documents)).set_index("doc_id")[
            "similarity"
        ].to_dict()
        collaborative = self.collaborative(
            doc_id, len(self.index.documents)
        ).set_index("doc_id")["similarity"].to_dict()

        rows = [
            {
                "doc_id": item,
                "similarity": float(
                    alpha * content.get(item, 0.0)
                    + (1 - alpha) * collaborative.get(item, 0.0)
                ),
            }
            for item in content
        ]
        return self._details(
            sorted(rows, key=lambda row: row["similarity"], reverse=True)[:top_k]
        )

    def _details(self, rows: list[dict]) -> pd.DataFrame:
        titles = {document.doc_id: document.title for document in self.index.documents}
        return pd.DataFrame(
            [
                {**row, "title": titles.get(row["doc_id"], row["doc_id"])}
                for row in rows
            ],
            columns=["doc_id", "similarity", "title"],
        )


def evaluate_query(
    retrieved: list[str],
    relevant: set[str],
    k: int | None = None,
) -> dict[str, float]:
    retrieved = list(dict.fromkeys(retrieved))
    ranked = retrieved[:k] if k else retrieved
    cutoff = k if k is not None else len(ranked)
    hits = [item for item in ranked if item in relevant]

    precision = len(hits) / len(ranked) if ranked else 0.0
    recall = len(hits) / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_at_k = len(hits) / cutoff if cutoff else 0.0
    r_at_k = recall

    average_precision = 0.0
    if relevant:
        precision_sum = 0.0
        hit_count = 0
        for rank, item in enumerate(ranked, start=1):
            if item in relevant:
                hit_count += 1
                precision_sum += hit_count / rank
        average_precision = precision_sum / len(relevant)

    first = next((rank for rank, item in enumerate(ranked, start=1) if item in relevant), 0)
    reciprocal_rank = 1 / first if first else 0.0

    dcg = sum(
        1 / np.log2(rank + 1)
        for rank, item in enumerate(ranked, start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), len(ranked))
    ideal_dcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "P@K": p_at_k,
        "R@K": r_at_k,
        "AP": average_precision,
        "RR": reciprocal_rank,
        "NDCG@K": ndcg,
        # These are intentionally aliased for a one-query view. True MAP/MRR
        # are produced by aggregate_evaluation over multiple queries.
        "MAP": average_precision,
        "MRR": reciprocal_rank,
    }


def evaluate(
    retrieved: list[str],
    relevant: set[str],
    k: int | None = None,
) -> dict[str, float]:
    return evaluate_query(retrieved, relevant, k)


def aggregate_evaluation(records: list[dict]) -> pd.DataFrame:
    """Macro-average query-level metrics; MAP/MRR become true multi-query means."""
    if not records:
        return pd.DataFrame()

    metric_rows = []
    for record in records:
        metrics = evaluate_query(
            record["retrieved"],
            set(record["relevant"]),
            int(record["k"]),
        )
        metric_rows.append(metrics)

    frame = pd.DataFrame(metric_rows)
    aggregate = frame[
        ["Precision", "Recall", "F1", "P@K", "R@K", "AP", "RR", "NDCG@K"]
    ].mean()
    return pd.DataFrame(
        {
            "metric": [
                "Precision",
                "Recall",
                "F1",
                "P@K",
                "R@K",
                "MAP",
                "MRR",
                "NDCG@K",
            ],
            "value": [
                aggregate["Precision"],
                aggregate["Recall"],
                aggregate["F1"],
                aggregate["P@K"],
                aggregate["R@K"],
                aggregate["AP"],
                aggregate["RR"],
                aggregate["NDCG@K"],
            ],
        }
    )


def evaluate_at_ks(retrieved: list[str], relevant: set[str], ks: list[int]) -> pd.DataFrame:
    rows = []
    for k in sorted(set(int(k) for k in ks if int(k) > 0)):
        rows.append({"K": k, **evaluate_query(retrieved, relevant, k)})
    return pd.DataFrame(rows)


def interactions_from_dataframe(
    frame: pd.DataFrame,
    valid_doc_ids: Iterable[str],
) -> dict[str, set[str]]:
    required = {"user_id", "doc_id"}
    if not required.issubset(frame.columns):
        raise ValueError("Interaction CSV must contain user_id and doc_id columns.")

    valid = set(valid_doc_ids)
    interactions: dict[str, set[str]] = {}
    for row in frame[["user_id", "doc_id"]].dropna().itertuples(index=False):
        user_id = str(row.user_id).strip()
        doc_id = str(row.doc_id).strip()
        if user_id and doc_id in valid:
            interactions.setdefault(user_id, set()).add(doc_id)
    return interactions


def save_corpus(documents: list[Document], directory: str = "data") -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    current_ids = {document.doc_id for document in documents}

    for text_path in path.glob("D*.txt"):
        if text_path.stem not in current_ids:
            text_path.unlink(missing_ok=True)

    for document in documents:
        (path / f"{document.doc_id}.txt").write_text(
            document.text,
            encoding="utf-8",
        )

    metadata = [
        {key: value for key, value in asdict(document).items() if key != "text"}
        for document in documents
    ]
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_corpus(directory: str = "data") -> list[Document]:
    metadata_path = Path(directory) / "metadata.json"
    if not metadata_path.exists():
        return []

    items = json.loads(metadata_path.read_text(encoding="utf-8"))
    documents = []
    for item in items:
        text_path = Path(directory) / f"{item['doc_id']}.txt"
        if not text_path.exists():
            continue
        documents.append(
            Document(
                doc_id=item["doc_id"],
                url=item.get("url", ""),
                title=item.get("title", item.get("url", item["doc_id"])),
                text=text_path.read_text(encoding="utf-8"),
                depth=int(item.get("depth", 0)),
                timestamp=item.get("timestamp", ""),
                content_length=int(item.get("content_length", 0)),
                content_hash=item.get("content_hash", ""),
                domain=item.get("domain", ""),
                out_links=item.get("out_links", []) or [],
            )
        )
    return documents
