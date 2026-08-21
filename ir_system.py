"""Core components for the end-to-end information retrieval assignment."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import networkx as nx
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
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


def document_frame(documents: list[Document]) -> pd.DataFrame:
    return pd.DataFrame([asdict(document) for document in documents])


class WebCrawler:
    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    def crawl(self, seeds: list[str], max_depth: int = 1, max_pages: int = 20) -> list[Document]:
        queue = [(self._normalise_url(url), 0) for url in seeds if url.strip()]
        visited: set[str] = set()
        hashes: set[str] = set()
        documents: list[Document] = []
        headers = {"User-Agent": "IR-Assignment-Crawler/1.0"}
        while queue and len(documents) < max_pages:
            url, depth = queue.pop(0)
            if not url or url in visited or depth > max_depth:
                continue
            visited.add(url)
            try:
                response = requests.get(url, timeout=self.timeout, headers=headers)
                response.raise_for_status()
                if "text/html" not in response.headers.get("content-type", "text/html"):
                    continue
            except requests.RequestException:
                continue
            soup = BeautifulSoup(response.text, "lxml")
            for element in soup(["script", "style", "noscript"]):
                element.decompose()
            text = " ".join(soup.get_text(" ").split())
            content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if not text or content_hash in hashes:
                continue
            hashes.add(content_hash)
            documents.append(Document(
                doc_id=f"D{len(documents) + 1:04d}", url=url,
                title=soup.title.get_text(" ", strip=True) if soup.title else url,
                text=text, depth=depth,
                timestamp=datetime.now(timezone.utc).isoformat(),
                content_length=len(text), content_hash=content_hash,
                domain=urlparse(url).netloc,
            ))
            if depth < max_depth:
                for link in soup.select("a[href]"):
                    child = self._normalise_url(urljoin(url, link["href"]))
                    if child and urlparse(child).netloc == urlparse(url).netloc and child not in visited:
                        queue.append((child, depth + 1))
            time.sleep(0.05)
        return documents

    @staticmethod
    def _normalise_url(url: str) -> str:
        url, _ = urldefrag(url.strip())
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        return url.rstrip("/")


class TextMiner:
    def __init__(self) -> None:
        self.stemmer = PorterStemmer()
        self.stopwords = set("a an and are as at be by for from has have in is it of on or that the to was were with this these those".split())

    def tokens(self, text: str, strategy: str = "stem_stopwords") -> list[str]:
        words = re.findall(r"[a-zA-Z]{2,}", text.lower())
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
            rows.append({"strategy": strategy, "vocabulary_size": len(vocabulary), "avg_document_length": round(float(np.mean(lengths)) if lengths else 0, 2)})
        return pd.DataFrame(rows)

    def keywords(self, documents: list[Document], top_n: int = 10, strategy: str = "stem_stopwords") -> dict[str, list[tuple[str, float]]]:
        texts = self.preprocess(documents, strategy)
        if not any(texts):
            return {document.doc_id: [] for document in documents}
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        terms = np.asarray(vectorizer.get_feature_names_out())
        result = {}
        for row, document in enumerate(documents):
            scores = matrix[row].toarray().ravel()
            indices = scores.argsort()[-top_n:][::-1]
            result[document.doc_id] = [(terms[index], round(float(scores[index]), 4)) for index in indices if scores[index] > 0]
        return result

    @staticmethod
    def _lemma(word: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    def cluster(self, matrix, clusters: int = 3) -> np.ndarray:
        count = max(1, min(clusters, matrix.shape[0]))
        return KMeans(n_clusters=count, random_state=42, n_init=10).fit_predict(matrix)

    def topics(self, matrix, topics: int = 3, top_n: int = 8) -> list[list[tuple[int, float]]]:
        count = max(1, min(topics, matrix.shape[1]))
        model = LatentDirichletAllocation(n_components=count, random_state=42, learning_method="batch").fit(matrix)
        return [list(zip(model.components_[row].argsort()[-top_n:][::-1], model.components_[row][model.components_[row].argsort()[-top_n:][::-1]])) for row in range(count)]


class SearchIndex:
    def __init__(self, documents: list[Document], texts: list[str]) -> None:
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(texts) if any(texts) else np.zeros((len(documents), 1))
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(document.doc_id for document in documents)
        self.pagerank = nx.pagerank(self.graph) if documents else {}
        if documents and self.graph.number_of_edges():
            self.hubs, self.authorities = nx.hits(self.graph, max_iter=1000, normalized=True)
        else:
            uniform = 1 / len(documents) if documents else 0.0
            self.hubs = {document.doc_id: uniform for document in documents}
            self.authorities = dict(self.hubs)

    def search(self, query: str, top_k: int = 10, pagerank_weight: float = 0.2) -> pd.DataFrame:
        columns = ["doc_id", "title", "url", "tfidf_similarity", "pagerank", "combined_score"]
        if not query.strip() or not self.documents:
            return pd.DataFrame(columns=columns)
        similarities = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        rows = []
        for index, document in enumerate(self.documents):
            page_rank = self.pagerank.get(document.doc_id, 0.0)
            similarity = float(similarities[index])
            rows.append({"doc_id": document.doc_id, "title": document.title, "url": document.url, "tfidf_similarity": similarity, "pagerank": page_rank, "combined_score": (1 - pagerank_weight) * similarity + pagerank_weight * page_rank})
        return pd.DataFrame(rows).sort_values("combined_score", ascending=False).head(top_k).reset_index(drop=True)


class Recommender:
    def __init__(self, index: SearchIndex) -> None:
        self.index = index
        ids = [document.doc_id for document in index.documents]
        self.interactions = {f"user_{number}": set(ids[number::3]) for number in range(3)} if ids else {}

    def content(self, doc_id: str, top_k: int = 5) -> pd.DataFrame:
        ids = [document.doc_id for document in self.index.documents]
        if doc_id not in ids:
            return pd.DataFrame(columns=["doc_id", "similarity", "title"])
        scores = cosine_similarity(self.index.matrix[ids.index(doc_id)], self.index.matrix).ravel()
        rows = [{"doc_id": item, "similarity": float(score)} for item, score in zip(ids, scores) if item != doc_id]
        return self._details(sorted(rows, key=lambda row: row["similarity"], reverse=True)[:top_k])

    def collaborative(self, doc_id: str, top_k: int = 5) -> pd.DataFrame:
        users = [items for items in self.interactions.values() if doc_id in items]
        counts = {}
        for items in users:
            for item in items - {doc_id}:
                counts[item] = counts.get(item, 0) + 1
        rows = [{"doc_id": item, "similarity": count / max(len(users), 1)} for item, count in counts.items()]
        return self._details(sorted(rows, key=lambda row: row["similarity"], reverse=True)[:top_k])

    def hybrid(self, doc_id: str, alpha: float = 0.5, top_k: int = 5) -> pd.DataFrame:
        content = self.content(doc_id, len(self.index.documents)).set_index("doc_id")["similarity"].to_dict()
        collaborative = self.collaborative(doc_id, len(self.index.documents)).set_index("doc_id")["similarity"].to_dict()
        rows = [{"doc_id": item, "similarity": alpha * content.get(item, 0) + (1 - alpha) * collaborative.get(item, 0)} for item in content]
        return self._details(sorted(rows, key=lambda row: row["similarity"], reverse=True)[:top_k])

    def _details(self, rows: list[dict]) -> pd.DataFrame:
        titles = {document.doc_id: document.title for document in self.index.documents}
        return pd.DataFrame(
            [{**row, "title": titles.get(row["doc_id"], row["doc_id"])} for row in rows],
            columns=["doc_id", "similarity", "title"],
        )


def evaluate(retrieved: list[str], relevant: set[str], k: int | None = None) -> dict[str, float]:
    ranked = retrieved[:k] if k else retrieved
    hits = [item for item in ranked if item in relevant]
    precision = len(hits) / len(ranked) if ranked else 0.0
    recall = len(hits) / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    average_precision = sum(sum(item in relevant for item in ranked[:i]) / i for i, item in enumerate(ranked, 1) if item in relevant) / len(relevant) if relevant else 0.0
    first = next((i for i, item in enumerate(ranked, 1) if item in relevant), 0)
    dcg = sum(1 / np.log2(i + 1) for i, item in enumerate(ranked, 1) if item in relevant)
    ideal = sum(1 / np.log2(i + 1) for i in range(1, min(len(relevant), len(ranked)) + 1))
    return {"Precision": precision, "Recall": recall, "F1": f1, "P@K": precision, "R@K": recall, "MAP": average_precision, "MRR": 1 / first if first else 0.0, "NDCG@K": dcg / ideal if ideal else 0.0}


def evaluate_at_ks(retrieved: list[str], relevant: set[str], ks: list[int]) -> pd.DataFrame:
    rows = []
    for k in ks:
        rows.append({"K": k, **evaluate(retrieved, relevant, k)})
    return pd.DataFrame(rows)


def save_corpus(documents: list[Document], directory: str = "data") -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    for document in documents:
        (path / f"{document.doc_id}.txt").write_text(document.text, encoding="utf-8")
    metadata = [{key: value for key, value in asdict(document).items() if key != "text"} for document in documents]
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_corpus(directory: str = "data") -> list[Document]:
    metadata_path = Path(directory) / "metadata.json"
    if not metadata_path.exists():
        return []
    items = json.loads(metadata_path.read_text(encoding="utf-8"))
    return [Document(**item, text=(Path(directory) / f"{item['doc_id']}.txt").read_text(encoding="utf-8")) for item in items]