import hashlib
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ir_system import (
    Document,
    Recommender,
    SearchIndex,
    TextMiner,
    WebCrawler,
    aggregate_evaluation,
    document_frame,
    evaluate,
    evaluate_at_ks,
    interactions_from_dataframe,
    load_corpus,
    save_corpus,
)


st.set_page_config(page_title="IR Assignment 2", page_icon="IR", layout="wide")
st.title("End-to-End Information Retrieval")
st.caption("Crawl, mine, index, rank, recommend, and evaluate a document collection.")


# -----------------------------
# Session state
# -----------------------------
if "documents" not in st.session_state:
    st.session_state.documents = load_corpus()
if "strategy" not in st.session_state:
    st.session_state.strategy = "stem_stopwords"
if "index" not in st.session_state:
    st.session_state.index = None
if "index_signature" not in st.session_state:
    st.session_state.index_signature = None
if "judgments" not in st.session_state:
    st.session_state.judgments = {}
for key, default in {
    "last_crawl_seconds": 0.0,
    "last_index_seconds": 0.0,
    "last_search_seconds": 0.0,
    "last_recommendation_seconds": 0.0,
}.items():
    st.session_state.setdefault(key, default)


def corpus_signature(documents: list[Document], strategy: str) -> str:
    payload = [
        strategy,
        *(
            f"{d.doc_id}|{d.content_hash}|{','.join(sorted(d.out_links))}"
            for d in documents
        ),
    ]
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def current_signature() -> str:
    return corpus_signature(st.session_state.documents, st.session_state.strategy)


def index_is_ready() -> bool:
    return (
        st.session_state.index is not None
        and st.session_state.index_signature == current_signature()
    )


def rebuild_index() -> None:
    documents = st.session_state.documents
    miner = TextMiner()
    started = time.perf_counter()
    texts = miner.preprocess(documents, st.session_state.strategy)
    st.session_state.index = SearchIndex(
        documents,
        texts,
        strategy=st.session_state.strategy,
    ) if documents else None
    st.session_state.index_signature = current_signature() if documents else None
    st.session_state.last_index_seconds = time.perf_counter() - started


def invalidate_index() -> None:
    st.session_state.index = None
    st.session_state.index_signature = None


# -----------------------------
# Shared objects
# -----------------------------
documents: list[Document] = st.session_state.documents
miner = TextMiner()
ready = index_is_ready()

st.sidebar.caption("Assignment 2 - Information Retrieval")
st.sidebar.metric("Documents", len(documents))
st.sidebar.metric("Index terms", len(st.session_state.index.inverted_index) if ready else 0)
st.sidebar.write("Index status:")
if ready:
    st.sidebar.success("Ready")
elif documents:
    st.sidebar.warning("Needs Build / Rebuild")
else:
    st.sidebar.info("No corpus")

page = st.sidebar.radio(
    "Module",
    [
        "Dashboard",
        "Crawling",
        "Preprocessing",
        "Index Management",
        "Search",
        "Recommendations",
        "Evaluation",
        "Performance Analytics",
    ],
)


# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    left, middle, right = st.columns(3)
    left.metric("Corpus documents", len(documents))
    middle.metric("Total words", sum(len(document.text.split()) for document in documents))
    right.metric(
        "Domains",
        len({document.domain for document in documents if document.domain}),
    )

    if documents:
        summary = document_frame(documents).copy()
        show_columns = [
            "doc_id",
            "title",
            "url",
            "depth",
            "content_length",
            "domain",
        ]
        st.dataframe(
            summary[show_columns],
            use_container_width=True,
            hide_index=True,
        )

        if ready:
            a, b, c = st.columns(3)
            a.metric("Graph nodes", ready and st.session_state.index.graph.number_of_nodes())
            b.metric("Graph edges", ready and st.session_state.index.graph.number_of_edges())
            c.metric("Indexed terms", len(st.session_state.index.inverted_index))
    else:
        st.info("Start with Crawling, or add sample text files.")


# -----------------------------
# Crawling
# -----------------------------
elif page == "Crawling":
    st.subheader("Web Crawling")
    seeds_text = st.text_area(
        "Seed URLs (one per line)",
        "https://en.wikipedia.org/wiki/Information_retrieval\nhttps://en.wikipedia.org/wiki/Search_engine",
        height=120,
    )
    depth = st.slider("Maximum crawl depth", 0, 3, 1)
    max_pages = st.number_input("Maximum stored pages", 1, 100, 10)

    st.caption(
        "The crawler follows same-domain links from each seed, removes URL fragments, "
        "deduplicates exact content, and stores discovered links for PageRank/HITS."
    )

    if st.button("Start Crawling", type="primary"):
        seeds = [line.strip() for line in seeds_text.splitlines() if line.strip()]
        crawler = WebCrawler()
        started = time.perf_counter()
        with st.spinner("Fetching pages..."):
            new_documents = crawler.crawl(seeds, int(depth), int(max_pages))
        st.session_state.last_crawl_seconds = time.perf_counter() - started
        st.session_state.documents = new_documents
        st.session_state.strategy = st.session_state.strategy
        save_corpus(new_documents)
        st.session_state.judgments = {}
        invalidate_index()
        st.success(f"Stored {len(new_documents)} unique documents.")
        st.json(crawler.last_stats)
        st.rerun()

    uploaded = st.file_uploader(
        "Optional local text files",
        accept_multiple_files=True,
        type=["txt"],
    )
    if uploaded and st.button("Load Local Files"):
        local_documents = []
        for number, file in enumerate(uploaded, start=1):
            text = file.read().decode("utf-8", errors="replace")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            local_documents.append(
                Document(
                    doc_id=f"D{number:04d}",
                    url=f"local://{Path(file.name).name}",
                    title=file.name,
                    text=text,
                    depth=0,
                    timestamp="local",
                    content_length=len(text),
                    content_hash=content_hash,
                    domain="local",
                    out_links=[],
                )
            )
        st.session_state.documents = local_documents
        st.session_state.judgments = {}
        save_corpus(local_documents)
        invalidate_index()
        st.success(f"Loaded {len(local_documents)} local documents.")
        st.rerun()


# -----------------------------
# Preprocessing / Mining
# -----------------------------
elif page == "Preprocessing":
    if not documents:
        st.warning("Collect documents first.")
    else:
        st.subheader("Text Preprocessing and Mining")
        strategies = ["none", "stopwords", "stem_stopwords", "lemma_stopwords"]
        selected_strategy = st.selectbox(
            "Preprocessing strategy",
            strategies,
            index=strategies.index(st.session_state.strategy),
        )
        if selected_strategy != st.session_state.strategy:
            st.session_state.strategy = selected_strategy
            st.info("Strategy changed. Rebuild the index in Index Management before searching.")
            ready = False

        top_n = st.slider("Top keywords per document", 3, 20, 8)
        texts = miner.preprocess(documents, st.session_state.strategy)

        st.subheader("Keyword extraction")
        keywords = miner.keywords(documents, top_n, st.session_state.strategy)
        selected = st.selectbox("Document", [document.doc_id for document in documents])
        st.dataframe(
            pd.DataFrame(keywords[selected], columns=["keyword", "tfidf"]),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Preprocessing strategy comparison")
        comparison = miner.compare_strategies(documents)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(
                comparison,
                x="strategy",
                y=["vocabulary_size", "avg_document_length"],
                barmode="group",
            ),
            use_container_width=True,
        )

        st.subheader("Document clustering")
        st.caption("KMeans is unsupervised clustering; do not describe it as supervised classification in the report.")
        cluster_count = (
            st.slider(
                "KMeans clusters",
                2,
                min(10, len(documents)),
                min(3, len(documents)),
            )
            if len(documents) > 1
            else 1
        )
        if st.button("Run Clustering"):
            temp_index = SearchIndex(
                documents,
                texts,
                strategy=st.session_state.strategy,
            )
            labels = miner.cluster(temp_index.matrix, cluster_count)
            st.dataframe(
                pd.DataFrame(
                    {
                        "doc_id": [document.doc_id for document in documents],
                        "cluster": labels,
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Document profiling")
        profile = pd.DataFrame(
            {
                "doc_id": [document.doc_id for document in documents],
                "title": [document.title for document in documents],
                "domain": [document.domain or "local" for document in documents],
                "word_count": [len(document.text.split()) for document in documents],
                "unique_terms": [len(set(text.split())) for text in texts],
            }
        )
        st.dataframe(profile, use_container_width=True, hide_index=True)

        st.subheader("Topic modeling")
        topic_count = st.slider(
            "LDA topics",
            2,
            min(10, max(2, len(documents))),
            min(3, max(2, len(documents))),
        )
        if st.button("Run Topic Modeling"):
            topic_rows = []
            for topic_number, scores in enumerate(
                miner.topics(texts, topic_count),
                start=1,
            ):
                topic_rows.append(
                    {
                        "topic": topic_number,
                        "terms": ", ".join(term for term, _ in scores),
                    }
                )
            st.dataframe(
                pd.DataFrame(topic_rows),
                use_container_width=True,
                hide_index=True,
            )


# -----------------------------
# Index Management
# -----------------------------
elif page == "Index Management":
    st.subheader("Index Management")
    if not documents:
        st.warning("Collect documents first.")
    else:
        if st.button("Build / Rebuild Index", type="primary"):
            with st.spinner("Building TF-IDF index and link graph..."):
                rebuild_index()
            st.success("Index rebuilt successfully.")
            st.rerun()

        if not ready:
            st.warning("The index is missing or stale. Build / Rebuild the index before Search, Recommendations, or Evaluation.")
        else:
            index = st.session_state.index
            a, b, c, d = st.columns(4)
            a.metric("Documents", len(index.documents))
            b.metric("Terms", len(index.inverted_index))
            c.metric("Graph edges", index.graph.number_of_edges())
            d.metric("Index build time", f"{st.session_state.last_index_seconds:.4f}s")

            st.subheader("Link graph")
            st.dataframe(
                pd.DataFrame(
                    {
                        "doc_id": list(index.pagerank),
                        "PageRank": [index.pagerank[d] for d in index.pagerank],
                        "Authority": [index.authorities.get(d, 0.0) for d in index.pagerank],
                        "Hub": [index.hubs.get(d, 0.0) for d in index.pagerank],
                    }
                ).sort_values("PageRank", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Explicit inverted index")
            st.caption("Posting-list view: term -> (document_id, term_frequency)")
            sample_terms = list(index.inverted_index.items())[:25]
            st.dataframe(
                pd.DataFrame(
                    {
                        "term": [term for term, _ in sample_terms],
                        "postings": [str(postings) for _, postings in sample_terms],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


# -----------------------------
# Search
# -----------------------------
elif page == "Search":
    if not ready:
        st.warning("Build / Rebuild the index first.")
    else:
        index = st.session_state.index
        st.subheader("Search")
        query = st.text_input("Search query")
        ranking_mode = st.selectbox(
            "Ranking strategy",
            [
                "tfidf_pagerank",
                "tfidf",
                "hits",
            ],
            format_func=lambda value: {
                "tfidf_pagerank": "TF-IDF + PageRank",
                "tfidf": "TF-IDF only",
                "hits": "TF-IDF + HITS authority",
            }[value],
        )
        weight = (
            st.slider("Link-ranking weight", 0.0, 1.0, 0.2)
            if ranking_mode != "tfidf"
            else 0.0
        )
        top_k = st.slider("Top-K", 1, max(1, len(documents)), min(10, len(documents)))

        search_started = time.perf_counter()
        results = index.search(query, top_k, weight, ranking_mode)
        st.session_state.last_search_seconds = time.perf_counter() - search_started

        if query:
            if results.empty:
                st.info("No indexed documents matched the query after preprocessing.")
            else:
                st.dataframe(results, use_container_width=True, hide_index=True)
                st.plotly_chart(
                    px.bar(
                        results,
                        x="doc_id",
                        y=["tfidf_similarity", "pagerank", "authority", "combined_score"],
                        barmode="group",
                    ),
                    use_container_width=True,
                )


# -----------------------------
# Recommendations
# -----------------------------
elif page == "Recommendations":
    if not ready:
        st.warning("Build / Rebuild the index first.")
    else:
        index = st.session_state.index
        st.subheader("Recommendations")
        selected = st.selectbox("Document", [document.doc_id for document in documents])
        mode = st.radio(
            "Method",
            ["Content-based", "Collaborative", "Hybrid"],
            horizontal=True,
        )
        top_k = st.slider(
            "Top-K recommendations",
            1,
            max(1, len(documents) - 1),
            min(5, max(1, len(documents) - 1)),
        )
        alpha = (
            st.slider("Hybrid content weight", 0.0, 1.0, 0.5)
            if mode == "Hybrid"
            else 0.5
        )

        interactions = None
        interaction_file = st.file_uploader(
            "Optional interaction CSV (columns: user_id, doc_id)",
            type=["csv"],
            key="interaction_csv",
        )
        if interaction_file is not None:
            try:
                interaction_frame = pd.read_csv(interaction_file)
                interactions = interactions_from_dataframe(
                    interaction_frame,
                    [document.doc_id for document in documents],
                )
                st.success(f"Loaded interactions for {len(interactions)} users.")
            except Exception as exc:
                st.error(f"Invalid interaction file: {exc}")

        recommender = Recommender(index, interactions=interactions)
        if recommender.interaction_source == "synthetic_demo" and mode != "Content-based":
            st.info("Collaborative/Hybrid mode is using deterministic synthetic interactions for demonstration. Upload a user-item CSV for real collaborative data.")

        recommendation_started = time.perf_counter()
        if mode == "Content-based":
            results = recommender.content(selected, top_k)
        elif mode == "Collaborative":
            results = recommender.collaborative(selected, top_k)
        else:
            results = recommender.hybrid(selected, alpha, top_k)
        st.session_state.last_recommendation_seconds = time.perf_counter() - recommendation_started
        st.dataframe(results, use_container_width=True, hide_index=True)


# -----------------------------
# Evaluation
# -----------------------------
elif page == "Evaluation":
    if not ready:
        st.warning("Build / Rebuild the index first.")
    else:
        index = st.session_state.index
        st.subheader("IR Evaluation")
        query = st.text_input("Evaluation query", key="evaluation_query")
        ranking_mode = st.selectbox(
            "Ranking strategy for evaluation",
            ["tfidf_pagerank", "tfidf", "hits"],
            format_func=lambda value: {
                "tfidf_pagerank": "TF-IDF + PageRank",
                "tfidf": "TF-IDF only",
                "hits": "TF-IDF + HITS authority",
            }[value],
            key="evaluation_ranking_mode",
        )
        weight = (
            st.slider("Link-ranking weight", 0.0, 1.0, 0.2, key="evaluation_weight")
            if ranking_mode != "tfidf"
            else 0.0
        )
        k = st.number_input(
            "Evaluate at K",
            1,
            max(1, len(documents)),
            min(5, len(documents)),
        )

        results = (
            index.search(query, len(documents), weight, ranking_mode)
            if query
            else pd.DataFrame()
        )
        if query and results.empty:
            st.info("No retrieved results for this evaluation query.")
        elif query:
            st.dataframe(results.head(int(k)), use_container_width=True, hide_index=True)
            relevant = st.multiselect(
                "Mark relevant documents",
                [document.doc_id for document in documents],
                key="evaluation_relevant",
            )

            metrics = evaluate(results.doc_id.tolist(), set(relevant), int(k))
            st.subheader("Current query metrics")
            st.dataframe(
                pd.DataFrame(
                    {
                        "metric": [
                            "Precision",
                            "Recall",
                            "F1",
                            "P@K",
                            "R@K",
                            "AP",
                            "RR",
                            "NDCG@K",
                        ],
                        "value": [
                            metrics["Precision"],
                            metrics["Recall"],
                            metrics["F1"],
                            metrics["P@K"],
                            metrics["R@K"],
                            metrics["AP"],
                            metrics["RR"],
                            metrics["NDCG@K"],
                        ],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            if st.button("Save Relevance Judgment"):
                st.session_state.judgments[query.strip()] = {
                    "query": query.strip(),
                    "retrieved": results.doc_id.tolist(),
                    "relevant": list(relevant),
                    "k": int(k),
                    "ranking_mode": ranking_mode,
                    "weight": float(weight),
                }
                st.success("Relevance judgment saved for this query.")

            comparison = evaluate_at_ks(
                results.doc_id.tolist(),
                set(relevant),
                sorted({1, min(3, len(documents)), int(k), len(documents)}),
            )
            st.subheader("Evaluation comparison by K")
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.plotly_chart(
                px.line(
                    comparison,
                    x="K",
                    y=["P@K", "R@K", "NDCG@K"],
                    markers=True,
                ),
                use_container_width=True,
            )

        st.subheader("Multi-query evaluation")
        if st.session_state.judgments:
            saved_rows = [
                {
                    "query": key,
                    "relevant_count": len(value["relevant"]),
                    "K": value["k"],
                    "ranking": value["ranking_mode"],
                }
                for key, value in st.session_state.judgments.items()
            ]
            st.dataframe(
                pd.DataFrame(saved_rows),
                use_container_width=True,
                hide_index=True,
            )
            aggregate = aggregate_evaluation(list(st.session_state.judgments.values()))
            st.caption("MAP and MRR below are macro-averages across the saved queries.")
            st.dataframe(aggregate, use_container_width=True, hide_index=True)
        else:
            st.info("Save judgments for at least two queries to report meaningful MAP/MRR across multiple queries.")


# -----------------------------
# Performance analytics
# -----------------------------
elif page == "Performance Analytics":
    total_words = sum(len(document.text.split()) for document in documents)
    total_kb = sum(len(document.text.encode("utf-8")) for document in documents) / 1024
    left, middle, right = st.columns(3)
    left.metric("Indexed documents", len(documents) if ready else 0)
    middle.metric("Average document length", round(total_words / len(documents), 2) if documents else 0)
    right.metric("Corpus storage", f"{total_kb:.1f} KB")

    timings = pd.DataFrame(
        {
            "component": [
                "Crawling",
                "Preprocessing + indexing",
                "Search",
                "Recommendations",
            ],
            "seconds": [
                st.session_state.last_crawl_seconds,
                st.session_state.last_index_seconds,
                st.session_state.last_search_seconds,
                st.session_state.last_recommendation_seconds,
            ],
        }
    )
    st.subheader("Component performance")
    st.dataframe(timings, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(timings, x="component", y="seconds", text_auto=".4f"),
        use_container_width=True,
    )
    st.metric(
        "Index throughput",
        (
            f"{len(documents) / max(st.session_state.last_index_seconds, 1e-9):.1f} documents/sec"
            if documents and ready
            else "0 documents/sec"
        ),
    )
    st.info("Crawling is network-bound; other timings are measured in the current Streamlit session.")
