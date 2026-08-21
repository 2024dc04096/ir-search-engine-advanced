import time

import pandas as pd
import plotly.express as px
import streamlit as st

from ir_system import Document, Recommender, SearchIndex, TextMiner, WebCrawler, document_frame, evaluate, evaluate_at_ks, load_corpus, save_corpus

st.set_page_config(page_title="IR Assignment 2", page_icon="IR", layout="wide")
st.title("End-to-End Information Retrieval")
st.caption("Crawl, mine, index, rank, recommend, and evaluate a document collection.")

if "documents" not in st.session_state:
    st.session_state.documents = load_corpus()
if "strategy" not in st.session_state:
    st.session_state.strategy = "stem_stopwords"

documents: list[Document] = st.session_state.documents
miner = TextMiner()
build_started = time.perf_counter()
texts = miner.preprocess(documents, st.session_state.strategy)
index = SearchIndex(documents, texts) if documents else None
st.session_state["last_index_seconds"] = time.perf_counter() - build_started
page = st.sidebar.radio("Module", ["Dashboard", "Crawling", "Preprocessing", "Index Management", "Search", "Recommendations", "Evaluation", "Performance Analytics"])
st.sidebar.metric("Documents", len(documents))

if page == "Dashboard":
    left, middle, right = st.columns(3)
    left.metric("Corpus documents", len(documents))
    middle.metric("Total words", sum(len(document.text.split()) for document in documents))
    right.metric("Domains", len({document.url.split('/')[2] for document in documents if '//' in document.url}))
    if documents:
        st.dataframe(document_frame(documents)[["doc_id", "title", "url", "depth", "content_length"]], use_container_width=True, hide_index=True)
    else:
        st.info("Start with Crawling, or add sample documents below.")
elif page == "Crawling":
    seeds = st.text_area("Seed URLs", "https://en.wikipedia.org/wiki/Information_retrieval\nhttps://en.wikipedia.org/wiki/Search_engine")
    depth = st.slider("Maximum depth", 0, 3, 1)
    max_pages = st.number_input("Maximum pages", 1, 100, 10)
    if st.button("Start Crawling", type="primary"):
        with st.spinner("Fetching pages..."):
            new_documents = WebCrawler().crawl(seeds.splitlines(), depth, int(max_pages))
        st.session_state.documents = new_documents
        save_corpus(new_documents)
        st.success(f"Collected {len(new_documents)} unique documents.")
    uploaded = st.file_uploader("Optional local text files", accept_multiple_files=True, type=["txt"])
    if uploaded and st.button("Load Local Files"):
        local_documents = [Document(f"D{number:04d}", file.name, file.name, file.read().decode("utf-8"), content_length=file.size, timestamp="local") for number, file in enumerate(uploaded, 1)]
        st.session_state.documents = local_documents
        save_corpus(local_documents)
        st.rerun()
elif page == "Preprocessing":
    if not documents:
        st.warning("Collect documents first.")
    else:
        strategy = st.selectbox("Preprocessing strategy", ["none", "stopwords", "stem_stopwords", "lemma_stopwords"], index=["none", "stopwords", "stem_stopwords", "lemma_stopwords"].index(st.session_state.strategy))
        st.session_state.strategy = strategy
        top_n = st.slider("Top keywords per document", 3, 20, 8)
        keywords = miner.keywords(documents, top_n, strategy)
        selected = st.selectbox("Document", [document.doc_id for document in documents])
        st.dataframe(pd.DataFrame(keywords[selected], columns=["keyword", "tfidf"]), use_container_width=True, hide_index=True)
        comparison = miner.compare_strategies(documents)
        st.plotly_chart(px.bar(comparison, x="strategy", y=["vocabulary_size", "avg_document_length"], barmode="group"), use_container_width=True)
        cluster_count = st.slider("KMeans clusters", 2, min(10, len(documents)), min(3, len(documents))) if len(documents) > 1 else 1
        if st.button("Run Clustering"):
            labels = miner.cluster(SearchIndex(documents, texts).matrix, cluster_count)
            st.dataframe(pd.DataFrame({"doc_id": [document.doc_id for document in documents], "cluster": labels}), use_container_width=True, hide_index=True)
        st.subheader("Document profiling")
        profile = pd.DataFrame({
            "doc_id": [document.doc_id for document in documents],
            "title": [document.title for document in documents],
            "domain": [document.domain or document.url.split('/')[2] if '//' in document.url else "local" for document in documents],
            "word_count": [len(document.text.split()) for document in documents],
            "unique_terms": [len(set(text.split())) for text in texts],
        })
        st.dataframe(profile, use_container_width=True, hide_index=True)
        topic_count = st.slider("LDA topics", 2, min(10, max(2, len(documents))), min(3, max(2, len(documents))))
        if st.button("Run Topic Modeling"):
            topic_vectorizer = SearchIndex(documents, texts).vectorizer
            topic_matrix = topic_vectorizer.transform(texts)
            topic_terms = topic_vectorizer.get_feature_names_out()
            topic_rows = []
            for topic_number, scores in enumerate(miner.topics(topic_matrix, topic_count), 1):
                topic_rows.append({"topic": topic_number, "terms": ", ".join(topic_terms[index] for index, _ in scores)})
            st.dataframe(pd.DataFrame(topic_rows), use_container_width=True, hide_index=True)
elif page == "Index Management":
    if index:
        st.success("TF-IDF inverted index is ready.")
        st.metric("Terms", len(index.vectorizer.get_feature_names_out()))
        st.dataframe(pd.DataFrame({"doc_id": list(index.pagerank), "PageRank": list(index.pagerank.values()), "Authority": [index.authorities.get(item, 0) for item in index.pagerank]}), use_container_width=True, hide_index=True)
    else:
        st.warning("Collect documents first.")
elif page == "Search":
    if index:
        query = st.text_input("Search query")
        weight = st.slider("PageRank weight", 0.0, 1.0, 0.2)
        top_k = st.slider("Top-K", 1, max(1, len(documents)), min(10, len(documents)))
        search_started = time.perf_counter()
        results = index.search(query, top_k, weight)
        st.session_state["last_search_seconds"] = time.perf_counter() - search_started
        if query:
            st.dataframe(results, use_container_width=True, hide_index=True)
            if not results.empty:
                st.plotly_chart(px.bar(results, x="doc_id", y=["tfidf_similarity", "pagerank", "combined_score"], barmode="group"), use_container_width=True)
    else:
        st.warning("Collect documents first.")
elif page == "Recommendations":
    if index:
        recommender = Recommender(index)
        selected = st.selectbox("Document", [document.doc_id for document in documents])
        mode = st.radio("Method", ["Content-based", "Collaborative", "Hybrid"], horizontal=True)
        top_k = st.slider("Top-K recommendations", 1, max(1, len(documents) - 1), min(5, max(1, len(documents) - 1)))
        alpha = st.slider("Hybrid content weight", 0.0, 1.0, 0.5) if mode == "Hybrid" else 0.5
        recommendation_started = time.perf_counter()
        results = recommender.content(selected, top_k) if mode == "Content-based" else recommender.collaborative(selected, top_k) if mode == "Collaborative" else recommender.hybrid(selected, alpha, top_k)
        st.session_state["last_recommendation_seconds"] = time.perf_counter() - recommendation_started
        st.dataframe(results, use_container_width=True, hide_index=True)
    else:
        st.warning("Collect documents first.")
elif page == "Evaluation":
    if index:
        query = st.text_input("Query to evaluate")
        results = index.search(query, len(documents), 0.2) if query else pd.DataFrame()
        relevant = st.multiselect("Mark relevant documents", [document.doc_id for document in documents])
        k = st.number_input("Evaluate at K", 1, max(1, len(documents)), min(5, len(documents)))
        if st.button("Compute Metrics"):
            metrics = evaluate(results.doc_id.tolist(), set(relevant), int(k))
            st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}), use_container_width=True)
            comparison = evaluate_at_ks(results.doc_id.tolist(), set(relevant), sorted({1, min(3, len(documents)), int(k), len(documents)}))
            st.subheader("Evaluation comparison by K")
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.plotly_chart(px.line(comparison, x="K", y=["P@K", "R@K", "NDCG@K"], markers=True), use_container_width=True)
    else:
        st.warning("Collect documents first.")
elif page == "Performance Analytics":
    total_words = sum(len(document.text.split()) for document in documents)
    total_kb = sum(len(document.text.encode('utf-8')) for document in documents) / 1024
    left, middle, right = st.columns(3)
    left.metric("Indexed documents", len(documents))
    middle.metric("Average document length", round(total_words / len(documents), 2) if documents else 0)
    right.metric("Storage", f"{total_kb:.1f} KB")
    timings = pd.DataFrame({
        "component": ["Preprocessing + indexing", "Search", "Recommendations"],
        "seconds": [st.session_state.get("last_index_seconds", 0), st.session_state.get("last_search_seconds", 0), st.session_state.get("last_recommendation_seconds", 0)],
    })
    st.subheader("Component performance")
    st.dataframe(timings, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(timings, x="component", y="seconds", text_auto=".4f"), use_container_width=True)
    st.metric("Index throughput", f"{len(documents) / max(st.session_state.get('last_index_seconds', 0), 1e-9):.1f} documents/sec" if documents else "0 documents/sec")
    st.info("Crawling is network-bound; preprocessing, indexing, search, and recommendation timings are measured during the current Streamlit session.")