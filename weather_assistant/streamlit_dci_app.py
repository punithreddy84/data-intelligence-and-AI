"""
Simple Educational Project: Dynamic Context Injection (DCI) using Groq
"""

import json
from typing import List, Dict, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
from groq import Groq


@st.cache_data
def load_sample_docs() -> List[Dict]:
    with open("sample_docs.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def setup_vector_db(docs: List[Dict]) -> Tuple[chromadb.Collection, SentenceTransformer]:

    client = chromadb.PersistentClient(path="./chroma_db")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    collection = client.get_or_create_collection(
        name="educational_dci_docs",
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:

        ids = [doc["id"] for doc in docs]
        contents = [doc["content"] for doc in docs]
        metadatas = [{"title": doc["title"]} for doc in docs]

        embeddings = embedding_model.encode(
            contents,
            show_progress_bar=False
        ).tolist()

        collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        st.success("✅ Vector database created!")

    else:
        st.info("📚 Using existing vector database.")

    return collection, embedding_model


def retrieve_dynamic_context(
        query: str,
        collection: chromadb.Collection,
        embedding_model: SentenceTransformer,
        top_k: int = 3
):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    for i in range(len(results["ids"][0])):

        retrieved.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "content": results["documents"][0][i],
            "relevance_score": round(
                1 - results["distances"][0][i],
                3
            )
        })

    return retrieved


def build_static_prompt(
        query: str,
        all_docs: List[Dict]
):

    context = "\n\n".join([
        f"### {doc['title']}\n{doc['content']}"
        for doc in all_docs
    ])

    return f"""
You are a helpful AI assistant for students learning about AI.

### ALL AVAILABLE KNOWLEDGE (Static Context)

{context}

### Student Question

{query}

Answer clearly and educationally.
"""


def build_dynamic_prompt(
        query: str,
        retrieved_context: List[Dict]
):

    context_parts = []

    for item in retrieved_context:
        context_parts.append(
            f"### {item['title']} "
            f"(Relevance: {item['relevance_score']})\n"
            f"{item['content']}"
        )

    dynamic_context = "\n\n".join(context_parts)

    return f"""
You are a helpful AI assistant for students learning about AI.

### RELEVANT KNOWLEDGE (Dynamically Injected)

{dynamic_context}

### Student Question

{query}

Answer clearly and educationally using only the relevant knowledge above.
"""


def get_groq_response(
        prompt: str,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile"
):

    if not api_key:
        return f"""
[DEMO MODE - No Groq API Key]

This prompt would be sent to Groq:

{'=' * 70}
{prompt}
{'=' * 70}
"""

    try:

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


def main():

    st.set_page_config(
        page_title="Dynamic Context Injection - Groq",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Dynamic Context Injection (Educational Demo)")
    st.subheader("Using Groq Llama 3.3")

    with st.sidebar:

        st.header("Settings")

        top_k = st.slider(
            "Top-K documents to retrieve",
            min_value=1,
            max_value=6,
            value=3
        )

        st.divider()

        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_..."
        )

        if groq_key:
            st.success("✅ API Key Loaded")
        else:
            st.warning("Enter your Groq API Key")

    docs = load_sample_docs()

    collection, embedding_model = setup_vector_db(docs)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Your Question")

        user_query = st.text_area(
            "Ask about AI:",
            value="What are the benefits of using dynamic context instead of putting all information in every prompt?",
            height=120
        )

        run = st.button(
            "Compare Static vs Dynamic",
            type="primary",
            use_container_width=True
        )

    with col2:

        st.subheader("Knowledge Base")

        for doc in docs:

            with st.expander(doc["title"]):
                st.write(doc["content"])

    if run and user_query.strip():

        st.divider()
        st.header("Comparison")

        with st.spinner("Generating Static Response..."):

            static_prompt = build_static_prompt(
                user_query,
                docs
            )

            static_response = get_groq_response(
                static_prompt,
                groq_key
            )

        with st.spinner("Generating Dynamic Response..."):

            retrieved = retrieve_dynamic_context(
                user_query,
                collection,
                embedding_model,
                top_k
            )

            dynamic_prompt = build_dynamic_prompt(
                user_query,
                retrieved
            )

            dynamic_response = get_groq_response(
                dynamic_prompt,
                groq_key
            )

        tab1, tab2 = st.tabs([
            "STATIC (All Docs)",
            "DYNAMIC (Smart Injection)"
        ])

        with tab1:

            st.caption(
                f"Approx Prompt Tokens: {len(static_prompt.split())}"
            )

            with st.expander("View Prompt"):
                st.code(static_prompt)

            st.write(static_response)

        with tab2:

            st.markdown("### Retrieved Context")

            for item in retrieved:
                st.success(
                    f"✅ {item['title']} "
                    f"(Score: {item['relevance_score']})"
                )

            st.caption(
                f"Approx Prompt Tokens: {len(dynamic_prompt.split())}"
            )

            with st.expander("View Prompt"):
                st.code(dynamic_prompt)

            st.write(dynamic_response)

        st.divider()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Static Docs",
            len(docs)
        )

        c2.metric(
            "Dynamic Docs",
            len(retrieved)
        )

        c3.metric(
            "Token Savings",
            max(
                0,
                len(static_prompt.split()) -
                len(dynamic_prompt.split())
            )
        )

        st.info(
            "Dynamic Context Injection sends only relevant documents to the LLM. "
            "This reduces token usage, improves focus, and lowers cost."
        )


if __name__ == "__main__":
    main()