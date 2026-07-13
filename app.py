"""
Document Q&A over RPR.0534.1.0.BN.DZ0001 (Rooppur NPP Unit 1 Technical
Specification) using the Google Gemini API (free tier), served as a
Streamlit app.

How it works
------------
1. On startup, the full text of the document is loaded from document.txt
   and split into overlapping page-chunks.
2. A TF-IDF index is built over the chunks (once, cached).
3. When the user asks a question, the app finds the most relevant chunks
   (by keyword/TF-IDF similarity) instead of sending the ENTIRE document.
4. Only those chunks + the question are sent to the Gemini API, and the
   answer is streamed back with page citations.

Why retrieval instead of sending the whole document: the document is
~250K tokens, which is right at (and sometimes over) the free tier's
per-minute input-token quota (250,000 TPM as of writing) -- a single
"send everything" request can trip a 429 RESOURCE_EXHAUSTED error, and
even when it doesn't, it burns your whole per-minute budget on one
question. Retrieval keeps each request small (a few thousand tokens),
so you comfortably stay inside the free tier and can ask many questions
per minute.

Get a free API key (no credit card) at https://aistudio.google.com/app/apikey
"""

import os
import re
import streamlit as st
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = "gemini-3.5-flash"  # current GA model, free-tier eligible
DOCUMENT_PATH = os.path.join(os.path.dirname(__file__), "document.txt")
DOCUMENT_TITLE = "RPR.0534.1.0.BN.DZ0001 -- Technical Specification of Safe Operation of Rooppur NPP Unit 1 (Version 2)"

CHUNK_SIZE_PAGES = 3   # pages per chunk
CHUNK_STRIDE_PAGES = 2  # overlap of 1 page between consecutive chunks
TOP_K_CHUNKS = 12       # how many chunks to retrieve per question

SYSTEM_PROMPT = f"""You are a careful technical assistant answering questions \
about a single reference document: {DOCUMENT_TITLE}.

You are given a set of EXCERPTS retrieved from the document (not the whole
document) -- they are the pages judged most relevant to the question.

Rules:
- Answer ONLY using the excerpts provided. Do not use outside knowledge
  about nuclear plants, Rooppur, or regulations in general.
- Each excerpt is labeled with page markers like "[PAGE 12]". Cite the
  page number(s) you drew the answer from, e.g. "(p. 37)" or "(pp. 40-41)".
- If the excerpts don't contain the answer, say so plainly -- don't guess,
  and mention that the relevant section may not have been retrieved (the
  user can try rephrasing the question to help retrieval find it).
- Quote short phrases only when exact wording matters (e.g. a limit value
  or defined term); otherwise explain in your own words.
- Be precise with numbers, units, and parameter names -- this is a nuclear
  safety document and accuracy matters.
"""

# --------------------------------------------------------------------------
# Document loading + chunking + retrieval
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading and indexing document...")
def build_index():
    """Load document.txt, split into overlapping page-chunks, and build a
    TF-IDF index over them. Runs once per server process."""
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    parts = re.split(r"\[PAGE (\d+)\]\n", text)
    pages = []
    for i in range(1, len(parts), 2):
        pages.append((int(parts[i]), parts[i + 1]))

    chunks = []
    i = 0
    while i < len(pages):
        group = pages[i : i + CHUNK_SIZE_PAGES]
        if not group:
            break
        page_nums = [p[0] for p in group]
        chunk_text = "\n".join(f"[PAGE {p[0]}]\n{p[1]}" for p in group)
        chunks.append({"pages": page_nums, "text": chunk_text})
        if i + CHUNK_SIZE_PAGES >= len(pages):
            break
        i += CHUNK_STRIDE_PAGES

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])

    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "num_pages": len(pages),
    }


def retrieve_chunks(index: dict, query: str, top_k: int = TOP_K_CHUNKS):
    """Return the top_k chunks most relevant to the query, sorted back into
    page order (so the excerpt reads coherently top-to-bottom)."""
    qvec = index["vectorizer"].transform([query])
    sims = cosine_similarity(qvec, index["matrix"])[0]
    top_idx = sims.argsort()[::-1][:top_k]
    selected = [index["chunks"][i] for i in top_idx if sims[i] > 0]
    # de-dupe overlapping pages, then sort by first page number
    selected.sort(key=lambda c: c["pages"][0])
    return selected


def get_api_key() -> str | None:
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GEMINI_API_KEY")
    if not key:
        key = st.session_state.get("manual_api_key")
    return key


def build_contents(excerpt_text: str, chat_history: list[dict], question: str) -> list[dict]:
    """Build the `contents` list for the API call. Unlike the whole-document
    approach, we send the (small) retrieved excerpt fresh with EVERY turn,
    since which excerpt is relevant changes per question."""
    contents = []

    for turn in chat_history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})

    contents.append(
        {
            "role": "user",
            "parts": [
                {
                    "text": f"Relevant excerpts from the document:\n\n{excerpt_text}\n\n"
                    f"Question: {question}"
                }
            ],
        }
    )

    return contents


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Document Q&A -- RPR.0534.1.0.BN.DZ0001", page_icon="[Q&A]")
st.title("Document Q&A")
st.caption(DOCUMENT_TITLE)

with st.sidebar:
    st.header("Setup")
    api_key = get_api_key()
    if not api_key:
        st.warning("No GEMINI_API_KEY found in secrets or environment.")
        st.markdown("Get a free key (no credit card) at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).")
        manual_key = st.text_input("Enter your Gemini API key", type="password")
        if manual_key:
            st.session_state["manual_api_key"] = manual_key
            api_key = manual_key
    else:
        st.success("API key loaded.")

    st.divider()
    index = build_index()
    st.markdown(
        "**Model:** `{}`\n\n"
        "**Document:** {} pages, indexed into {} searchable chunks.".format(
            MODEL, index["num_pages"], len(index["chunks"])
        )
    )
    st.caption(
        "Each question retrieves only the most relevant pages instead of "
        "sending the whole document -- this keeps requests small and "
        "fast, and fits comfortably inside the free-tier quota."
    )

    if st.button("Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for turn in st.session_state["chat_history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about the document...")

if question:
    if not api_key:
        st.error("Please provide a Gemini API key in the sidebar first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    index = build_index()
    retrieved = retrieve_chunks(index, question)

    if not retrieved:
        with st.chat_message("assistant"):
            answer_text = (
                "I couldn't find any pages in the document that match this "
                "question closely enough. Try rephrasing it or using terms "
                "more likely to appear in the document."
            )
            st.markdown(answer_text)
        st.session_state["chat_history"].append({"role": "user", "content": question})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})
        st.stop()

    excerpt_text = "\n\n---\n\n".join(c["text"] for c in retrieved)
    pages_used = sorted({p for c in retrieved for p in c["pages"]})

    client = genai.Client(api_key=api_key)
    contents = build_contents(excerpt_text, st.session_state["chat_history"], question)

    with st.chat_message("assistant"):
        st.caption(f"Searched pages: {pages_used[0]}-{pages_used[-1]} ({len(retrieved)} excerpts retrieved)")
        placeholder = st.empty()
        answer_text = ""
        try:
            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=20000,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    answer_text += chunk.text
                    placeholder.markdown(answer_text + "▌")
            placeholder.markdown(answer_text)
        except Exception as e:
            answer_text = f"Error calling the Gemini API: {e}"
            placeholder.error(answer_text)

    st.session_state["chat_history"].append({"role": "user", "content": question})
    st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})
