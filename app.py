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

Chat history is persisted in Supabase (Postgres) so it survives page
reloads and app restarts. Set SUPABASE_URL and SUPABASE_KEY in
.streamlit/secrets.toml (or as environment variables) -- see
supabase_schema.sql for the table this app expects. If those aren't
set, the app still works, it just keeps history only in memory for the
current browser tab.

Get a free API key (no credit card) at https://aistudio.google.com/app/apikey
"""

import difflib
import io
import os
import re
import time
import uuid
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
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
MIN_SIMILARITY = 0.03   # below this combined score, a chunk is too weak to use

MAX_API_RETRIES = 3          # attempts for transient Gemini errors (503/429/500...)
RETRY_BACKOFF_SECONDS = 2    # doubles each retry: 2s, 4s, ...

SUPABASE_TABLE = "qa_chat_messages"  # see supabase_schema.sql

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
- Be precise with numbers, units, and parameter names -- this is a nuclear
  safety document and accuracy matters.

Language:
- Reply in the same language the user asked in.
- If the question is in Bangla (Bengali script or Banglish/romanized
  Bangla), answer in natural mixed Bangla-English (Bangla বাক্যগঠন দিয়ে
  ব্যাখ্যা করবে) -- keep technical terms, units, numbers, and page
  citations in English (e.g. "pressure", "coolant", "p. 37"), since
  those don't have natural Bangla equivalents in this document, but
  write the surrounding explanation in Bangla, the way a Bangladeshi
  engineer would naturally speak/write about this topic.
- If the question is in English, answer in English as usual.

How to write the answer (plain-language style):
- Explain things in your OWN words, the way you'd explain it to an
  intelligent but non-specialist reader. Do NOT lift whole sentences
  verbatim from the document -- paraphrase, and spell out what a rule or
  number actually MEANS in practice, not just what it says.
- Quote short phrases (a few words) only when exact wording genuinely
  matters, e.g. a defined term or a precise limit value.
- Prefer short paragraphs, plain vocabulary, and bullet points over dense
  technical prose. If a term is jargon, briefly say what it means the
  first time you use it.
- After explaining, you may add one short "In short:" takeaway line if it
  helps the reader.

Diagrams and charts (use only when they genuinely help):
- If the answer describes a process, sequence of steps, a hierarchy, or
  how parts of a system relate to each other, you may include a small
  diagram as a fenced code block labeled ```mermaid```, using valid
  Mermaid syntax (e.g. flowchart TD / graph TD, or sequenceDiagram).
- If the answer compares several related numbers (e.g. limits for
  different parameters, values across categories or conditions), you may
  include a fenced code block labeled ```chart``` containing simple CSV
  data with a header row, for example:
  ```chart
  Parameter,Value
  Max Pressure (MPa),10
  Max Temperature (C),25
  ```
- Only include ONE diagram or chart if it truly clarifies the answer --
  do not add one to every response, and never invent numbers that aren't
  in the excerpts.
"""

# --------------------------------------------------------------------------
# Document loading + chunking + retrieval
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading and indexing document...")
def build_index():
    """Load document.txt, split into overlapping page-chunks, and build TWO
    TF-IDF indexes over them:
    - a word-level index (precise, but misses on misspelled words)
    - a character n-gram index (fuzzy -- a misspelled word still shares
      most of its letter sequences with the correct one, so this finds
      the right pages even without correcting the query).
    Runs once per server process."""
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

    chunk_texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(chunk_texts)

    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)
    char_matrix = char_vectorizer.fit_transform(chunk_texts)

    # Vocabulary of the document's own words, used to spell-correct queries
    # against terms that actually appear in the document (rather than a
    # generic English dictionary, which wouldn't know domain terms).
    vocabulary = {w for w in vectorizer.vocabulary_.keys() if len(w) >= 4}

    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "char_vectorizer": char_vectorizer,
        "char_matrix": char_matrix,
        "vocabulary": vocabulary,
        "num_pages": len(pages),
    }


def correct_query(query: str, vocabulary: set) -> tuple[str, bool]:
    """Best-effort spelling correction: for each word in the query that
    ISN'T in the document's own vocabulary, snap it to the closest word
    that IS (via fuzzy string matching), if one is close enough. Returns
    (corrected_query, was_anything_changed).

    Deliberately conservative: short words (e.g. transliterated Bangla like
    "kore", "jodi") are left alone, and a candidate must start with the same
    letter as the original word, since real typos almost never change the
    first letter -- this avoids "correcting" a non-English word into an
    unrelated English one that merely looks similar.
    """
    tokens = query.split()
    changed = False
    for idx, tok in enumerate(tokens):
        stripped = re.sub(r"[^A-Za-z]", "", tok).lower()
        if len(stripped) < 5 or stripped in vocabulary:
            continue
        same_start = {w for w in vocabulary if w[0] == stripped[0]}
        match = difflib.get_close_matches(stripped, same_start, n=1, cutoff=0.82)
        if match and match[0] != stripped:
            tokens[idx] = match[0]
            changed = True
    return " ".join(tokens), changed


def retrieve_chunks(
    index: dict,
    query: str,
    top_k: int = TOP_K_CHUNKS,
    word_weight: float = 0.7,
    char_weight: float = 0.3,
):
    """Return the top_k chunks most relevant to the query, sorted back into
    page order (so the excerpt reads coherently top-to-bottom). Combines
    word-level similarity (precise) with character n-gram similarity
    (typo-tolerant) so a misspelled query word still surfaces the right
    pages even if it wasn't corrected."""
    word_sims = cosine_similarity(index["vectorizer"].transform([query]), index["matrix"])[0]
    char_sims = cosine_similarity(
        index["char_vectorizer"].transform([query]), index["char_matrix"]
    )[0]
    combined = word_weight * word_sims + char_weight * char_sims

    top_idx = combined.argsort()[::-1][:top_k]
    selected = [index["chunks"][i] for i in top_idx if combined[i] > MIN_SIMILARITY]
    # de-dupe overlapping pages, then sort by first page number
    selected.sort(key=lambda c: c["pages"][0])
    return selected


def is_transient_api_error(e: Exception) -> bool:
    """True for errors worth retrying automatically: server overload (503),
    rate limiting (429), or generic backend hiccups (500) -- as opposed to
    e.g. a bad API key, which retrying won't fix."""
    msg = str(e)
    return any(code in msg for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"))


def is_error_message(text: str) -> bool:
    """True if an assistant message is one of our own API-failure messages
    (as opposed to a normal answer, or the 'nothing relevant found' message)."""
    return text.startswith("Sorry, the Gemini API is temporarily overloaded") or text.startswith(
        "Error calling the Gemini API:"
    )


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


def _get_secret(name: str) -> str | None:
    """Look up a config value in st.secrets first, then env vars."""
    value = None
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    if not value:
        value = os.environ.get(name)
    return value


@st.cache_resource(show_spinner=False)
def get_supabase_client():
    """Create a Supabase client if SUPABASE_URL / SUPABASE_KEY are set.
    Returns None (and the app falls back to in-memory-only history) if
    they're missing or the client can't be created."""
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def load_history_from_supabase(client, session_id: str) -> list[dict]:
    """Load all messages for a conversation, oldest first."""
    if client is None:
        return []
    try:
        res = (
            client.table(SUPABASE_TABLE)
            .select("role, content")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return [{"role": row["role"], "content": row["content"]} for row in res.data]
    except Exception:
        return []


def save_message_to_supabase(client, session_id: str, role: str, content: str) -> None:
    if client is None:
        return
    try:
        client.table(SUPABASE_TABLE).insert(
            {"session_id": session_id, "role": role, "content": content}
        ).execute()
    except Exception:
        pass  # logging failures shouldn't break the chat


def clear_session_in_supabase(client, session_id: str) -> None:
    if client is None:
        return
    try:
        client.table(SUPABASE_TABLE).delete().eq("session_id", session_id).execute()
    except Exception:
        pass


def list_past_sessions(client, limit: int = 20) -> list[dict]:
    """Return past conversations (excluding none in particular -- caller
    filters out the current one), each as {session_id, first_question,
    started_at}, most recently-started first."""
    if client is None:
        return []
    try:
        res = (
            client.table(SUPABASE_TABLE)
            .select("session_id, content, created_at")
            .eq("role", "user")
            .order("created_at", desc=False)
            .limit(1000)
            .execute()
        )
        first_seen = {}
        for row in res.data:
            sid = row["session_id"]
            if sid not in first_seen:
                first_seen[sid] = {
                    "session_id": sid,
                    "first_question": row["content"],
                    "started_at": row["created_at"],
                }
        sessions = sorted(first_seen.values(), key=lambda s: s["started_at"], reverse=True)
        return sessions[:limit]
    except Exception:
        return []


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
# Rendering: turn ```mermaid``` / ```chart``` fenced blocks in the model's
# answer into an actual diagram / bar chart instead of raw text.
# --------------------------------------------------------------------------

_VISUAL_BLOCK_RE = re.compile(r"```(mermaid|chart)\n(.*?)```", re.DOTALL)


def render_mermaid(code: str, height: int = 420) -> None:
    """Render a Mermaid diagram using mermaid.js via a components.html iframe."""
    html = f"""
    <div class="mermaid">
    {code}
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.9.1/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
    </script>
    """
    components.html(html, height=height, scrolling=True)


def render_chart(csv_text: str) -> None:
    """Render a small ```chart``` CSV block (header + rows) as a bar chart."""
    try:
        df = pd.read_csv(io.StringIO(csv_text.strip()))
        if df.shape[1] >= 2:
            df = df.set_index(df.columns[0])
            st.bar_chart(df)
        else:
            st.code(csv_text.strip())
    except Exception:
        st.code(csv_text.strip())


def render_answer(text: str) -> None:
    """Split the model's answer on ```mermaid```/```chart``` fenced blocks and
    render each part appropriately (markdown text, diagram, or chart)."""
    parts = _VISUAL_BLOCK_RE.split(text)
    idx, n = 0, len(parts)
    while idx < n:
        if idx % 3 == 0:
            segment = parts[idx]
            if segment and segment.strip():
                st.markdown(segment)
            idx += 1
        else:
            tag = parts[idx]
            content = parts[idx + 1] if idx + 1 < n else ""
            if tag == "mermaid":
                render_mermaid(content)
            elif tag == "chart":
                render_chart(content)
            idx += 2


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Document Q&A -- RPR.0534.1.0.BN.DZ0001", page_icon="[Q&A]")
st.title("Document Q&A")
st.caption(DOCUMENT_TITLE)

supabase_client = get_supabase_client()

# The conversation id lives in the URL (?session=...) so reloading the page
# (or sharing the link) keeps you in the same conversation.
if "session_id" not in st.session_state:
    existing = st.query_params.get("session")
    st.session_state["session_id"] = existing if existing else str(uuid.uuid4())
    st.query_params["session"] = st.session_state["session_id"]
session_id = st.session_state["session_id"]

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

    st.divider()
    st.header("History")
    if supabase_client is None:
        st.caption(
            "Conversation history isn't being saved permanently -- "
            "SUPABASE_URL / SUPABASE_KEY aren't configured, so history "
            "only lasts for this browser tab."
        )
    else:
        st.caption(f"Conversation ID: `{session_id[:8]}`")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New chat", use_container_width=True):
                new_id = str(uuid.uuid4())
                st.session_state["session_id"] = new_id
                st.session_state["chat_history"] = []
                st.query_params["session"] = new_id
                st.rerun()
        with col2:
            if st.button("Delete this chat", use_container_width=True):
                clear_session_in_supabase(supabase_client, session_id)
                st.session_state["chat_history"] = []
                st.rerun()

        past_sessions = [
            s for s in list_past_sessions(supabase_client) if s["session_id"] != session_id
        ]
        if past_sessions:
            with st.expander(f"Past conversations ({len(past_sessions)})"):
                for s in past_sessions:
                    q = s["first_question"].strip().replace("\n", " ")
                    label = q[:60] + ("..." if len(q) > 60 else "")
                    if st.button(label or "(untitled)", key=f"switch_{s['session_id']}"):
                        st.session_state["session_id"] = s["session_id"]
                        st.session_state["chat_history"] = load_history_from_supabase(
                            supabase_client, s["session_id"]
                        )
                        st.query_params["session"] = s["session_id"]
                        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = load_history_from_supabase(supabase_client, session_id)

history = st.session_state["chat_history"]
for i, turn in enumerate(history):
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            render_answer(turn["content"])
            is_last_turn = i == len(history) - 1
            if is_last_turn and is_error_message(turn["content"]):
                if st.button("🔄 Retry", key=f"retry_{session_id}_{i}"):
                    failed_question = history[i - 1]["content"] if i > 0 else None
                    if failed_question:
                        st.session_state["chat_history"] = history[: i - 1]
                        st.session_state["retry_question"] = failed_question
                        st.rerun()
        else:
            st.markdown(turn["content"])

question = st.chat_input("Ask a question about the document...")
if not question:
    question = st.session_state.pop("retry_question", None)

if question:
    if not api_key:
        st.error("Please provide a Gemini API key in the sidebar first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    index = build_index()
    search_query, was_corrected = correct_query(question, index["vocabulary"])
    retrieved = retrieve_chunks(index, search_query)

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
        save_message_to_supabase(supabase_client, session_id, "user", question)
        save_message_to_supabase(supabase_client, session_id, "assistant", answer_text)
        st.stop()

    excerpt_text = "\n\n---\n\n".join(c["text"] for c in retrieved)
    pages_used = sorted({p for c in retrieved for p in c["pages"]})

    client = genai.Client(api_key=api_key)
    contents = build_contents(excerpt_text, st.session_state["chat_history"], question)

    with st.chat_message("assistant"):
        if was_corrected:
            st.caption(f"Also searched for: \"{search_query}\" (in case of a typo)")
        st.caption(f"Searched pages: {pages_used[0]}-{pages_used[-1]} ({len(retrieved)} excerpts retrieved)")
        placeholder = st.empty()
        answer_text = ""
        last_error = None

        for attempt in range(1, MAX_API_RETRIES + 1):
            answer_text = ""
            try:
                stream = client.models.generate_content_stream(
                    model=MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=80000,
                    ),
                )
                for chunk in stream:
                    if chunk.text:
                        answer_text += chunk.text
                        placeholder.markdown(answer_text + "▌")
                # Streaming is done -- clear the raw placeholder and render the
                # final answer properly (turns any ```mermaid``` / ```chart```
                # blocks into an actual diagram or bar chart).
                placeholder.empty()
                render_answer(answer_text)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_API_RETRIES and is_transient_api_error(e):
                    wait_seconds = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    placeholder.info(
                        f"Gemini is under high demand right now -- retrying in "
                        f"{wait_seconds}s (attempt {attempt}/{MAX_API_RETRIES})..."
                    )
                    time.sleep(wait_seconds)
                else:
                    break

        if last_error is not None:
            if is_transient_api_error(last_error):
                answer_text = (
                    "Sorry, the Gemini API is temporarily overloaded and didn't "
                    "respond after a few tries. This is usually short-lived -- "
                    "please try asking again in a moment."
                )
            else:
                answer_text = f"Error calling the Gemini API: {last_error}"
            placeholder.error(answer_text)

    st.session_state["chat_history"].append({"role": "user", "content": question})
    st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})
    save_message_to_supabase(supabase_client, session_id, "user", question)
    save_message_to_supabase(supabase_client, session_id, "assistant", answer_text)
