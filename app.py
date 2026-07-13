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
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = "gemini-3.5-flash"          # accurate mode -- current GA model, free-tier eligible
FAST_MODEL = "gemini-3.1-flash-lite"  # fast mode -- Google's low-latency, high-throughput tier

DOCUMENT_PATH = os.path.join(os.path.dirname(__file__), "document.txt")
DOCUMENT_TITLE = "RPR.0534.1.0.BN.DZ0001 -- Technical Specification of Safe Operation of Rooppur NPP Unit 1 (Version 2)"

CHUNK_SIZE_PAGES = 3   # pages per chunk
CHUNK_STRIDE_PAGES = 2  # overlap of 1 page between consecutive chunks

# Fewer/more chunks and a shorter reply cap trade a little context for a lot
# of latency -- smaller prompts reach the model faster and there's less for
# it to generate before you see the full answer.
TOP_K_CHUNKS = 8         # was 12 -- smaller prompt, faster first token
MIN_SIMILARITY = 0.03    # below this combined score, a chunk is too weak to use
MAX_OUTPUT_TOKENS = 6192 # was 4096 -- 4096 was cutting off longer/detailed answers

# Hard cap on the RETRIEVED EXCERPT text sent per request, regardless of how
# many chunks TOP_K_CHUNKS picks. Some page ranges (dense tables, protocol
# lists) are much heavier per-page than others, so chunk COUNT alone doesn't
# bound token count -- this does. 80,000 tokens leaves comfortable headroom
# under the free tier's 250,000 TPM quota once you add the system prompt,
# chat history, and output tokens, which is what was tripping the
# "high demand" retry loop. ~4 chars/token is a safe rough estimate for
# English/technical text (real tokenizer isn't available client-side).
MAX_EXCERPT_TOKENS = 50000
CHARS_PER_TOKEN_ESTIMATE = 4
THINKING_LEVEL = "low"   # Gemini 3.x "thinking" budget: low = fast, minimal deliberation

MAX_API_RETRIES = 3          # attempts for transient Gemini errors (503/429/500...)
RETRY_BACKOFF_SECONDS = 2    # doubles each retry: 2s, 4s, ...

SUPABASE_TABLE = "qa_chat_messages"       # see supabase_schema.sql
SUPABASE_FEEDBACK_TABLE = "qa_feedback"   # thumbs-up/down learning signal, see supabase_schema.sql
FEEDBACK_EXAMPLES_K = 2                   # how many past 👍 answers to reuse as style/accuracy examples

# --------------------------------------------------------------------------
# Fallback provider: Groq (free, no card, OpenAI-compatible). Used ONLY when
# Gemini itself reports a quota/overload error after retrying -- i.e. an
# automatic "use whichever is available" path, not a primary switch. Groq's
# free tier has a much higher requests-per-minute ceiling than Gemini's free
# tier, but a much SMALLER tokens-per-minute ceiling (6,000 TPM vs Gemini's
# 250,000), so the fallback path retrieves fewer/smaller excerpts -- it
# trades a bit of document coverage for actually getting an answer through.
# Quality/citation discipline may differ slightly from Gemini since this is
# a different (open-weights) model -- flagged clearly in the UI when used.
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TOP_K_CHUNKS = 3
GROQ_MAX_EXCERPT_TOKENS = 3000
GROQ_MAX_OUTPUT_TOKENS = 2048
GROQ_MAX_RETRIES = 2

SYSTEM_PROMPT = f"""You are a careful technical assistant answering questions \
about a single reference document: {DOCUMENT_TITLE}.

You are given a set of EXCERPTS retrieved from the document (not the whole
document) -- they are the pages judged most relevant to the question.

Rules:
- PREFER the excerpts over everything else. For anything the excerpts do
  answer, answer ONLY from them.
- Each excerpt is labeled with page markers like "[PAGE 12]". Cite the
  page number(s) you drew the answer from, e.g. "(p. 37)" or "(pp. 40-41)".
- Be precise with numbers, units, and parameter names -- this is a nuclear
  safety document and accuracy matters. Never invent a page number, a
  value, or a citation.

Outside knowledge (clearly labeled, used only when the document doesn't cover it):
- If the excerpts don't contain the answer, but you can reasonably answer
  from general nuclear-engineering, regulatory, or technical knowledge --
  or by logically connecting/mapping something the excerpts DO say to what
  is being asked -- you may do so, but you MUST clearly separate it out
  under its own heading: "🌐 Outside information (not from this document)".
- Never blend outside knowledge into a document-cited sentence without that
  label. The reader must always be able to tell, at a glance, which part
  of your answer came from RPR.0534.1.0.BN.DZ0001 and which came from
  general knowledge or inference.
- Under that heading, briefly say WHY you believe it (general practice,
  standard nuclear engineering convention, inference from a related rule
  in the excerpts, etc.), and note that it should be verified against the
  authoritative document/standard before being relied on operationally.
- If you have neither a document answer nor a credible outside answer, say
  so plainly rather than guessing, and suggest the user rephrase the
  question to help retrieval find the right pages.

Learning from past helpful answers:
- You may be given a few "Previously confirmed helpful" Q&A examples --
  real answers to similar past questions that a user marked 👍. Treat them
  as a guide to preferred phrasing, structure, and level of detail, and as
  a consistency check (don't contradict a previously-confirmed answer
  without good reason). Never treat them as a substitute for the actual
  excerpts -- if they conflict with the current excerpts, the excerpts win.

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
    max_excerpt_tokens: int = MAX_EXCERPT_TOKENS,
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
    ranked = [index["chunks"][i] for i in top_idx if combined[i] > MIN_SIMILARITY]

    # Enforce the token budget: walk the list in relevance order (most
    # relevant first) and keep adding chunks until the next one would push
    # the estimated total over the budget. Always keep at least the single
    # best chunk, even if it alone is large, so a question never comes back
    # completely empty-handed.
    selected = []
    budget_chars = max_excerpt_tokens * CHARS_PER_TOKEN_ESTIMATE
    used_chars = 0
    for chunk in ranked:
        chunk_chars = len(chunk["text"])
        if selected and used_chars + chunk_chars > budget_chars:
            continue
        selected.append(chunk)
        used_chars += chunk_chars

    # de-dupe overlapping pages, then sort by first page number
    selected.sort(key=lambda c: c["pages"][0])
    return selected


def classify_api_error(e: Exception) -> str:
    """Classify an API exception so the UI can show something actionable
    instead of one generic 'high demand' message for every case:
    - 'quota'     : 429 / RESOURCE_EXHAUSTED -- YOUR project hit its RPM/TPM/
                    RPD limit. Waiting a bit helps; asking less often helps
                    more. Not Google being 'busy'.
    - 'overload'  : 503 / UNAVAILABLE -- Google's backend is overloaded.
                    Genuinely not your fault; retrying is the right move.
    - 'internal'  : 500 / INTERNAL -- generic backend hiccup, also worth
                    retrying.
    - 'other'     : not automatically retryable (bad key, bad model name,
                    malformed request, etc).
    """
    msg = str(e)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "quota"
    if "503" in msg or "UNAVAILABLE" in msg:
        return "overload"
    if "500" in msg or "INTERNAL" in msg:
        return "internal"
    return "other"


def is_transient_api_error(e: Exception) -> bool:
    """True for errors worth retrying automatically (quota/overload/internal)
    -- as opposed to e.g. a bad API key or bad model name, which retrying
    won't fix."""
    return classify_api_error(e) != "other"


def is_error_message(text: str) -> bool:
    """True if an assistant message is one of our own API-failure messages
    (as opposed to a normal answer, or the 'nothing relevant found' message)."""
    return text.startswith("Sorry, the Gemini API is temporarily overloaded") or text.startswith(
        "Error calling the Gemini API:"
    )


def get_groq_api_key() -> str | None:
    key = None
    try:
        key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GROQ_API_KEY")
    if not key:
        key = st.session_state.get("manual_groq_key")
    return key


def stream_groq_answer(
    api_key: str,
    excerpt_text: str,
    chat_history: list[dict],
    question: str,
    helpful_examples: list[dict] | None,
):
    """Stream an answer from Groq's OpenAI-compatible chat API, reusing the
    same SYSTEM_PROMPT and excerpt/example format as the Gemini path so
    behavior (page citations, outside-knowledge labeling) stays consistent.
    Yields text deltas."""
    client = Groq(api_key=api_key)

    examples_block = ""
    if helpful_examples:
        rendered = "\n\n".join(
            f"Q: {e['question']}\nA: {e['answer']}" for e in helpful_examples
        )
        examples_block = (
            "Previously confirmed helpful (👍) answers to similar questions -- "
            "use these only as a style/consistency guide, the excerpts below "
            f"are still the source of truth:\n\n{rendered}\n\n"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in chat_history:
        role = "assistant" if turn["role"] == "assistant" else "user"
        messages.append({"role": role, "content": turn["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"{examples_block}Relevant excerpts from the document:\n\n{excerpt_text}\n\n"
            f"Question: {question}",
        }
    )

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=GROQ_MAX_OUTPUT_TOKENS,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


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


def save_feedback(
    client, session_id: str, question: str, answer: str, pages_used: list[int], rating: str
) -> None:
    """Store a 👍/👎 rating for a Q&A pair. This is the raw signal the
    'learning' feature below reuses -- since the Gemini API can't be
    fine-tuned per-user, we approximate learning by feeding well-rated
    past answers back in as in-context examples for similar future
    questions (see get_helpful_examples)."""
    if client is None:
        return
    try:
        client.table(SUPABASE_FEEDBACK_TABLE).insert(
            {
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "pages_used": pages_used,
                "rating": rating,
            }
        ).execute()
    except Exception:
        pass


@st.cache_data(ttl=120, show_spinner=False)
def _load_positive_feedback(_client) -> list[dict]:
    """Pull all 👍-rated Q&A pairs (cached for 2 minutes so every question
    doesn't re-hit Supabase). Underscore param tells st.cache_data not to
    try to hash the (unhashable) Supabase client."""
    if _client is None:
        return []
    try:
        res = (
            _client.table(SUPABASE_FEEDBACK_TABLE)
            .select("question, answer")
            .eq("rating", "up")
            .order("created_at", desc=True)
            .limit(300)
            .execute()
        )
        return res.data
    except Exception:
        return []


def get_helpful_examples(client, question: str, top_k: int = FEEDBACK_EXAMPLES_K) -> list[dict]:
    """Find past 👍-rated Q&A pairs whose QUESTION is most similar to the
    current one (simple TF-IDF over the small feedback set -- no need for
    the full document index here). Returns the ones worth reusing as
    examples, or [] if there's no feedback history yet / nothing close."""
    examples = _load_positive_feedback(client)
    if len(examples) < 1:
        return []
    try:
        past_questions = [e["question"] for e in examples]
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(past_questions + [question])
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        ranked = sims.argsort()[::-1][:top_k]
        return [examples[i] for i in ranked if sims[i] > 0.15]
    except Exception:
        return []


def build_contents(
    excerpt_text: str,
    chat_history: list[dict],
    question: str,
    helpful_examples: list[dict] | None = None,
) -> list[dict]:
    """Build the `contents` list for the API call. Unlike the whole-document
    approach, we send the (small) retrieved excerpt fresh with EVERY turn,
    since which excerpt is relevant changes per question."""
    contents = []

    for turn in chat_history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})

    examples_block = ""
    if helpful_examples:
        rendered = "\n\n".join(
            f"Q: {e['question']}\nA: {e['answer']}" for e in helpful_examples
        )
        examples_block = (
            "Previously confirmed helpful (👍) answers to similar questions -- "
            "use these only as a style/consistency guide, the excerpts below "
            f"are still the source of truth:\n\n{rendered}\n\n"
        )

    contents.append(
        {
            "role": "user",
            "parts": [
                {
                    "text": f"{examples_block}Relevant excerpts from the document:\n\n{excerpt_text}\n\n"
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
st.caption(
    "Answers are grounded in the document and cite page numbers. When the "
    "document doesn't cover something, I may add a clearly separate "
    "\"🌐 Outside information\" section from general knowledge -- always "
    "verify that part independently. Rate answers 👍/👎 so future similar "
    "questions can reuse what worked."
)


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

    groq_api_key = get_groq_api_key()
    with st.expander("⚡ Groq fallback (optional)", expanded=not groq_api_key):
        st.caption(
            "If Gemini hits a rate limit, the app automatically retries the "
            "same question through Groq (free, no card, much higher "
            "requests/minute) instead of just failing. Answers via fallback "
            "are clearly labeled, and Groq uses a different (open-weights) "
            "model, so citation precision may vary slightly."
        )
        if groq_api_key:
            st.success("Groq fallback is active.")
        else:
            st.markdown("Get a free key at [console.groq.com/keys](https://console.groq.com/keys).")
            manual_groq_key = st.text_input("Enter your Groq API key", type="password", key="groq_key_input")
            if manual_groq_key:
                st.session_state["manual_groq_key"] = manual_groq_key
                groq_api_key = manual_groq_key

    st.divider()
    st.session_state["fast_mode"] = st.toggle(
        "⚡ Fast mode",
        value=st.session_state.get("fast_mode", False),
        help=(
            f"Switches from {MODEL} to {FAST_MODEL}, Google's low-latency "
            "tier. Answers arrive noticeably faster; for a document this "
            "technical, keep it off when precision matters most."
        ),
    )

    st.divider()
    index = build_index()
    active_model_label = FAST_MODEL if st.session_state.get("fast_mode") else MODEL
    st.markdown(
        "**Model:** `{}`\n\n"
        "**Document:** {} pages, indexed into {} searchable chunks.".format(
            active_model_label, index["num_pages"], len(index["chunks"])
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
    forced_search_query = st.session_state.pop("force_search_query", None)
    search_query, was_corrected = correct_query(question, index["vocabulary"])
    if forced_search_query:
        search_query, was_corrected = forced_search_query, False
    else:
        st.session_state["last_search_query"] = search_query
    retrieved = retrieve_chunks(index, search_query)

    no_excerpts = not retrieved
    if no_excerpts:
        excerpt_text = (
            "(No document pages matched this question closely enough -- none "
            "are included. If you can answer this from general knowledge, do "
            "so under the '🌐 Outside information' heading as instructed; "
            "otherwise say plainly that the document doesn't seem to cover it.)"
        )
        pages_used = []
    else:
        excerpt_text = "\n\n---\n\n".join(c["text"] for c in retrieved)
        pages_used = sorted({p for c in retrieved for p in c["pages"]})

    helpful_examples = get_helpful_examples(supabase_client, question)

    active_model = FAST_MODEL if st.session_state.get("fast_mode") else MODEL
    client = genai.Client(api_key=api_key)
    contents = build_contents(
        excerpt_text, st.session_state["chat_history"], question, helpful_examples
    )

    with st.chat_message("assistant"):
        if was_corrected:
            st.caption(f"Also searched for: \"{search_query}\" (in case of a typo)")
        if no_excerpts:
            st.caption("No matching pages found -- answering from outside knowledge only, if possible.")
        else:
            est_tokens = sum(len(c["text"]) for c in retrieved) // CHARS_PER_TOKEN_ESTIMATE
            st.caption(
                f"Searched pages: {pages_used[0]}-{pages_used[-1]} "
                f"({len(retrieved)} excerpts, ~{est_tokens:,} tokens)"
            )
        if helpful_examples:
            st.caption(f"Reusing {len(helpful_examples)} previously 👍-rated similar answer(s) as a guide.")
        placeholder = st.empty()
        answer_text = ""
        last_error = None

        for attempt in range(1, MAX_API_RETRIES + 1):
            answer_text = ""
            was_truncated = False
            try:
                stream = client.models.generate_content_stream(
                    model=active_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                    ),
                )
                last_chunk = None
                for chunk in stream:
                    last_chunk = chunk
                    if chunk.text:
                        answer_text += chunk.text
                        placeholder.markdown(answer_text + "▌")
                # Detect truncation: the model was cut off mid-answer by the
                # max_output_tokens cap rather than finishing naturally.
                if last_chunk is not None and getattr(last_chunk, "candidates", None):
                    finish_reason = str(getattr(last_chunk.candidates[0], "finish_reason", ""))
                    was_truncated = "MAX_TOKENS" in finish_reason
                # Streaming is done -- clear the raw placeholder and render the
                # final answer properly (turns any ```mermaid``` / ```chart```
                # blocks into an actual diagram or bar chart).
                placeholder.empty()
                render_answer(answer_text)
                if was_truncated:
                    st.warning(
                        "⚠️ This answer was cut off (hit the length limit)."
                    )
                    cont_key = f"cont_{session_id}_{len(st.session_state['chat_history'])}"
                    if st.button("▶️ Continue this answer", key=cont_key):
                        st.session_state["retry_question"] = (
                            "Continue your previous answer exactly where it left off. "
                            "Do not repeat anything you already said, and do not "
                            "restart the explanation."
                        )
                        st.session_state["force_search_query"] = st.session_state.get(
                            "last_search_query", question
                        )
                        st.rerun()
                last_error = None
                break
            except Exception as e:
                last_error = e
                error_kind = classify_api_error(e)
                if attempt < MAX_API_RETRIES and error_kind != "other":
                    wait_seconds = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    if error_kind == "quota":
                        retry_msg = (
                            f"Rate/quota limit hit on this project (429) -- retrying in "
                            f"{wait_seconds}s (attempt {attempt}/{MAX_API_RETRIES})... "
                            "This means requests are going out faster than your tier "
                            "allows, not that Google is busy."
                        )
                    else:
                        retry_msg = (
                            f"Gemini's backend is temporarily overloaded -- retrying in "
                            f"{wait_seconds}s (attempt {attempt}/{MAX_API_RETRIES})..."
                        )
                    placeholder.info(retry_msg)
                    time.sleep(wait_seconds)
                else:
                    break

        # Gemini failed after retries -- automatically try Groq instead of
        # just giving up, if a Groq key is configured and the failure was
        # the kind a different provider could plausibly route around
        # (quota/overload/internal, not e.g. a bad Gemini API key).
        used_groq_fallback = False
        if last_error is not None and groq_api_key and classify_api_error(last_error) != "other":
            placeholder.info("Gemini is unavailable right now -- trying Groq instead...")
            groq_retrieved = retrieve_chunks(
                index, search_query, top_k=GROQ_TOP_K_CHUNKS, max_excerpt_tokens=GROQ_MAX_EXCERPT_TOKENS
            )
            groq_excerpt_text = (
                "\n\n---\n\n".join(c["text"] for c in groq_retrieved)
                if groq_retrieved
                else excerpt_text
            )
            for groq_attempt in range(1, GROQ_MAX_RETRIES + 1):
                answer_text = ""
                try:
                    for delta in stream_groq_answer(
                        groq_api_key, groq_excerpt_text, st.session_state["chat_history"], question, helpful_examples
                    ):
                        answer_text += delta
                        placeholder.markdown(answer_text + "▌")
                    placeholder.empty()
                    st.caption("⚡ Answered via Groq (Llama 3.3 70B) -- Gemini was rate-limited/unavailable.")
                    render_answer(answer_text)
                    if groq_retrieved:
                        pages_used = sorted({p for c in groq_retrieved for p in c["pages"]})
                    last_error = None
                    used_groq_fallback = True
                    break
                except Exception as groq_e:
                    last_error = groq_e
                    if groq_attempt < GROQ_MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF_SECONDS)

        if last_error is not None and not used_groq_fallback:
            error_kind = classify_api_error(last_error)
            if error_kind == "quota":
                answer_text = (
                    "Rate/quota limit: this project has hit its Gemini API request "
                    "or token limit (429 RESOURCE_EXHAUSTED) and didn't recover after "
                    "a few retries. Check the live limits for this project/key at "
                    "aistudio.google.com/rate-limit -- on the free tier this is often "
                    "the requests-per-minute cap (as low as 5-15 RPM depending on the "
                    "model), which a handful of quick questions in a row can trip."
                )
            elif error_kind in ("overload", "internal"):
                answer_text = (
                    "Sorry, the Gemini API backend is temporarily overloaded and didn't "
                    "respond after a few tries. This is usually short-lived and on "
                    "Google's side -- please try asking again in a moment."
                )
            else:
                answer_text = f"Error calling the Gemini API: {last_error}"
            with st.expander("Technical details"):
                st.code(f"{type(last_error).__name__}: {last_error}")
            placeholder.error(answer_text)

        if last_error is None and not is_error_message(answer_text):
            fb_col1, fb_col2, _ = st.columns([1, 1, 8])
            fb_key_base = f"fb_{session_id}_{len(st.session_state['chat_history'])}"
            with fb_col1:
                if st.button("👍", key=f"{fb_key_base}_up"):
                    save_feedback(supabase_client, session_id, question, answer_text, pages_used, "up")
                    st.toast("Thanks -- I'll reuse this as an example for similar questions.")
            with fb_col2:
                if st.button("👎", key=f"{fb_key_base}_down"):
                    save_feedback(supabase_client, session_id, question, answer_text, pages_used, "down")
                    st.toast("Thanks -- noted, this one won't be reused.")

    st.session_state["chat_history"].append({"role": "user", "content": question})
    st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})
    save_message_to_supabase(supabase_client, session_id, "user", question)
    save_message_to_supabase(supabase_client, session_id, "assistant", answer_text)
