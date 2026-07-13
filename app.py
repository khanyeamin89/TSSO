"""
Document Q&A over RPR.0534.1.0.BN.DZ0001 (Rooppur NPP Unit 1 Technical
Specification) using the Claude API, served as a Streamlit app.

How it works
------------
1. On startup, the full text of the document is loaded from document.txt
   (already extracted from the source PDF -- see extract_pdf.py).
2. The user types a question in a chat box.
3. The question + the ENTIRE document text is sent to the Claude API in a
   single request (Claude Sonnet 5 has a 1M-token context window, so the
   ~250K-token document comfortably fits alongside the question).
4. Claude's answer is streamed back and printed in the chat window.

Prompt caching is used on the document block so that if you ask several
questions in the same session, the (large, expensive) document text is only
processed once by Anthropic's servers -- follow-up questions in the same
session hit the cache and are much cheaper/faster.
"""

import os
import streamlit as st
from anthropic import Anthropic

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = "claude-sonnet-5"
DOCUMENT_PATH = os.path.join(os.path.dirname(__file__), "document.txt")
DOCUMENT_TITLE = "RPR.0534.1.0.BN.DZ0001 -- Technical Specification of Safe Operation of Rooppur NPP Unit 1 (Version 2)"

SYSTEM_PROMPT = f"""You are a careful technical assistant answering questions \
about a single reference document: {DOCUMENT_TITLE}.

Rules:
- Answer ONLY using the document text provided to you. Do not use outside
  knowledge about nuclear plants, Rooppur, or regulations in general.
- The document text is broken into pages with markers like "[PAGE 12]".
  When you answer, cite the page number(s) you drew the answer from, e.g.
  "(p. 37)" or "(pp. 40-41)".
- If the document does not contain the answer, say so plainly instead of
  guessing.
- Quote short phrases only when the exact wording matters (e.g. a limit
  value or a defined term); otherwise explain in your own words.
- Be precise with numbers, units, and parameter names -- this is a nuclear
  safety document and accuracy matters.
"""

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_document_text() -> str:
    """Load the full extracted document text once per server process."""
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_api_key() -> str | None:
    """Look for the API key in Streamlit secrets, then env vars, then the
    sidebar input the user can fill in manually."""
    key = None
    try:
        # st.secrets raises if no secrets.toml exists at all, so guard it.
        key = st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        key = st.session_state.get("manual_api_key")
    return key


def build_messages(document_text: str, chat_history: list[dict], question: str) -> list[dict]:
    """Build the messages array for the API call.

    The document is placed in the FIRST user turn as a cached block so that
    Anthropic's prompt cache can reuse it across turns in the same session.
    Later turns just carry the conversation (short) plus a reminder marker.
    """
    messages = []

    # First turn: attach the full document as a cached content block.
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the full text of the document, extracted "
                    f"page by page (page markers look like [PAGE N]):\n\n{document_text}",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": "Understood. I have the full document loaded and I'm "
            "ready to answer questions about it, citing page numbers.",
        }
    )

    # Replay prior Q&A turns (kept short -- just text, no need to re-cache).
    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # The new question.
    messages.append({"role": "user", "content": question})

    return messages


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Document Q&A -- RPR.0534.1.0.BN.DZ0001", page_icon="[Q&A]")
st.title("Document Q&A")
st.caption(DOCUMENT_TITLE)

# --- Sidebar: API key + info ---
with st.sidebar:
    st.header("Setup")
    api_key = get_api_key()
    if not api_key:
        st.warning("No ANTHROPIC_API_KEY found in secrets or environment.")
        manual_key = st.text_input("Enter your Anthropic API key", type="password")
        if manual_key:
            st.session_state["manual_api_key"] = manual_key
            api_key = manual_key
    else:
        st.success("API key loaded.")

    st.divider()
    st.markdown(
        "**Model:** `{}`\n\n"
        "**Document:** loaded from `document.txt` "
        "({:,} characters).".format(MODEL, len(load_document_text()))
    )
    st.caption(
        "The full document text is sent with every question. Prompt "
        "caching keeps repeat questions in the same session fast and cheap."
    )

    if st.button("Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()

# --- Main chat area ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of {"role": ..., "content": ...}

for turn in st.session_state["chat_history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about the document...")

if question:
    if not api_key:
        st.error("Please provide an Anthropic API key in the sidebar first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    document_text = load_document_text()
    client = Anthropic(api_key=api_key)

    messages = build_messages(document_text, st.session_state["chat_history"], question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_text = ""
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text_chunk in stream.text_stream:
                    answer_text += text_chunk
                    placeholder.markdown(answer_text + "▌")
            placeholder.markdown(answer_text)
        except Exception as e:
            answer_text = f"Error calling the Claude API: {e}"
            placeholder.error(answer_text)

    # Save turns to history (as simple strings, not the cached document block)
    st.session_state["chat_history"].append({"role": "user", "content": question})
    st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})
