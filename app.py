"""
Document Q&A over RPR.0534.1.0.BN.DZ0001 (Rooppur NPP Unit 1 Technical
Specification) using the Google Gemini API (free tier), served as a
Streamlit app.

How it works
------------
1. On startup, the full text of the document is loaded from document.txt
   (already extracted from the source PDF -- see extract_pdf.py).
2. The user types a question in a chat box.
3. The question + the ENTIRE document text is sent to the Gemini API in a
   single request (Gemini 2.5 Flash has a 1M-token context window, so the
   ~250K-token document comfortably fits alongside the question).
4. Gemini's answer is streamed back and printed in the chat window.

Why Gemini: Google AI Studio issues a genuinely free API key (no credit
card) with a workable daily quota for the Flash models -- unlike the
Anthropic/OpenAI APIs, which are pay-as-you-go from the first request.
Get a free key at https://aistudio.google.com/app/apikey
"""

import os
import streamlit as st
from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = "gemini-3.5-flash"  # current GA model, free-tier eligible, 1M-token context window
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
        key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GEMINI_API_KEY")
    if not key:
        key = st.session_state.get("manual_api_key")
    return key


def build_contents(document_text: str, chat_history: list[dict], question: str) -> list[dict]:
    """Build the `contents` list for the API call in Gemini's format:
    a list of {"role": "user"|"model", "parts": [{"text": ...}]} turns.
    The document is placed in the FIRST user turn.
    """
    contents = []

    contents.append(
        {
            "role": "user",
            "parts": [
                {
                    "text": f"Here is the full text of the document, extracted "
                    f"page by page (page markers look like [PAGE N]):\n\n{document_text}"
                }
            ],
        }
    )
    contents.append(
        {
            "role": "model",
            "parts": [
                {
                    "text": "Understood. I have the full document loaded and I'm "
                    "ready to answer questions about it, citing page numbers."
                }
            ],
        }
    )

    for turn in chat_history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})

    contents.append({"role": "user", "parts": [{"text": question}]})

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
    st.markdown(
        "**Model:** `{}`\n\n"
        "**Document:** loaded from `document.txt` "
        "({:,} characters).".format(MODEL, len(load_document_text()))
    )
    st.caption(
        "The full document text is sent with every question. Free tier "
        "daily quotas apply -- see Google AI Studio for current limits."
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

    document_text = load_document_text()
    client = genai.Client(api_key=api_key)

    contents = build_contents(document_text, st.session_state["chat_history"], question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_text = ""
        try:
            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=2000,
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
