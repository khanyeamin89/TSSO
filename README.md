# Document Q&A -- RPR.0534.1.0.BN.DZ0001

A small Streamlit app that lets you ask questions about the Rooppur NPP
Unit 1 Technical Specification (RPR.0534.1.0.BN.DZ0001, Version 2) using
the Claude API. The full document text (459 pages) is sent to Claude with
every question, and Claude answers with page citations.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (chat UI + Claude API calls) |
| `document.txt` | Pre-extracted text of the PDF, with `[PAGE N]` markers |
| `extract_pdf.py` | Re-run this if you want to swap in a different/updated PDF |
| `requirements.txt` | Python dependencies |
| `.streamlit/secrets.toml.example` | Template for your API key |

## 1. Run it locally

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt

# provide your API key one of two ways:
export ANTHROPIC_API_KEY="sk-ant-..."
# OR: cp .streamlit/secrets.toml.example .streamlit/secrets.toml  (and edit it)

streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## 2. Deploy on Streamlit Community Cloud (via GitHub)

1. Push this folder to a GitHub repo (make sure `document.txt` is included --
   it's ~0.8 MB of plain text, well within GitHub's limits).
   **Do not** commit a real `secrets.toml` -- keep it out of the repo
   (it's already excluded via the `.example` suffix).
2. Go to https://share.streamlit.io, sign in, and click "New app".
3. Point it at your repo/branch and set the main file path to `app.py`.
4. In the app's **Settings -> Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-real-key"
   ```
5. Deploy. The app will install `requirements.txt` automatically.

## 3. Using a different PDF

Replace the source PDF and regenerate `document.txt`:

```bash
python extract_pdf.py path/to/new_document.pdf --out document.txt
```

Then update `DOCUMENT_TITLE` in `app.py` to match, and re-deploy.

## Notes on cost and context

- The app uses `claude-sonnet-5`, which has a 1M-token context window --
  large enough to hold the entire ~250K-token document alongside your
  question in a single request.
- The document is sent as a **prompt-cached** block, so if you ask several
  questions in a row, Anthropic only charges full price for processing the
  document once; follow-up questions in the same session reuse the cache
  (cache lifetime is a few minutes of inactivity).
- If you swap in a much larger PDF and it no longer fits in context, you'd
  need to switch to a retrieval-based (chunking) approach instead of
  sending the whole document every time -- ask if you'd like that version.
