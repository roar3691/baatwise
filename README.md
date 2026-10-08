# Baatwise

Baatwise is an early conversational AI prototype. It uses Claude for replies and selects a small set of earlier turns that are relevant to the current question, alongside the most recent turns. This is a lightweight lexical retrieval method; it is not a trained attention model, and it does not claim persistent memory.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-anthropic-api-key"
streamlit run app.py
```

Alternatively, create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

The secrets file is ignored by Git. Do not commit API keys. Anthropic API usage is billed separately from Claude subscriptions or any startup credits.

## Included

- Claude Sonnet 5.5 by default, with Claude Haiku 4.5 as a lower-cost choice.
- Session-only conversation history and a JSON transcript download.
- Lexical retrieval of relevant earlier chat turns, plus recent turns.
- Optional PDF text context (10 MB, 50 pages, and 18,000 extracted characters maximum). Scanned PDFs need OCR before upload.
- Optional Anthropic web search, off by default. Web search can add separate search and token charges; Anthropic currently prices its search tool separately from model tokens.
- English, Hindi, and Telugu reply preferences.

## Data and limitations

Chat and extracted PDF text remain in the Streamlit browser session and are sent to Anthropic when a prompt is submitted. They are not saved to a Baatwise database. Clearing the chat removes the session transcript. The PDF is held in session memory until removed or the session ends. Anthropic processes API requests under its own policies and account settings.

This prototype does not provide accounts, persistent storage, OCR, an independent search index, or production security controls. It is not evidence of incorporation, paying customers, production reliability, or validated results. Review the relevant service terms and privacy requirements before using real customer or sensitive data.

## Configuration

Set `ANTHROPIC_API_KEY` in the environment or Streamlit secrets. The app does not need MongoDB, Google Search credentials, or a Gemini key.
