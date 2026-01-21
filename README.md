# MarketingGuard

Multimodal audit helper for fund marketing posters. The app supports:

- Uploading posters (PNG/JPG) and OCR text files
- Rule-based and LLM-based compliance checks
- Hybrid rule recall (keyword + vector) via a RAG toggle
- Structured JSON outputs using Pydantic schemas
- Batch gallery browsing with confidence bars and audit tables

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment variables

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
export OPENAI_VLM_MODEL="gpt-4o-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export OPENAI_EXTRA_HEADERS='{"X-Custom-Header": "value"}'
```

## Notes

- If you only want to run the rule-only baseline, you can skip API keys.
- For batch audit, upload OCR text files with the same stem name as the image.
- Ground truth files should include `poster_name` and `rule_id` columns.
