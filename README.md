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
- Ground truth can use `poster_name`/`rule_id` or the Chinese headers listed below.

## Evaluation / test plan

Ground truth CSV/XLSX columns (Chinese headers):

- 文件名
- 一级分类
- 二级分类
- 是否校验
- 错误id
- 错误描述
- 合规规则名称

You can export predictions from the Streamlit app via **Export predictions**. The
downloaded CSV uses the same headers so it can be compared directly.

Run offline evaluation:

```bash
python evaluate_compliance.py \
  --predictions predictions.csv \
  --ground-truth ground_truth.xlsx \
  --metrics-out metrics.json
```

Metrics are similarity-based (LLM generation friendly):

- avg_gt_similarity: avg best similarity per ground-truth row
- avg_pred_similarity: avg best similarity per predicted row
- similarity_f1: harmonic mean of the two
- gt_match_rate / pred_match_rate: fraction above threshold (default 0.7)

Similarity uses `错误描述 + 合规规则名称` by default. Filenames are normalized by
stem. You can customize fields/threshold via CLI:

```bash
python evaluate_compliance.py \
  --predictions predictions.csv \
  --ground-truth ground_truth.xlsx \
  --similarity-threshold 0.75 \
  --similarity-fields error_description,rule_name,level1
```
