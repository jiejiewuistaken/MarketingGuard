# MarketingGuard

Multimodal audit helper for fund marketing posters. The app supports:

- Uploading posters (PNG/JPG) and OCR text files
- LLM-based compliance checks with baseline/rag/advanced rag strategies
- Tagger-based metadata filtering with multi-query hard matching
- Optional VLM image description to capture visual risk cues
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
export OPENAI_EXTRA_HEADERS='{"X-Custom-Header": "value"}'
```

## Notes

- All strategies require API keys (baseline still calls the LLM).
- For batch audit, upload OCR text files with the same stem name as the image.
- Ground truth can use `poster_name`/`rule_id` or the Chinese headers listed below.

## Rule format (JSON)

The app accepts `.json` rules in the following format:

```json
{
  "id": "R001",
  "category": ["货币型基金", "通用"],
  "text": "不得承诺保本保收益",
  "trigger_keywords": ["保本", "稳赚", "无风险"]
}
```

## Convert TXT rules to JSON

```bash
python convert_rules.py --input "审核规则库/rules.txt" --output "审核规则库/rules_structured.json"
```

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

Metrics include precision/recall/F1 on (文件名 + 错误id), exact-match rate per file,
and field accuracies for 一级分类/二级分类/是否校验/合规规则名称 when present. Filenames
are normalized by stem and error_id values are normalized to digits (e.g. R334 = 334).
