from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from audit_core import (
    AuditResult,
    ComplianceMetrics,
    ComplianceRow,
    build_openai_client,
    coerce_compliance_rows,
    compliance_rows_from_audit_results,
    compliance_rows_to_records,
    compute_compliance_metrics,
    explode_compliance_rows,
    model_dump_safe,
    parse_extra_headers,
    parse_rules,
    run_audit,
)
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="MarketingGuard", layout="wide")


def bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def load_ocr_texts(files: List) -> Dict[str, str]:
    ocr_map: Dict[str, str] = {}
    for file in files:
        stem = Path(file.name).stem
        text = file.getvalue().decode("utf-8", errors="ignore")
        ocr_map[stem] = text
    return ocr_map


def load_ground_truth_rows(file) -> List[ComplianceRow]:
    if file is None:
        return []
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    rows = explode_compliance_rows(coerce_compliance_rows(df.to_dict(orient="records")))
    if not any(row.filename and row.error_id for row in rows):
        st.warning(
            "Ground truth file must include 文件名/错误id (or poster_name/rule_id) columns."
        )
    return rows


def render_gallery(image_files: List, selected_name: str) -> None:
    items = []
    for file in image_files:
        image_bytes = file.getvalue()
        encoded = bytes_to_base64(image_bytes)
        border = "2px solid #2563eb" if file.name == selected_name else "1px solid #e2e8f0"
        items.append(
            f"""
            <div style="flex: 0 0 auto; text-align: center;">
                <img src="data:{file.type};base64,{encoded}"
                     style="height: 120px; border-radius: 8px; border: {border};" />
                <div style="font-size: 12px; margin-top: 4px; color: #475467;">
                    {file.name}
                </div>
            </div>
            """
        )
    html = f"""
    <div style="display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px;">
        {''.join(items)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_confidence_bars(results: Dict[str, AuditResult]) -> None:
    st.markdown("**Poster confidence bars**")
    for name, result in results.items():
        st.caption(name)
        score = int(result.overall_confidence * 100)
        bar_col, value_col = st.columns([6, 1])
        with bar_col:
            st.progress(score)
        with value_col:
            st.markdown(f"**{score}%**")


st.title("MarketingGuard Audit Assistant")
st.caption("Multimodal audit helper for efund marketing posters.")

with st.sidebar:
    st.header("Config")
    api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    base_url = st.text_input("OPENAI_BASE_URL", value=os.getenv("OPENAI_BASE_URL", ""))
    extra_headers_raw = st.text_area(
        "OPENAI_EXTRA_HEADERS (JSON)",
        value=os.getenv("OPENAI_EXTRA_HEADERS", ""),
        height=100,
    )
    vlm_model = st.text_input("VLM model", value=os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini"))
    strategy_labels = {
        "baseline": "baseline（全量规则入prompt）",
        "rag": "rag（Tagger + metadata过滤）",
        "advanced_rag": "advanced rag（Tagger + 多query硬匹配）",
    }
    strategy = st.selectbox(
        "Strategy",
        list(strategy_labels.keys()),
        index=0,
        format_func=strategy_labels.get,
    )

    st.divider()
    st.subheader("Rules")
    rules_file = st.file_uploader("Upload rules (.txt/.md/.json)", type=["txt", "md", "json"])
    rules_text = st.text_area(
        "Or paste rules here",
        value="Rule 1. Do not promise returns or guarantee principal.",
        height=140,
    )

    if rules_file is not None:
        rules_text = rules_file.getvalue().decode("utf-8", errors="ignore")

st.header("上传待审核海报和OCR文本")
image_files = st.file_uploader(
    "上传待审核海报 (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)
ocr_files = st.file_uploader(
    "上传OCR文本 (MD/TXT)", type=["md", "txt"], accept_multiple_files=True
)
run_button = st.button("Run audit")

if run_button:
    if not image_files:
        st.error("Please upload at least one poster image.")
        st.stop()

# 解析规则文本
    parsed_rules = parse_rules(rules_text)
    if not parsed_rules:
        st.error("Rules are empty. Provide at least one rule.")
        st.stop()

    extra_headers = parse_extra_headers(extra_headers_raw)
# 初始化openai客户端
    if not api_key:
        st.error("OPENAI_API_KEY is required for LLM strategies.")
        st.stop()
    client = build_openai_client(api_key=api_key, base_url=base_url)
    ocr_map = load_ocr_texts(ocr_files)

    results: Dict[str, AuditResult] = {}
    latency_map: Dict[str, float] = {}
    for image_file in image_files:
        poster_name = image_file.name
        ocr_text = ocr_map.get(Path(poster_name).stem, "")
        start = time.time()
        result = run_audit(
            poster_name=poster_name,
            image_bytes=image_file.getvalue(),
            image_mime=image_file.type,
            ocr_text=ocr_text,
            rules=parsed_rules,
            client=client,
            model=vlm_model,
            extra_headers=extra_headers,
            strategy=strategy,
            rule_index=None,
            query_hint=None,
        )
        latency_map[poster_name] = (time.time() - start) * 1000
        results[poster_name] = result

    st.session_state["audit_results"] = results
    st.session_state["ocr_map"] = ocr_map
    st.session_state["latency_map"] = latency_map

if "audit_results" in st.session_state and image_files:
    results = st.session_state["audit_results"]
    ocr_map = st.session_state.get("ocr_map", {})
    poster_names = [file.name for file in image_files]
    if len(poster_names) > 1:
        selected_index = st.slider("Poster index", 0, len(poster_names) - 1, 0)
    else:
        selected_index = 0
    selected_name = poster_names[selected_index]

    render_gallery(image_files, selected_name)

    left_col, right_col = st.columns([1.35, 1])
    with left_col:
        st.subheader("Poster preview")
        selected_file = next(file for file in image_files if file.name == selected_name)
        st.image(selected_file.getvalue(), caption=selected_name, use_column_width=True)
        if selected_name in results:
            st.caption("Overall confidence")
            st.progress(int(results[selected_name].overall_confidence * 100))

    with right_col:
        st.subheader("OCR text")
        ocr_text = ocr_map.get(Path(selected_name).stem, "")
        st.text_area("OCR output", value=ocr_text, height=240, key=f"ocr_{selected_name}")

        result = results.get(selected_name)
        if result is not None:
            st.subheader("Image description")
            st.text_area(
                "VLM description",
                value=result.image_description or "",
                height=140,
                key=f"desc_{selected_name}",
            )
            tags = "，".join(result.poster_tags) if result.poster_tags else "未识别"
            st.caption(f"Detected tags: {tags}")

        st.subheader("Audit results")
        if result is None or not result.violations:
            st.info("No violations detected.")
        else:
            df = pd.DataFrame([model_dump_safe(v) for v in result.violations])
            st.dataframe(df, use_container_width=True)
        if result is not None and result.audit_steps:
            with st.expander("Audit steps (summary)"):
                st.write("\n".join(result.audit_steps))

    st.divider()
    render_confidence_bars(results)

    st.divider()
    st.subheader("Export predictions")
    pred_rows = compliance_rows_from_audit_results(results.values(), normalize_ids=True)
    if pred_rows:
        pred_records = compliance_rows_to_records(pred_rows, by_alias=True)
        pred_df = pd.DataFrame(pred_records)
        csv_data = pred_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download predictions CSV",
            csv_data,
            file_name="predictions.csv",
            mime="text/csv",
        )
        st.caption("Columns: 文件名, 一级分类, 二级分类, 是否校验, 错误id, 错误描述, 合规规则名称.")
    else:
        st.info("No violations to export.")

    st.divider()
    st.subheader("Comparison runner")
    st.caption("Run baseline comparisons: baseline, rag, and advanced rag.")
    gt_file = st.file_uploader("Upload ground truth (CSV/XLSX)", type=["csv", "xlsx"])
    confirm_run = st.checkbox("Confirm running comparison")
    if st.button("Run comparison") and confirm_run:
        if not api_key:
            st.error("OPENAI_API_KEY is required for LLM comparisons.")
            st.stop()
        ground_truth_rows = load_ground_truth_rows(gt_file)
        if not ground_truth_rows:
            st.warning("Ground truth missing or invalid. Metrics will be zero.")
        comparison_rules = parse_rules(rules_text)
        extra_headers = parse_extra_headers(extra_headers_raw)
        comparison_client = build_openai_client(api_key=api_key, base_url=base_url)

        metrics: List[ComplianceMetrics] = []
        for mode in ["baseline", "rag", "advanced_rag"]:
            start = time.time()
            predictions: List[AuditResult] = []
            for image_file in image_files:
                poster_name = image_file.name
                ocr_text = ocr_map.get(Path(poster_name).stem, "")
                predictions.append(
                    run_audit(
                        poster_name=poster_name,
                        image_bytes=image_file.getvalue(),
                        image_mime=image_file.type,
                        ocr_text=ocr_text,
                        rules=comparison_rules,
                        client=comparison_client,
                        model=vlm_model,
                        extra_headers=extra_headers,
                        strategy=mode,
                        rule_index=None,
                        query_hint=None,
                    )
                )
            latency_ms = (time.time() - start) * 1000
            avg_conf = (
                sum(p.overall_confidence for p in predictions) / len(predictions)
                if predictions
                else 0.0
            )
            pred_rows = compliance_rows_from_audit_results(predictions, normalize_ids=True)
            metrics.append(
                compute_compliance_metrics(
                    predicted_rows=pred_rows,
                    ground_truth_rows=ground_truth_rows,
                    strategy=mode,
                    latency_ms=latency_ms,
                    avg_confidence=avg_conf,
                    all_filenames=poster_names,
                )
            )
        metrics_df = pd.DataFrame([model_dump_safe(m) for m in metrics])
        st.dataframe(metrics_df, use_container_width=True)