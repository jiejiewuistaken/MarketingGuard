import argparse
import json
import os
from typing import Iterable, List, Optional

import pandas as pd

from audit_core import (
    ComplianceMetrics,
    ComplianceRow,
    coerce_compliance_rows,
    compute_compliance_metrics,
    explode_compliance_rows,
    model_dump_safe,
)


def load_rows(path: str) -> List[ComplianceRow]:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    records = df.to_dict(orient="records")
    return explode_compliance_rows(coerce_compliance_rows(records))


def load_all_filenames(path: str) -> List[str]:
    df = pd.read_csv(path, header=None)
    if df.empty:
        return []
    return [str(value).strip() for value in df.iloc[:, 0].tolist() if str(value).strip()]


def save_metrics(metrics: ComplianceMetrics, path: str) -> None:
    payload = model_dump_safe(metrics)
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return
    pd.DataFrame([payload]).to_csv(path, index=False)


def run_eval(
    predictions_path: str,
    ground_truth_path: str,
    strategy: str,
    latency_ms: float,
    avg_confidence: float,
    all_filenames: Optional[Iterable[str]] = None,
    similarity_threshold: float = 0.7,
    similarity_fields: Optional[Iterable[str]] = None,
) -> ComplianceMetrics:
    predicted_rows = load_rows(predictions_path)
    ground_truth_rows = load_rows(ground_truth_path)
    return compute_compliance_metrics(
        predicted_rows=predicted_rows,
        ground_truth_rows=ground_truth_rows,
        strategy=strategy,
        latency_ms=latency_ms,
        avg_confidence=avg_confidence,
        all_filenames=all_filenames,
        similarity_threshold=similarity_threshold,
        similarity_fields=tuple(similarity_fields) if similarity_fields else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate compliance predictions.")
    parser.add_argument("--predictions", required=True, help="Predictions CSV/XLSX")
    parser.add_argument("--ground-truth", required=True, help="Ground truth CSV/XLSX")
    parser.add_argument("--metrics-out", help="Write metrics JSON/CSV")
    parser.add_argument("--strategy", default="offline_eval")
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--avg-confidence", type=float, default=0.0)
    parser.add_argument(
        "--all-files",
        help="Optional CSV of filenames (kept for compatibility)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for match rate",
    )
    parser.add_argument(
        "--similarity-fields",
        help="Comma-separated fields (e.g. error_description,rule_name)",
    )
    args = parser.parse_args()

    all_filenames = None
    if args.all_files:
        all_filenames = load_all_filenames(args.all_files)

    similarity_fields = None
    if args.similarity_fields:
        similarity_fields = [
            field.strip() for field in args.similarity_fields.split(",") if field.strip()
        ]

    metrics = run_eval(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        strategy=args.strategy,
        latency_ms=args.latency_ms,
        avg_confidence=args.avg_confidence,
        all_filenames=all_filenames,
        similarity_threshold=args.similarity_threshold,
        similarity_fields=similarity_fields,
    )
    print(json.dumps(model_dump_safe(metrics), ensure_ascii=False, indent=2))

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        save_metrics(metrics, args.metrics_out)


if __name__ == "__main__":
    main()
