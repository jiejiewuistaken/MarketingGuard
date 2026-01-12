from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


AuditModality = Literal["text", "image", "text+image"]


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    text: str
    keywords: List[str] = Field(default_factory=list)


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = "UNKNOWN"
    rule_text: str = ""
    evidence: str
    modality: AuditModality = "text"
    confidence: float = 0.5


class AuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poster_name: str
    compliant: bool
    overall_confidence: float
    summary: str
    violations: List[Violation] = Field(default_factory=list)


class ComplianceRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    filename: str = Field(alias="文件名")
    level1: str = Field(alias="一级分类")
    level2: str = Field(alias="二级分类")
    error_id: str = Field(alias="错误id")
    error_description: str = Field(alias="错误描述")
    rule_name: str = Field(alias="合规规则名称")


class ComplianceTable(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    rows: List[ComplianceRow] = Field(default_factory=list, alias="结果列表")


class AuditMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    precision: float
    recall: float
    f1: float
    avg_confidence: float
    latency_ms: float


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def model_json_schema_safe(model_cls: type) -> Dict:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def model_validate_json_safe(model_cls: type, payload: str):
    if hasattr(model_cls, "model_validate_json"):
        return model_cls.model_validate_json(payload)
    return model_cls.parse_raw(payload)


def model_dump_safe(model: BaseModel) -> Dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_rules(raw_text: str) -> List[Rule]:
    lines = [line.strip() for line in raw_text.splitlines()]
    rules: List[Rule] = []
    current_id: Optional[str] = None
    current_chunks: List[str] = []

    def commit_rule(rule_id: Optional[str], chunks: List[str]) -> None:
        text = " ".join(chunk for chunk in chunks if chunk).strip()
        if not text:
            return
        if rule_id is None:
            rule_id = f"R{len(rules) + 1:03d}"
        rule = Rule(rule_id=rule_id, text=text)
        rule.keywords = extract_keywords(rule.text)
        rules.append(rule)

    for line in lines:
        if not line:
            continue
        match = re.match(r"^(\d{1,4})[.)、]\s*(.+)$", line)
        if match:
            commit_rule(current_id, current_chunks)
            current_id = f"R{int(match.group(1)):03d}"
            current_chunks = [match.group(2)]
        else:
            current_chunks.append(line)
    commit_rule(current_id, current_chunks)

    if not rules and raw_text.strip():
        for idx, chunk in enumerate(re.split(r"\n{2,}", raw_text.strip()), start=1):
            text = chunk.strip()
            if not text:
                continue
            rule = Rule(rule_id=f"R{idx:03d}", text=text)
            rule.keywords = extract_keywords(rule.text)
            rules.append(rule)

    return rules


def extract_keywords(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", text)
    stopwords = {
        "the",
        "and",
        "or",
        "for",
        "with",
        "must",
        "shall",
        "should",
        "not",
        "this",
        "that",
        "from",
        "into",
    }
    keywords = []
    for token in tokens:
        cleaned = token.lower().strip()
        if len(cleaned) < 2:
            continue
        if cleaned in stopwords:
            continue
        keywords.append(cleaned)
    return sorted(set(keywords))


def format_rules(rules: Sequence[Rule]) -> str:
    return "\n".join(f"{rule.rule_id}. {rule.text}" for rule in rules)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def find_keyword_evidence(text: str, keywords: Sequence[str], window: int = 40) -> Optional[str]:
    normalized = text or ""
    for keyword in keywords:
        if not keyword:
            continue
        idx = normalized.lower().find(keyword.lower())
        if idx == -1:
            continue
        start = max(0, idx - window)
        end = min(len(normalized), idx + len(keyword) + window)
        return normalized[start:end].strip()
    return None


def hash_embedding(text: str, dim: int = 256) -> List[float]:
    vector = [0.0] * dim
    if not text:
        return vector
    normalized = normalize_text(text)
    for idx in range(len(normalized) - 1):
        gram = normalized[idx : idx + 2]
        bucket = hash(gram) % dim
        vector[bucket] += 1.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


class EmbeddingProvider:
    def __init__(self, client: Optional[OpenAI], model: str):
        self.client = client
        self.model = model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if self.client is None:
            return [hash_embedding(text) for text in texts]
        try:
            response = self.client.embeddings.create(model=self.model, input=list(texts))
        except Exception:
            return [hash_embedding(text) for text in texts]
        embeddings = [item.embedding for item in response.data]
        return embeddings


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        total += x * y
        norm_a += x * x
        norm_b += y * y
    denom = (norm_a ** 0.5) * (norm_b ** 0.5) or 1.0
    return total / denom


class RuleIndex:
    def __init__(self, rules: Sequence[Rule], embedder: EmbeddingProvider, cache_path: Optional[str] = None):
        self.rules = list(rules)
        self.embedder = embedder
        self.cache_path = cache_path
        self.embeddings: List[List[float]] = []

    def _fingerprint(self) -> str:
        joined = "|".join(f"{rule.rule_id}:{rule.text}" for rule in self.rules)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def build(self) -> None:
        if self.cache_path:
            cached = self._load_cache()
            if cached:
                return
        self.embeddings = self.embedder.embed_texts([rule.text for rule in self.rules])
        if self.cache_path:
            self._save_cache()

    def _load_cache(self) -> bool:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return False
        if payload.get("fingerprint") != self._fingerprint():
            return False
        self.embeddings = payload.get("embeddings", [])
        return bool(self.embeddings)

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        dirpath = os.path.dirname(self.cache_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        payload = {"fingerprint": self._fingerprint(), "embeddings": self.embeddings}
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def search(self, query: str, top_k: int = 8) -> List[Tuple[Rule, float]]:
        if not self.embeddings:
            self.build()
        query_embedding = self.embedder.embed_texts([query])[0]
        scored: List[Tuple[Rule, float]] = []
        for rule, vector in zip(self.rules, self.embeddings):
            score = cosine_similarity(query_embedding, vector)
            scored.append((rule, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


def keyword_recall(rules: Sequence[Rule], query: str) -> List[Rule]:
    matches: List[Rule] = []
    normalized = normalize_text(query)
    for rule in rules:
        for keyword in rule.keywords:
            if keyword in normalized:
                matches.append(rule)
                break
    return matches


def hybrid_recall(
    rules: Sequence[Rule],
    rule_index: RuleIndex,
    query: str,
    keyword_top_k: int = 20,
    vector_top_k: int = 8,
) -> List[Rule]:
    keyword_hits = keyword_recall(rules, query)[:keyword_top_k]
    vector_hits = [rule for rule, _ in rule_index.search(query, top_k=vector_top_k)]
    merged: Dict[str, Rule] = {rule.rule_id: rule for rule in keyword_hits}
    for rule in vector_hits:
        merged.setdefault(rule.rule_id, rule)
    return list(merged.values())


def build_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("OPENAI_BASE_URL")
    if url:
        return OpenAI(api_key=key, base_url=url)
    return OpenAI(api_key=key)


def parse_extra_headers(raw_headers: Optional[str]) -> Dict[str, str]:
    if not raw_headers:
        return {}
    try:
        parsed = json.loads(raw_headers)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(value) for key, value in parsed.items()}


def encode_image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def call_openai_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: Optional[str],
    response_model: type,
    extra_headers: Optional[Dict[str, str]] = None,
):
    chat_messages = [{"role": "system", "content": system_prompt}]
    if image_data_url:
        chat_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        )
    else:
        chat_messages.append({"role": "user", "content": user_prompt})

    response_input = [{"role": "system", "content": system_prompt}]
    if image_data_url:
        response_input.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        )
    else:
        response_input.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }
        )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": model_json_schema_safe(response_model),
        },
    }

    if hasattr(client, "responses") and hasattr(client.responses, "create"):
        try:
            response = client.responses.create(
                model=model,
                input=response_input,
                response_format=response_format,
                extra_headers=extra_headers,
            )
            payload = getattr(response, "output_text", "") or ""
            if not payload and getattr(response, "output", None):
                chunks: List[str] = []
                for output in response.output:
                    content = getattr(output, "content", None)
                    if not content:
                        continue
                    for item in content:
                        text = getattr(item, "text", None)
                        if text:
                            chunks.append(text)
                payload = "\n".join(chunks)
            if payload:
                return model_validate_json_safe(response_model, payload)
        except Exception:
            pass

    if hasattr(client.chat.completions, "parse"):
        try:
            kwargs = {
                "model": model,
                "messages": chat_messages,
                "response_format": response_model,
            }
            if extra_headers:
                kwargs["extra_headers"] = extra_headers
            response = client.chat.completions.parse(**kwargs)
            parsed = getattr(response.choices[0].message, "parsed", None)
            if parsed is not None:
                return parsed
            payload = response.choices[0].message.content or ""
            if payload:
                return model_validate_json_safe(response_model, payload)
        except Exception:
            pass

    response = client.chat.completions.create(
        model=model,
        messages=chat_messages,
        response_format=response_format,
        extra_headers=extra_headers,
    )
    payload = response.choices[0].message.content
    return model_validate_json_safe(response_model, payload)


def build_audit_prompts(
    poster_name: str,
    ocr_text: str,
    rules: Sequence[Rule],
    include_rules: bool,
) -> Tuple[str, str]:
    system_prompt = (
        "You are a compliance auditor for fund marketing materials. "
        "Use the provided rules to identify violations. "
        "Output JSON that strictly matches the schema. "
        "Keep evidence concise and directly quoted when possible."
    )
    rules_block = format_rules(rules) if include_rules else "No rules provided."
    user_prompt = (
        f"Poster name: {poster_name}\n\n"
        f"OCR text:\n{ocr_text or '[EMPTY]'}\n\n"
        f"Rules:\n{rules_block}\n\n"
        "Task:\n"
        "- List every violation found in the poster.\n"
        "- Each violation must cite a rule_id and include evidence.\n"
        "- If no violations, return an empty list and compliant=true.\n"
        "- Provide an overall_confidence between 0 and 1.\n"
        "- Use modality=text, image, or text+image.\n"
    )
    return system_prompt, user_prompt


def normalize_audit_result(result: AuditResult, fallback_name: str) -> AuditResult:
    if not result.poster_name:
        result.poster_name = fallback_name
    result.compliant = len(result.violations) == 0
    for violation in result.violations:
        violation.confidence = clamp(float(violation.confidence))
    if result.violations:
        avg_conf = sum(v.confidence for v in result.violations) / len(result.violations)
        result.overall_confidence = clamp(avg_conf)
    else:
        result.overall_confidence = clamp(max(result.overall_confidence or 0.0, 0.8))
    return result


def audit_rule_only(poster_name: str, ocr_text: str, rules: Sequence[Rule]) -> AuditResult:
    violations: List[Violation] = []
    for rule in rules:
        evidence = find_keyword_evidence(ocr_text, rule.keywords)
        if not evidence:
            continue
        violations.append(
            Violation(
                rule_id=rule.rule_id,
                rule_text=rule.text,
                evidence=evidence,
                modality="text",
                confidence=0.55,
            )
        )
    result = AuditResult(
        poster_name=poster_name,
        compliant=len(violations) == 0,
        overall_confidence=0.7 if violations else 0.85,
        summary="Rule-only baseline.",
        violations=violations,
    )
    return normalize_audit_result(result, poster_name)


def map_violation_to_rule(
    violation_text: str,
    rules: Sequence[Rule],
    rule_index: Optional[RuleIndex] = None,
) -> Optional[Rule]:
    if not rules:
        return None
    if rule_index:
        scored = rule_index.search(violation_text, top_k=1)
        return scored[0][0] if scored else None
    normalized = normalize_text(violation_text)
    for rule in rules:
        for keyword in rule.keywords:
            if keyword in normalized:
                return rule
    return rules[0]


def audit_with_llm(
    poster_name: str,
    ocr_text: str,
    rules: Sequence[Rule],
    client: OpenAI,
    model: str,
    extra_headers: Optional[Dict[str, str]],
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/png",
    include_rules: bool = True,
) -> AuditResult:
    system_prompt, user_prompt = build_audit_prompts(poster_name, ocr_text, rules, include_rules)
    image_data_url = None
    if image_bytes:
        image_data_url = encode_image_to_data_url(image_bytes, image_mime)
    result = call_openai_json(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_data_url=image_data_url,
        response_model=AuditResult,
        extra_headers=extra_headers,
    )
    return normalize_audit_result(result, poster_name)


def fill_rule_references(
    result: AuditResult,
    rules: Sequence[Rule],
    rule_index: Optional[RuleIndex],
) -> AuditResult:
    for violation in result.violations:
        if violation.rule_id and violation.rule_id != "UNKNOWN":
            continue
        mapping = map_violation_to_rule(
            f"{violation.evidence} {violation.rule_text}",
            rules=rules,
            rule_index=rule_index,
        )
        if mapping:
            violation.rule_id = mapping.rule_id
            if not violation.rule_text:
                violation.rule_text = mapping.text
    return normalize_audit_result(result, result.poster_name)


def run_audit(
    poster_name: str,
    image_bytes: Optional[bytes],
    image_mime: str,
    ocr_text: str,
    rules: Sequence[Rule],
    client: Optional[OpenAI],
    model: str,
    extra_headers: Optional[Dict[str, str]],
    strategy: Literal["rule_only", "llm_only", "rule_vlm"],
    use_rag: bool,
    rule_index: Optional[RuleIndex],
    query_hint: Optional[str] = None,
) -> AuditResult:
    text_query = " ".join([ocr_text or "", query_hint or ""]).strip()
    if use_rag and rule_index:
        candidate_rules = hybrid_recall(rules, rule_index, text_query)
    else:
        candidate_rules = list(rules)

    if strategy == "rule_only":
        return audit_rule_only(poster_name, ocr_text, candidate_rules)

    if client is None:
        raise RuntimeError("OpenAI client is required for LLM strategies.")

    include_rules = strategy == "rule_vlm"
    result = audit_with_llm(
        poster_name=poster_name,
        ocr_text=ocr_text,
        rules=candidate_rules if include_rules else [],
        client=client,
        model=model,
        extra_headers=extra_headers,
        image_bytes=image_bytes,
        image_mime=image_mime,
        include_rules=include_rules,
    )
    if not include_rules:
        result = fill_rule_references(result, rules, rule_index)
    return result


def compute_metrics(
    predictions: Sequence[AuditResult],
    ground_truth: Dict[str, List[str]],
    strategy: str,
    latency_ms: float,
) -> AuditMetrics:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    confidences: List[float] = []
    for prediction in predictions:
        predicted = {v.rule_id for v in prediction.violations}
        truth = set(ground_truth.get(prediction.poster_name, []))
        total_tp += len(predicted & truth)
        total_fp += len(predicted - truth)
        total_fn += len(truth - predicted)
        confidences.append(prediction.overall_confidence)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return AuditMetrics(
        strategy=strategy,
        precision=precision,
        recall=recall,
        f1=f1,
        avg_confidence=avg_conf,
        latency_ms=latency_ms,
    )
