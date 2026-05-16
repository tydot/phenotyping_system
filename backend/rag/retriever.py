from typing import Dict, List
import pandas as pd

from backend.rag.query_builder import (
    build_patient_query_terms,
    build_cluster_query_terms,
)


def parse_tags(tag_str: str) -> List[str]:
    if not isinstance(tag_str, str):
        return []

    separators = [",", ";", "|", "/", "，", "；"]
    tags = [tag_str]

    for sep in separators:
        new_tags = []
        for t in tags:
            new_tags.extend(str(t).split(sep))
        tags = new_tags

    cleaned = []
    seen = set()

    for t in tags:
        value = str(t).strip().lower()
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            cleaned.append(value)

    return cleaned


def normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_compatible_row(row: pd.Series) -> Dict:
    question_tags = safe_str(row.get("question_tags", ""))
    keywords = safe_str(row.get("keywords", ""))
    merged_tags = ";".join([x for x in [question_tags, keywords] if x])

    return {
        "chunk_id": safe_str(row.get("chunk_id", "")),
        "title": safe_str(row.get("title", "")) or safe_str(row.get("doc_title", "")),
        "source": safe_str(row.get("source", "")) or safe_str(row.get("source_filename", "")),
        "chunk_text": safe_str(row.get("chunk_text", "")),
        "tags": safe_str(row.get("tags", "")) or merged_tags,
        "use": safe_str(row.get("use", "")) or safe_str(row.get("subkb_type", "")),
        "page_target": safe_str(row.get("page_target", "")),
        "evidence_level": safe_str(row.get("evidence_level", "")),
        "retrieval_priority": safe_str(row.get("retrieval_priority", "")) or safe_str(row.get("priority", "")),
        "doc_id": safe_str(row.get("doc_id", "")),
        "doc_type": safe_str(row.get("doc_type", "")),
        "subkb_type": safe_str(row.get("subkb_type", "")),
        "topic_key": safe_str(row.get("topic_key", "")),
        "use_case": safe_str(row.get("use_case", "")),
        "linked_core_doc": safe_str(row.get("linked_core_doc", "")),
        "status": safe_str(row.get("status", "")),
        "review_notes": safe_str(row.get("review_notes", "")),
    }


def score_chunk(row: Dict, query_terms: List[str], page_type: str) -> Dict:
    text = normalize_text(row.get("chunk_text", ""))
    title = normalize_text(row.get("title", ""))
    tags = parse_tags(row.get("tags", ""))
    use = normalize_text(row.get("use", ""))
    subkb_type = normalize_text(row.get("subkb_type", ""))
    page_target = normalize_text(row.get("page_target", ""))
    use_case = normalize_text(row.get("use_case", ""))
    evidence = normalize_text(row.get("evidence_level", ""))
    priority = normalize_text(row.get("retrieval_priority", ""))

    score = 0.0
    matched_terms = []
    matched_tags = []

    for term in query_terms:
        term_l = normalize_text(term)
        if not term_l:
            continue

        # 文本命中
        if term_l in text:
            score += 2.0
            matched_terms.append(term)

        # 标题命中
        if term_l in title:
            score += 2.5
            matched_terms.append(term)

        # tag 精确命中
        if term_l in tags:
            score += 3.0
            matched_tags.append(term)
        else:
            for tg in tags:
                if term_l in tg or tg in term_l:
                    score += 2.5
                    matched_tags.append(tg)
                    break

    # 页面归属匹配
    page_fields = [use, subkb_type, page_target, use_case]
    if page_type:
        if any(page_type == x for x in page_fields if x):
            score += 3.0
        elif any(x in ["both", "shared", "general"] for x in page_fields if x):
            score += 0.8
        else:
            score -= 1.2

    # 证据等级
    if evidence in ["core", "high", "1", "a", "strong"]:
        score += 1.0
    elif evidence in ["supporting", "medium", "2", "b", "moderate"]:
        score += 0.5
    elif evidence in ["supplementary", "low", "3", "c", "weak"]:
        score += 0.2

    # 检索优先级
    if priority in ["high", "1", "p1", "top"]:
        score += 1.0
    elif priority in ["medium", "2", "p2"]:
        score += 0.5
    elif priority in ["low", "3", "p3"]:
        score += 0.2

    # starter 轻微降权
    status = normalize_text(row.get("status", ""))
    if status == "starter":
        score -= 0.3

    # 太短文本轻微降权
    if len(text) < 20:
        score -= 0.5

    return {
        "score": round(score, 4),
        "matched_terms": list(dict.fromkeys(matched_terms)),
        "matched_tags": list(dict.fromkeys(matched_tags)),
    }


def sort_results(results: List[Dict]) -> List[Dict]:
    return sorted(
        results,
        key=lambda x: (
            x["score"],
            str(x.get("retrieval_priority", "")).lower() in ["high", "1", "p1", "top"],
            str(x.get("evidence_level", "")).lower() in ["core", "high", "1", "a", "strong"],
        ),
        reverse=True,
    )


def _dedup_results(results: List[Dict]) -> List[Dict]:
    deduped = []
    seen = set()

    for item in results:
        key = (
            str(item.get("chunk_id", "")).strip().lower(),
            str(item.get("doc_id", "")).strip().lower(),
            str(item.get("title", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def _build_result_row(row: Dict, score_info: Dict) -> Dict:
    return {
        "chunk_id": row["chunk_id"],
        "title": row["title"],
        "source": row["source"],
        "chunk_text": row["chunk_text"],
        "score": score_info["score"],
        "evidence_level": row["evidence_level"],
        "retrieval_priority": row["retrieval_priority"],
        "matched_terms": score_info["matched_terms"],
        "matched_tags": score_info["matched_tags"],
        "doc_id": row["doc_id"],
        "doc_type": row["doc_type"],
        "subkb_type": row["subkb_type"],
        "topic_key": row["topic_key"],
        "use_case": row["use_case"],
        "page_target": row["page_target"],
        "linked_core_doc": row["linked_core_doc"],
        "status": row["status"],
        "review_notes": row["review_notes"],
    }


def retrieve_top_chunks_for_patient(
    df: pd.DataFrame,
    features: Dict[str, str],
    top_k: int = 5,
) -> List[Dict]:
    query_terms = build_patient_query_terms(features)
    results = []

    for _, raw_row in df.iterrows():
        row = build_compatible_row(raw_row)
        score_info = score_chunk(row, query_terms, page_type="patient")

        if score_info["score"] <= 0:
            continue

        results.append(_build_result_row(row, score_info))

    results = sort_results(results)
    results = _dedup_results(results)
    return results[:top_k]


def retrieve_top_chunks_for_cluster(
    df: pd.DataFrame,
    summary_features: List[str],
    top_k: int = 5,
) -> List[Dict]:
    query_terms = build_cluster_query_terms(summary_features)
    results = []

    for _, raw_row in df.iterrows():
        row = build_compatible_row(raw_row)
        score_info = score_chunk(row, query_terms, page_type="cluster")

        if score_info["score"] <= 0:
            continue

        results.append(_build_result_row(row, score_info))

    results = sort_results(results)
    results = _dedup_results(results)
    return results[:top_k]