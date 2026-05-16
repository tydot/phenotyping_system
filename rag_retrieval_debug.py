from pprint import pprint
import pandas as pd

from backend.rag.retriever import (
    retrieve_top_chunks_for_patient,
    retrieve_top_chunks_for_cluster,
)
from backend.rag.query_builder import (
    build_patient_query_terms,
    build_cluster_query_terms,
)


EXCEL_PATH = "data/ARM_RAG_Final_Master_Table_merged_8papers_plus3chunks_tuned_plus2chunks.xlsx"
SHEET_NAME = "Master_SubKB_Final"


def print_header(title: str):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def debug_patient(df, case_name, features, top_k=5):
    print_header(f"RETRIEVAL PATIENT: {case_name}")
    print("features:")
    pprint(features)
    print("query_terms:")
    pprint(build_patient_query_terms(features))

    results = retrieve_top_chunks_for_patient(df, features, top_k=top_k)
    for i, item in enumerate(results, start=1):
        print(f"\n--- Rank {i} ---")
        pprint({
            "chunk_id": item["chunk_id"],
            "doc_id": item["doc_id"],
            "title": item["title"],
            "score": item["score"],
            "matched_terms": item["matched_terms"],
            "matched_tags": item["matched_tags"],
            "evidence_level": item["evidence_level"],
            "retrieval_priority": item["retrieval_priority"],
            "subkb_type": item["subkb_type"],
            "use_case": item["use_case"],
        })


def debug_cluster(df, case_name, summary_features, top_k=5):
    print_header(f"RETRIEVAL CLUSTER: {case_name}")
    print("summary_features:")
    pprint(summary_features)
    print("query_terms:")
    pprint(build_cluster_query_terms(summary_features))

    results = retrieve_top_chunks_for_cluster(df, summary_features, top_k=top_k)
    for i, item in enumerate(results, start=1):
        print(f"\n--- Rank {i} ---")
        pprint({
            "chunk_id": item["chunk_id"],
            "doc_id": item["doc_id"],
            "title": item["title"],
            "score": item["score"],
            "matched_terms": item["matched_terms"],
            "matched_tags": item["matched_tags"],
            "evidence_level": item["evidence_level"],
            "retrieval_priority": item["retrieval_priority"],
            "subkb_type": item["subkb_type"],
            "use_case": item["use_case"],
        })


def debug_free_query(df, query, top_k=5):
    print_header(f"FREE QUERY KEYWORD DEBUG: {query}")

    q = str(query).strip().lower()
    rows = []

    for _, row in df.iterrows():
        doc_id = str(row.get("doc_id", ""))
        doc_title = str(row.get("doc_title", ""))
        question_tags = str(row.get("question_tags", ""))
        keywords = str(row.get("keywords", ""))
        chunk_text = str(row.get("chunk_text", ""))
        subkb_type = str(row.get("subkb_type", ""))
        priority = str(row.get("priority", ""))
        evidence = str(row.get("evidence_level", ""))

        score = 0.0
        if q in doc_title.lower():
            score += 2.0
        if q in question_tags.lower():
            score += 2.5
        if q in keywords.lower():
            score += 2.5
        if q in chunk_text.lower():
            score += 1.5

        if score > 0:
            rows.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "subkb_type": subkb_type,
                "priority": priority,
                "evidence_level": evidence,
                "score": round(score, 4),
            })

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)[:top_k]
    pprint(rows)


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME).fillna("")

    # =========================
    # PATIENT TESTS
    # =========================
    debug_patient(
        df,
        "low_pressure_poor_propulsion_hyposensation",
        {
            "resting_pressure": "low",
            "msp": "low",
            "defecatory_rectal_pressure": "low",
            "rectal_sensation": "high_threshold",
            "rair": "present",
        },
    )

    debug_patient(
        df,
        "absent_rair_case",
        {
            "resting_pressure": "normal",
            "msp": "normal",
            "defecatory_rectal_pressure": "normal",
            "rectal_sensation": "normal",
            "rair": "absent",
        },
    )

    debug_patient(
        df,
        "high_resting_pressure_low_propulsion",
        {
            "resting_pressure": "high",
            "msp": "normal",
            "defecatory_rectal_pressure": "low",
            "rectal_sensation": "normal",
            "rair": "present",
        },
    )

    debug_patient(
        df,
        "rectal_hyposensitivity_case",
        {
            "resting_pressure": "normal",
            "msp": "normal",
            "defecatory_rectal_pressure": "normal",
            "rectal_sensation": "high_threshold",
            "rair": "present",
        },
    )

    debug_patient(
        df,
        "low_squeeze_pressure_case",
        {
            "resting_pressure": "normal",
            "msp": "low",
            "defecatory_rectal_pressure": "normal",
            "rectal_sensation": "normal",
            "rair": "present",
        },
    )

    # =========================
    # CLUSTER TESTS
    # =========================
    debug_cluster(
        df,
        "cluster_dd_biofeedback",
        [
            "dyssynergia",
            "poor propulsion",
            "biofeedback",
        ],
    )

    debug_cluster(
        df,
        "cluster_rectal_sensation",
        [
            "rectal sensation",
            "hyposensitivity",
            "RAIR",
            "classification",
        ],
    )

    debug_cluster(
        df,
        "cluster_sensory_dysfunction",
        [
            "phenotype",
            "sensory dysfunction",
            "hyposensitivity",
            "pathophysiology",
        ],
    )

    # =========================
    # FREE QUERY DEBUG
    # =========================
    def debug_free_query(df, query, top_k=5):
        print_header(f"FREE QUERY KEYWORD DEBUG: {query}")

        q = str(query).strip().lower()
        if not q:
            print([])
            return

        stopwords = {
            "why", "what", "how", "is", "are", "the", "a", "an", "of", "and", "or",
            "to", "for", "in", "on", "by", "with", "between", "relationship",
            "can", "be", "not", "do", "does", "did", "first", "line"
        }

        raw_terms = q.replace("/", " ").replace("-", " ").split()
        query_terms = [t.strip() for t in raw_terms if t.strip() and t.strip() not in stopwords]

        print("query_terms:")
        pprint(query_terms)

        rows = []

        for _, row in df.iterrows():
            doc_id = str(row.get("doc_id", ""))
            doc_title = str(row.get("doc_title", ""))
            question_tags = str(row.get("question_tags", ""))
            keywords = str(row.get("keywords", ""))
            chunk_text = str(row.get("chunk_text", ""))
            subkb_type = str(row.get("subkb_type", ""))
            priority = str(row.get("priority", ""))
            evidence = str(row.get("evidence_level", ""))
            use_case = str(row.get("use_case", ""))

            doc_title_l = doc_title.lower()
            question_tags_l = question_tags.lower()
            keywords_l = keywords.lower()
            chunk_text_l = chunk_text.lower()

            score = 0.0
            matched_terms = []

            for term in query_terms:
                hit = False

                if term in doc_title_l:
                    score += 2.0
                    hit = True

                if term in question_tags_l or term in keywords_l:
                    score += 2.5
                    hit = True

                if term in chunk_text_l:
                    score += 1.5
                    hit = True

                if hit:
                    matched_terms.append(term)

            if score <= 0:
                continue

            evidence_l = evidence.strip().lower()
            priority_l = priority.strip().lower()
            subkb_type_l = subkb_type.strip().lower()

            if evidence_l == "core":
                score += 1.0
            elif evidence_l == "supporting":
                score += 0.5
            elif evidence_l == "supplementary":
                score += 0.2

            if priority_l == "p1":
                score += 1.0
            elif priority_l == "p2":
                score += 0.5
            elif priority_l == "p3":
                score += 0.2

            if subkb_type_l in ["shared", "both"]:
                score += 0.3

            rows.append({
                "chunk_id": row.get("chunk_id", ""),
                "doc_id": doc_id,
                "doc_title": doc_title,
                "subkb_type": subkb_type,
                "priority": priority,
                "evidence_level": evidence,
                "use_case": use_case,
                "matched_terms": list(dict.fromkeys(matched_terms)),
                "score": round(score, 4),
            })

        rows = sorted(rows, key=lambda x: x["score"], reverse=True)[:top_k]
        pprint(rows)


if __name__ == "__main__":
    main()