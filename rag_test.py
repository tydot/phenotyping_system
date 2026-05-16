from pprint import pprint
from backend.rag.rag_service import RAGService


EXCEL_PATH = "data/ARM_RAG_Final_Master_Table_merged_8papers_plus3chunks_tuned_plus2chunks.xlsx"
SHEET_NAME = "Master_SubKB_Final"


def print_header(title: str):
    print("\n" + "=" * 25 + f" {title} " + "=" * 25)


def run_patient_case(service, case_name, features, top_k=5):
    print_header(f"PATIENT CASE: {case_name}")
    patient_input = {
        "page_type": "patient",
        "features": features,
    }
    result = service.explain_patient(patient_input, top_k=top_k)
    pprint(result)


def run_cluster_case(service, case_name, summary_features, top_k=5):
    print_header(f"CLUSTER CASE: {case_name}")
    cluster_input = {
        "page_type": "cluster",
        "cluster_name": case_name,
        "summary_features": summary_features,
    }
    result = service.explain_cluster(cluster_input, top_k=top_k)
    pprint(result)


def run_free_query_case(service, query, top_k=5):
    print_header(f"FREE QUERY: {query}")
    result = service.search_knowledge(query, top_k=top_k)
    pprint(result)


def main():
    service = RAGService(
        excel_path=EXCEL_PATH,
        sheet_name=SHEET_NAME,
    )

    # =========================
    # PATIENT TESTS
    # =========================

    # 1. 综合型 patient case
    run_patient_case(
        service,
        "low_pressure_poor_propulsion_hyposensation",
        {
            "resting_pressure": "low",
            "msp": "low",
            "defecatory_rectal_pressure": "low",
            "rectal_sensation": "high_threshold",
            "rair": "present",
        },
    )

    # 2. RAIR 缺失
    run_patient_case(
        service,
        "absent_rair_case",
        {
            "resting_pressure": "normal",
            "msp": "normal",
            "defecatory_rectal_pressure": "normal",
            "rectal_sensation": "normal",
            "rair": "absent",
        },
    )

    # 3. 高静息压 + 推进不足
    run_patient_case(
        service,
        "high_resting_pressure_low_propulsion",
        {
            "resting_pressure": "high",
            "msp": "normal",
            "defecatory_rectal_pressure": "low",
            "rectal_sensation": "normal",
            "rair": "present",
        },
    )

    # 4. 直肠感觉减退主导
    run_patient_case(
        service,
        "rectal_hyposensitivity_case",
        {
            "resting_pressure": "normal",
            "msp": "normal",
            "defecatory_rectal_pressure": "normal",
            "rectal_sensation": "high_threshold",
            "rair": "present",
        },
    )

    # 5. 低 squeeze pressure 主导
    run_patient_case(
        service,
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

    # 6. DD / poor propulsion / biofeedback
    run_cluster_case(
        service,
        "cluster_dd_biofeedback",
        [
            "dyssynergia",
            "poor propulsion",
            "biofeedback",
        ],
    )

    # 7. 感觉/RAIR/分类
    run_cluster_case(
        service,
        "cluster_rectal_sensation",
        [
            "rectal sensation",
            "hyposensitivity",
            "RAIR",
            "classification",
        ],
    )

    # 8. 表型 / 感觉功能异常
    run_cluster_case(
        service,
        "cluster_sensory_dysfunction",
        [
            "phenotype",
            "sensory dysfunction",
            "hyposensitivity",
            "pathophysiology",
        ],
    )

    # =========================
    # FREE QUERY / LOGIC TESTS
    # =========================

    # 9. DD 不能只靠单一 ARM 指标诊断
    run_free_query_case(
        service,
        "why can dyssynergic defecation not be diagnosed by a single anorectal manometry metric",
    )

    # 10. HR-ARM、BET、defecography 关系
    run_free_query_case(
        service,
        "relationship between HR-ARM balloon expulsion test and defecography in diagnosing dyssynergic defecation",
    )

    # 11. biofeedback 为什么是一线治疗
    run_free_query_case(
        service,
        "why is biofeedback first line treatment for dyssynergic defecation",
    )


if __name__ == "__main__":
    main()