
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.auth.auth_service import require_login

require_login()
from backend.db.query_cohort import (
    get_cohort_overview,
    get_clinical_field_coverage,
    get_key_clinical_summary,
)


def _safe_dict(x):
    return x if isinstance(x, dict) else {}


def get_cohort_view():
    overview = _safe_dict(get_cohort_overview())
    field_coverage = get_clinical_field_coverage() or []
    clinical_summary = get_key_clinical_summary() or []

    n_patients = int(overview.get("n_patients", 0) or 0)
    n_stable = int(overview.get("n_stable", 0) or 0)
    n_boundary = int(overview.get("n_boundary", 0) or 0)

    stable_ratio = n_stable / n_patients if n_patients else 0.0
    boundary_ratio = n_boundary / n_patients if n_patients else 0.0

    cluster_dist = overview.get("cluster_dist") or []

    return {
        "overview": {
            **overview,
            "n_patients": n_patients,
            "n_stable": n_stable,
            "n_boundary": n_boundary,
            "cluster_dist": cluster_dist,
            "stable_ratio": stable_ratio,
            "boundary_ratio": boundary_ratio,
        },
        "clinical_field_coverage": field_coverage,
        "clinical_summary": clinical_summary,
        "summary_text": (
            f"当前系统已完成 {n_patients} 名患者的临床联合表接入，"
            f"其中稳定患者 {n_stable} 名，边界患者 {n_boundary} 名。"
            f"当前数据已可支持 cohort 总览、patient 查询、cluster 级画像和稳定性分析展示。"
        ),
    }
