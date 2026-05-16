from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

KB_PATH = DATA_DIR / "ARM_RAG_Final_Master_Table_merged_8papers.xlsx"
MASTER_SHEET_NAME = "Master_SubKB_Final"

_master_df = None


def _pick(row, *candidates) -> str:
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return str(row[c]).strip()
    return ""


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    records = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()

        question_tags = _pick(row_dict, "question_tags")
        keywords = _pick(row_dict, "keywords")
        merged_tags = ";".join([x for x in [question_tags, keywords] if x])

        use_value = _pick(
            row_dict,
            "use",
            "normalized_use",
            "recommended_use",
            "subkb_type",
            "Use",
        )

        page_target = _pick(
            row_dict,
            "page_target",
            "normalized_page_target",
            "Page Target",
        )

        subkb_type = _pick(row_dict, "subkb_type")
        use_case = _pick(row_dict, "use_case")

        record = {
            "chunk_id": _pick(row_dict, "chunk_id", "Chunk ID", "ChunkID"),

            # 文本主体
            "chunk_text": _pick(
                row_dict,
                "chunk_text",
                "Chunk Text",
                "Text",
                "text",
                "content",
            ),

            # tags
            "tags": _pick(
                row_dict,
                "tags",
                "normalized_tags",
                "suggested_tags",
                "Tags",
            ) or merged_tags,

            # use / page metadata
            "use": use_value,
            "page_target": page_target,

            # ranking metadata
            "evidence_level": _pick(
                row_dict,
                "evidence_level",
                "Evidence Level",
            ),
            "retrieval_priority": _pick(
                row_dict,
                "retrieval_priority",
                "priority",
                "Retrieval Priority",
            ),

            # source / title
            "source": _pick(
                row_dict,
                "source",
                "source_filename",
                "source_file",
                "Source",
            ),
            "title": _pick(
                row_dict,
                "title",
                "doc_title",
                "section",
                "doc_id",
                "Title",
            ),

            # raw metadata
            "doc_id": _pick(row_dict, "doc_id"),
            "doc_title": _pick(row_dict, "doc_title"),
            "doc_type": _pick(row_dict, "doc_type"),
            "subkb_type": subkb_type,
            "topic_key": _pick(row_dict, "topic_key"),
            "use_case": use_case,
            "linked_core_doc": _pick(row_dict, "linked_core_doc"),
            "status": _pick(row_dict, "status"),
            "review_notes": _pick(row_dict, "review_notes"),
        }

        records.append(record)

    out = pd.DataFrame(records)

    required_columns = [
        "chunk_id",
        "chunk_text",
        "tags",
        "use",
        "page_target",
        "evidence_level",
        "retrieval_priority",
        "source",
        "title",
        "doc_id",
        "doc_title",
        "doc_type",
        "subkb_type",
        "topic_key",
        "use_case",
        "linked_core_doc",
        "status",
        "review_notes",
    ]

    for col in required_columns:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str).map(_normalize_text)

    return out


def _read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"知识库文件不存在：{path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    df = _standardize_df(df)
    return df


def load_master_kb(force_reload: bool = False) -> pd.DataFrame:
    global _master_df
    if _master_df is None or force_reload:
        _master_df = _read_excel_sheet(KB_PATH, MASTER_SHEET_NAME)
    return _master_df.copy()


def _filter_by_subkb(df: pd.DataFrame, allowed_types) -> pd.DataFrame:
    allowed = {str(x).strip().lower() for x in allowed_types}

    subkb = df["subkb_type"].astype(str).str.strip().str.lower()
    use_col = df["use"].astype(str).str.strip().str.lower()
    page_target = df["page_target"].astype(str).str.strip().str.lower()
    use_case = df["use_case"].astype(str).str.strip().str.lower()

    mask = (
        subkb.isin(allowed)
        | use_col.isin(allowed)
        | page_target.isin(allowed)
        | use_case.isin(allowed)
    )

    return df[mask].copy()


def load_patient_kb(force_reload: bool = False) -> pd.DataFrame:
    """
    patient 侧可见知识：
    patient / both / shared / general
    """
    df = load_master_kb(force_reload=force_reload)
    return _filter_by_subkb(df, ["patient", "both", "shared", "general"])


def load_cluster_kb(force_reload: bool = False) -> pd.DataFrame:
    """
    cluster 侧可见知识：
    cluster / both / shared / general
    """
    df = load_master_kb(force_reload=force_reload)
    return _filter_by_subkb(df, ["cluster", "both", "shared", "general"])