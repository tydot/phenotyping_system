from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PatientInput:
    page_type: str = "patient"
    features: Dict[str, str] = field(default_factory=dict)


@dataclass
class ClusterInput:
    page_type: str = "cluster"
    cluster_name: str = ""
    summary_features: List[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    chunk_id: str
    title: str
    source: str
    chunk_text: str
    score: float
    evidence_level: str
    retrieval_priority: str
    matched_terms: List[str] = field(default_factory=list)
    matched_tags: List[str] = field(default_factory=list)

    # 兼容总表元数据
    doc_id: str = ""
    doc_type: str = ""
    subkb_type: str = ""
    topic_key: str = ""
    use_case: str = ""
    page_target: str = ""
    linked_core_doc: str = ""