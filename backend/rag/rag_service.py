"""
RAG 专家库服务（Excel / DataFrame 驱动版）
用于基于总表知识库进行 patient / cluster 检索与解释生成
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from backend.rag.retriever import (
    retrieve_top_chunks_for_patient,
    retrieve_top_chunks_for_cluster,
)
from backend.rag.generator import (
    generate_patient_explanation,
    generate_cluster_explanation,
)


class RAGService:
    """
    基于 Master_SubKB_Final 的 RAG 服务
    """

    def __init__(
        self,
        excel_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        sheet_name: str = "Master_SubKB_Final",
    ):
        if dataframe is not None:
            self.df = dataframe.copy()
        elif excel_path:
            self.df = self._load_knowledge_base_from_excel(excel_path, sheet_name=sheet_name)
        else:
            raise ValueError("必须提供 excel_path 或 dataframe 之一。")

        self._validate_dataframe()

    def _load_knowledge_base_from_excel(self, excel_path: str, sheet_name: str) -> pd.DataFrame:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        return df.fillna("")

    def _validate_dataframe(self) -> None:
        required_columns = [
            "chunk_id",
            "doc_id",
            "doc_title",
            "evidence_level",
            "subkb_type",
            "topic_key",
            "use_case",
            "priority",
            "page_target",
            "chunk_text",
            "keywords",
            "source_filename",
        ]

        missing = [c for c in required_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"知识库总表缺少必要列: {missing}")

    def get_recommendations(
        self,
        patient_id: str,
        cluster: int,
        symptoms: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        兼容旧接口：按 cluster 返回推荐。
        """
        summary_features = [f"cluster_{cluster}"]
        retrieved = retrieve_top_chunks_for_cluster(
            df=self.df,
            summary_features=summary_features,
            top_k=5,
        )

        recommendations = []
        for item in retrieved:
            recommendations.append({
                "type": "cluster_specific" if item.get("subkb_type") == "cluster" else "general",
                "title": item.get("title", ""),
                "content": item.get("chunk_text", ""),
                "source": item.get("source", ""),
                "relevance": item.get("score", 0),
                "doc_id": item.get("doc_id", ""),
                "topic_key": item.get("topic_key", ""),
                "use_case": item.get("use_case", ""),
                "linked_core_doc": item.get("linked_core_doc", ""),
            })

        return recommendations

    def get_cluster_evidence(
        self,
        cluster_id: int,
        profile: Dict[str, float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        summary_features = [f"cluster_{cluster_id}"]
        return retrieve_top_chunks_for_cluster(
            df=self.df,
            summary_features=summary_features,
            top_k=top_k,
        )

    def search_knowledge(
            self,
            query: str,
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        轻量关键词搜索：
        不再要求整句 query 原样出现在字段里，
        而是拆成关键词后分别在 title/tags/chunk_text 中匹配。
        """
        query_lower = str(query).strip().lower()
        if not query_lower:
            return []

        stopwords = {
            "why", "what", "how", "is", "are", "the", "a", "an", "of", "and", "or",
            "to", "for", "in", "on", "by", "with", "between", "relationship",
            "can", "be", "not", "do", "does", "did", "first", "line"
        }

        raw_terms = query_lower.replace("/", " ").replace("-", " ").split()
        query_terms = [t.strip() for t in raw_terms if t.strip() and t.strip() not in stopwords]

        if not query_terms:
            return []

        results = []

        for _, row in self.df.iterrows():
            doc_title = str(row.get("doc_title", "")).lower()
            question_tags = str(row.get("question_tags", "")).lower()
            keywords = str(row.get("keywords", "")).lower()
            chunk_text = str(row.get("chunk_text", "")).lower()

            searchable_tags = " ".join([question_tags, keywords])
            score = 0.0
            matched_terms = []

            for term in query_terms:
                hit = False

                if term in doc_title:
                    score += 2.0
                    hit = True

                if term in searchable_tags:
                    score += 2.5
                    hit = True

                if term in chunk_text:
                    score += 1.5
                    hit = True

                if hit:
                    matched_terms.append(term)

            if score <= 0:
                continue

            evidence = str(row.get("evidence_level", "")).strip().lower()
            priority = str(row.get("priority", "")).strip().lower()
            subkb_type = str(row.get("subkb_type", "")).strip().lower()

            if evidence == "core":
                score += 1.0
            elif evidence == "supporting":
                score += 0.5
            elif evidence == "supplementary":
                score += 0.2

            if priority == "p1":
                score += 1.0
            elif priority == "p2":
                score += 0.5
            elif priority == "p3":
                score += 0.2

            if subkb_type in ["shared", "both"]:
                score += 0.3

            results.append({
                "chunk_id": row.get("chunk_id", ""),
                "doc_id": row.get("doc_id", ""),
                "title": row.get("doc_title", ""),
                "source": row.get("source_filename", ""),
                "content": row.get("chunk_text", ""),
                "relevance": round(score, 4),
                "matched_terms": list(dict.fromkeys(matched_terms)),
                "evidence_level": row.get("evidence_level", ""),
                "priority": row.get("priority", ""),
                "subkb_type": row.get("subkb_type", ""),
                "topic_key": row.get("topic_key", ""),
                "use_case": row.get("use_case", ""),
                "linked_core_doc": row.get("linked_core_doc", ""),
            })

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

    def explain_patient(
        self,
        patient_input: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        features = patient_input.get("features", {})
        retrieved = retrieve_top_chunks_for_patient(
            df=self.df,
            features=features,
            top_k=top_k,
        )
        return generate_patient_explanation(patient_input, retrieved)

    def explain_cluster(
        self,
        cluster_input: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        summary_features = cluster_input.get("summary_features", [])
        retrieved = retrieve_top_chunks_for_cluster(
            df=self.df,
            summary_features=summary_features,
            top_k=top_k,
        )
        return generate_cluster_explanation(cluster_input, retrieved)