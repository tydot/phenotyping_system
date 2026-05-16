from pprint import pprint

from backend.graph.build_patient_graph import build_patient_knowledge_graph


def main():
    patient = {
        "patient_id": "210259070",
        "ai_result": {
            "cluster": 2,
            "confidence": 0.87,
            "is_boundary": False,
        },
        "rair": {
            "features": {
                "relaxation_amplitude": 12.3,
                "dose_valid": True,
            }
        },
        "rome_iv": {
            "category": "排便障碍型",
            "propulsion": "不足",
            "coordination": "异常",
        },
        "rag": {
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk_001",
                    "title": "RAIR abnormality and defecatory dysfunction",
                    "source": "RAG",
                    "score": 0.91,
                    "chunk_text": "RAIR异常与部分功能性排便障碍患者相关。",
                }
            ]
        },
        "rag_recommendations": [
            {
                "title": "建议结合生物反馈评估",
                "text": "建议结合临床进一步评估是否适合生物反馈训练。",
                "source": "RAG",
            }
        ],
    }

    graph = build_patient_knowledge_graph(patient)
    pprint(graph, sort_dicts=False)


if __name__ == "__main__":
    main()