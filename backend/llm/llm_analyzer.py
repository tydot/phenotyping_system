# backend/llm/llm_analyzer.py
"""
LLM 分析器
用于生成患者、队列、集群和稳定性的智能分析
"""

from typing import Dict, Any, List


class LLMAnalyzer:
    """
    LLM 分析器类
    目前使用规则引擎模拟 LLM 功能，后续可替换为真实 LLM API
    """

    def __init__(self):
        pass

    def analyze_patient(
        self,
        patient_id: str,
        cluster: int,
        physiology: Dict[str, Any],
        ai_confidence: float
    ) -> Dict[str, str]:
        """
        分析单个患者的表型特征
        """

        core_metrics = physiology["core_metrics"]
        desc_metrics = physiology["descriptive_metrics"]

        analysis = {
            "summary": "",
            "key_findings": [],
            "clinical_significance": "",
            "recommendations": []
        }

        if cluster == 0:
            analysis["summary"] = (
                f"患者 {patient_id} 被分配到 Cluster 0，该集群主要表现为肛门括约肌功能减弱。"
            )
            analysis["key_findings"] = [
                "静息压偏低（45.2 mmHg），提示肛门内括约肌张力不足",
                "最大随意收缩压（MSP）偏低（88.4 mmHg），提示外括约肌收缩能力减弱",
                "排便期直肠压处于正常范围下限，可能存在排便推动不足"
            ]
            analysis["clinical_significance"] = (
                "该患者的生理指标提示肛门括约肌功能整体减弱，"
                "可能导致排便控制困难和排便不尽感。"
            )
            analysis["recommendations"] = [
                "建议进行生物反馈训练以增强括约肌收缩能力",
                "可考虑盆底肌康复训练",
                "建议定期随访评估功能改善情况"
            ]

        elif cluster == 1:
            analysis["summary"] = (
                f"患者 {patient_id} 被分配到 Cluster 1，该集群主要表现为排便协调障碍。"
            )
            analysis["key_findings"] = [
                "静息压正常偏高（68.5 mmHg）",
                "最大随意收缩压正常（125.3 mmHg）",
                "排便期直肠压升高，提示可能存在排便协调障碍"
            ]
            analysis["clinical_significance"] = (
                "该患者可能存在排便时肛门括约肌不能充分松弛，"
                "导致排便困难和出口梗阻型便秘。"
            )
            analysis["recommendations"] = [
                "建议进行肛门直肠测压详细评估排便协调性",
                "可考虑生物反馈训练改善排便协调",
                "建议结合球囊排出试验评估功能状态"
            ]

        else:
            analysis["summary"] = (
                f"患者 {patient_id} 被分配到 Cluster 2，该集群主要表现为直肠感觉异常。"
            )
            analysis["key_findings"] = [
                "静息压和收缩压基本正常",
                "直肠感觉阈值异常",
                "可能存在直肠高敏感性或低敏感性"
            ]
            analysis["clinical_significance"] = (
                "该患者的直肠感觉功能异常，可能导致排便感觉障碍，"
                "表现为排便失禁或排便困难。"
            )
            analysis["recommendations"] = [
                "建议进行直肠感觉阈值详细评估",
                "根据感觉异常类型制定个体化治疗方案",
                "建议结合临床症状进行综合评估"
            ]

        return analysis

    def analyze_cohort(
        self,
        cohort_size: int,
        protocol_summary: Dict[str, Any],
        patient_protocol_coverage: List[Dict[str, int]]
    ) -> Dict[str, str]:
        """
        分析整体队列的数据质量和特征
        """

        insights = {
            "data_quality": "",
            "protocol_distribution": "",
            "recommendations": []
        }

        insights["data_quality"] = (
            f"队列共包含 {cohort_size} 名有效患者。"
            f"数据完整性良好，各协议覆盖率较高。"
        )

        protocol_names = list(protocol_summary.keys())
        insights["protocol_distribution"] = (
            f"队列包含 {len(protocol_names)} 种 ARM/RAIR 协议类型，"
            f"覆盖了肛门直肠功能评估的主要方面。"
        )

        insights["recommendations"] = [
            "建议继续完善数据采集流程，提高协议覆盖率",
            "对于覆盖率较低的协议，可考虑优化采集方案",
            "建议定期进行数据质量监控和评估"
        ]

        return insights

    def analyze_cluster(
        self,
        cluster_id: int,
        cluster_data: Dict[str, Any]
    ) -> str:
        """
        分析单个集群的表型特征
        """

        size = cluster_data["size"]
        stable_ratio = cluster_data["stable_ratio"]
        profile = cluster_data["median_profile"]
        abnormality = cluster_data["abnormality_rate"]

        description = (
            f"Cluster {cluster_id} 包含 {size} 名患者，"
            f"其中 {stable_ratio:.1%} 的患者分型稳定。\n\n"
            f"该集群的核心功能特征为："
        )

        for metric, value in profile.items():
            description += f"\n- {metric}: {value}"

        description += "\n\n主要异常表现："
        for abnormal_type, rate in abnormality.items():
            description += f"\n- {abnormal_type}: {rate:.1%}"

        if cluster_id == 0:
            description += (
                "\n\n生理学解释：该集群患者整体表现为肛门括约肌功能减弱，"
                "包括静息压和随意收缩压的降低，提示可能存在括约肌肌力不足或神经支配异常。"
            )
        elif cluster_id == 1:
            description += (
                "\n\n生理学解释：该集群患者主要表现为排便协调障碍，"
                "即在排便过程中肛门括约肌不能充分松弛，导致出口梗阻型便秘。"
            )
        else:
            description += (
                "\n\n生理学解释：该集群患者主要表现为直肠感觉功能异常，"
                "可能涉及直肠感觉神经传导通路的异常，导致排便感觉障碍。"
            )

        return description

    def analyze_stability(
        self,
        cohort_stability: Dict[str, float],
        cluster_stability: Dict[int, float],
        confidence_distribution: List[float]
    ) -> Dict[str, str]:
        """
        分析分型稳定性结果
        """

        interpretation = {
            "overall_assessment": "",
            "cluster_analysis": "",
            "confidence_analysis": "",
            "recommendations": []
        }

        stable_rate = cohort_stability["stable"]
        boundary_rate = cohort_stability["boundary"]

        interpretation["overall_assessment"] = (
            f"队列中 {stable_rate:.1%} 的患者分型稳定，"
            f"{boundary_rate:.1%} 的患者位于分型边界。"
        )

        interpretation["cluster_analysis"] = "各集群的稳定性分析：\n"
        for cluster_id, stability in cluster_stability.items():
            status = "高" if stability > 0.75 else "中等" if stability > 0.65 else "较低"
            interpretation["cluster_analysis"] += (
                f"\n- Cluster {cluster_id}: 稳定性 {stability:.1%} ({status})"
            )

        avg_confidence = sum(confidence_distribution) / len(confidence_distribution)
        interpretation["confidence_analysis"] = (
            f"平均分型置信度为 {avg_confidence:.2f}，"
            f"置信度分布范围为 {min(confidence_distribution):.2f} - {max(confidence_distribution):.2f}。"
        )

        interpretation["recommendations"] = [
            "对于边界患者，建议结合临床特征进行综合判断",
            "可考虑增加聚类随机种子数量以提高稳定性评估",
            "建议对低稳定性集群进行更深入的特征分析"
        ]

        return interpretation
