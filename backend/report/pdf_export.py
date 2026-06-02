"""
backend/report/pdf_export.py
Patient comprehensive report PDF export
"""

import os
from datetime import datetime
from pathlib import Path
from fpdf import FPDF

ROOT_DIR = Path(__file__).resolve().parents[2]


class PatientReportPDF(FPDF):
    def __init__(self, patient_id):
        super().__init__("P", "mm", "A4")
        self.patient_id = patient_id
        self.set_auto_page_break(True, 15)
        self.font_name = self._load_cjk_font()

    def _load_cjk_font(self):
        """Load a Chinese-capable font: local SimSun > cached Noto Sans SC > download Noto."""
        # 1. Try Windows system fonts
        for name, path in [
            ("SimSun", "C:/Windows/Fonts/simsun.ttc"),
            ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
            ("MSYH", "C:/Windows/Fonts/msyh.ttc"),
        ]:
            if os.path.exists(path):
                self.add_font(name, "", path, uni=True)
                self.add_font(name, "B", path, uni=True)
                return name

        # 2. Try cached Noto Sans SC
        cache_dir = ROOT_DIR / ".cache" / "fonts"
        cache_dir.mkdir(parents=True, exist_ok=True)
        font_path = cache_dir / "NotoSansSC-Regular.ttf"
        bold_path = cache_dir / "NotoSansSC-Bold.ttf"

        if not font_path.exists():
            self._download_noto_font(font_path, bold_path, cache_dir)

        if font_path.exists():
            self.add_font("NotoSansSC", "", str(font_path), uni=True)
            if bold_path.exists():
                self.add_font("NotoSansSC", "B", str(bold_path), uni=True)
            else:
                self.add_font("NotoSansSC", "B", str(font_path), uni=True)  # fallback to regular
            return "NotoSansSC"

        # 3. Fallback
        return "Helvetica"

    def _download_noto_font(self, font_path, bold_path, cache_dir):
        """Download Noto Sans SC from Google Fonts (subset, ~5MB)."""
        import requests
        urls = [
            ("https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf", str(font_path)),
        ]
        for url, dest in urls:
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(resp.content)
            except Exception:
                pass

    def header(self):
        self.set_font(self.font_name, "B", 10)
        self.cell(0, 6, f"ARM 功能表型综合报告 | 患者 {self.patient_id}", align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_name, "", 8)
        self.cell(0, 10, f"第 {self.page_no()} 页 | {datetime.now().strftime('%Y-%m-%d %H:%M')} | 仅用于科研分析", align="C")

    def _s(self, v, d="-"):
        if v is None: return d
        if isinstance(v, float) and v != v: return d
        return str(v)

    def _t(self, title):
        self.set_font(self.font_name, "B", 12)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 6, f"  {title}", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(10)

    def _kv(self, k, v):
        self.set_font(self.font_name, "B", 9); self.cell(50, 5, f"{k}: ")
        self.set_font(self.font_name, "", 9); self.cell(140, 5, self._s(v)); self.ln(5)

    def _p(self, t):
        self.set_font(self.font_name, "", 9); self.multi_cell(0, 5, t)

    def _np(self, n=60):
        if self.get_y() > 297 - 15 - n: self.add_page()

    def add_ai(self, ai):
        self._t("AI 共识分型")
        self._kv("版本", ai.get("version","-"))
        self._kv("共识簇", ai.get("cluster","-"))
        self._kv("置信度", ai.get("confidence","-"))
        self._kv("边界患者", "是" if ai.get("is_boundary") else "否")
        self.ln(2)

    def add_clinical(self, am):
        self._np(50); self._t("临床指标与参考范围判定")
        if am:
            for item in am:
                if isinstance(item, dict):
                    self.set_font(self.font_name, "", 9)
                    self._p(f"{self._s(item.get('metric','?'))}: {self._s(item.get('value'))} [{self._s(item.get('state_text','?'))}]")
        else: self._p("当前患者未检出明显异常指标。")
        self.ln(2)

    def add_llm(self, report):
        self._np(80); self._t("LLM 科研解释报告")
        t = report if isinstance(report, str) else "报告未生成"
        self._p(t[:4000] if len(t) > 4000 else t)
        self.ln(2)

    def add_rair(self, rair):
        self._np(40); self._t("RAIR 外部验证")
        if rair:
            self._kv("RAIR 状态", rair.get("status","-"))
            self._kv("H 统计量", rair.get("H","-"))
            self._kv("p 值", rair.get("p_value","-"))
        else: self._p("当前患者 RAIR 数据不可用。")
        self.ln(2)

    def add_rome(self, rome):
        self._t("Rome IV 代理分型")
        if rome:
            self._kv("四分型", rome.get("rome_iv_type","-"))
            self._kv("三分类", rome.get("rome_three_class","-"))
        else: self._p("当前患者 Rome IV 数据不可用。")
        self.ln(2)

    def add_kg(self, kg):
        self._t("知识图谱推理链")
        if kg:
            for i, p in enumerate(kg[:5], 1): self._p(f"{i}. {str(p)}")
        else: self._p("当前患者无可用知识图谱路径。")
        self.ln(2)

    def add_rag(self, chunks):
        self._np(40); self._t("RAG 文献证据")
        if chunks:
            for i, ch in enumerate(chunks[:4], 1):
                if isinstance(ch, dict):
                    self.set_font(self.font_name, "B", 9)
                    self.cell(0, 5, f"[{i}] {ch.get('title','文献条目')}"); self.ln(5)
                    self.set_font(self.font_name, "", 9)
                    self._p((ch.get("chunk_text","") or ch.get("text",""))[:400])
                    self.ln(2)
        else: self._p("当前患者无可用 RAG 文献证据。")
        self.ln(2)

    def add_vlm(self, findings):
        self._np(50); self._t("VLM 图像侧区域解释")
        if findings:
            for item in findings[:5]:
                if isinstance(item, dict):
                    self.set_font(self.font_name, "B", 9)
                    proto = item.get("matched_protocol", item.get("region_name", "区域"))
                    self.cell(0, 5, f"协议: {proto}"); self.ln(5)
                    self.set_font(self.font_name, "", 9)
                    self._p(f"  {item.get('finding','-')} [{item.get('visual_support','?')}]")
                    self.ln(2)
        else: self._p("当前患者无可用的 VLM 图像侧解释。")
        self.ln(2)

    def add_disclaimer(self):
        self._np(20)
        self.set_font(self.font_name, "", 7)
        self.set_text_color(128, 128, 128)
        self.multi_cell(0, 4, "Research use only. Not for clinical diagnosis.")
        self.set_text_color(0, 0, 0)


def generate_patient_report_pdf(patient_id, **kwargs):
    pdf = PatientReportPDF(str(patient_id))
    pdf.add_page()
    pdf.add_ai(kwargs.get("ai_result") or {})
    pdf.add_clinical(kwargs.get("abnormal_metrics") or [])
    pdf.add_llm(kwargs.get("llm_report") or "N/A")
    pdf.add_rair(kwargs.get("rair_info") or {})
    pdf.add_rome(kwargs.get("rome") or {})
    pdf.add_kg(kwargs.get("kg_paths") or [])
    pdf.add_rag(kwargs.get("rag_chunks") or [])
    pdf.add_vlm(kwargs.get("vlm_findings") or [])
    pdf.add_disclaimer()
    return pdf.output()
