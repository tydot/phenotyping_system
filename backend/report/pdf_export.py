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
        self.cjk = False
        for name, path in [
            ("SimSun", "C:/Windows/Fonts/simsun.ttc"),
            ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
        ]:
            if os.path.exists(path):
                self.add_font(name, "", path, uni=True)
                self.font_name = name
                self.cjk = True
                return
        self.font_name = "Helvetica"

    def header(self):
        self.set_font(self.font_name, "", 10)
        self.cell(0, 6, self._ascii(f"ARM Report | Patient {self.patient_id}"), align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_name, "", 8)
        self.cell(0, 10, self._ascii(f"Page {self.page_no()} | {datetime.now().strftime('%Y-%m-%d %H:%M')}"), align="C")

    def _ascii(self, t):
        if self.cjk: return str(t)
        try:
            return str(t).encode("ascii", errors="replace").decode("ascii")
        except Exception:
            return "?"

    def _s(self, v, d="-"):
        if v is None: return d
        if isinstance(v, float) and v != v: return d
        return str(v)

    def _t(self, title):
        self.set_font(self.font_name, "", 12)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 6, self._ascii(f"  {title}"), fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(10)

    def _kv(self, k, v):
        self.set_font(self.font_name, "", 9)
        self.cell(55, 5, self._ascii(f"{k}: "))
        self.cell(135, 5, self._ascii(self._s(v)))
        self.ln(5)

    def _p(self, t):
        self.set_font(self.font_name, "", 9)
        self.multi_cell(0, 5, self._ascii(t))

    def _np(self, n=60):
        if self.get_y() > 297 - 15 - n:
            self.add_page()

    def add_ai(self, ai):
        self._t("AI Consensus")
        self._kv("Version", ai.get("version", "-"))
        self._kv("Cluster", ai.get("cluster", "-"))
        self._kv("Confidence", ai.get("confidence", "-"))
        self._kv("Boundary", "Yes" if ai.get("is_boundary") else "No")
        self.ln(2)

    def add_clinical(self, am):
        self._np(50)
        self._t("Clinical Metrics")
        if am:
            for item in am:
                if isinstance(item, dict):
                    txt = f"{self._ascii(self._s(item.get('metric','?')))}: {self._s(item.get('value'))} [{self._ascii(self._s(item.get('state_text','?')))}]"
                    self.set_font(self.font_name, "", 9)
                    self.cell(0, 5, txt)
                    self.ln(5)
        else:
            self._p("No abnormal metrics.")
        self.ln(2)

    def add_llm(self, report):
        self._np(80)
        self._t("LLM Report")
        t = report if isinstance(report, str) else "N/A"
        self._p(t[:4000] if len(t) > 4000 else t)
        self.ln(2)

    def add_rair(self, rair):
        self._np(40)
        self._t("RAIR Validation")
        if rair:
            self._kv("Status", rair.get("status", "-"))
            self._kv("H", rair.get("H", "-"))
            self._kv("p", rair.get("p_value", "-"))
        else:
            self._p("RAIR N/A")
        self.ln(2)

    def add_rome(self, rome):
        self._t("Rome IV Proxy")
        if rome:
            self._kv("Type", rome.get("rome_iv_type", "-"))
            self._kv("Class", rome.get("rome_three_class", "-"))
        else:
            self._p("Rome IV N/A")
        self.ln(2)

    def add_kg(self, kg):
        self._t("Knowledge Graph")
        if kg:
            for i, p in enumerate(kg[:5], 1):
                self._p(f"{i}. {str(p)}")
        else:
            self._p("No KG paths.")
        self.ln(2)

    def add_rag(self, chunks):
        self._np(40)
        self._t("RAG Evidence")
        if chunks:
            for i, ch in enumerate(chunks[:4], 1):
                if isinstance(ch, dict):
                    self.set_font(self.font_name, "", 9)
                    self.cell(0, 5, self._ascii(f"[{i}] {ch.get('title','Ref')}"))
                    self.ln(5)
                    self._p((ch.get("chunk_text", "") or ch.get("text", ""))[:400])
                    self.ln(2)
        else:
            self._p("No RAG evidence.")
        self.ln(2)

    def add_vlm(self, findings):
        self._np(50)
        self._t("VLM Regional")
        if findings:
            for item in findings[:5]:
                if isinstance(item, dict):
                    self.set_font(self.font_name, "", 9)
                    proto = item.get("matched_protocol", item.get("region_name", "N/A"))
                    self.cell(0, 5, self._ascii(f"Protocol: {proto}"))
                    self.ln(5)
                    self._p(f"  {item.get('finding','-')} [{item.get('visual_support','?')}]")
                    self.ln(2)
        else:
            self._p("No VLM findings.")
        self.ln(2)

    def add_disclaimer(self):
        self._np(20)
        self.set_font(self.font_name, "", 7)
        self.set_text_color(128, 128, 128)
        self.multi_cell(0, 4, self._ascii("Research use only. Not for clinical diagnosis."))
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
    return bytes(pdf.output())
