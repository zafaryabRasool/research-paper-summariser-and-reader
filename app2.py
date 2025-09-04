from __future__ import annotations
import io
import json
import re
import base64
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import streamlit as st
import fitz  # PyMuPDF
from streamlit.components.v1 import html

try:
    import requests
except Exception:
    requests = None

@dataclass
class ParagraphData:
    text: str
    page_num: int
    bbox: Tuple[float, float, float, float]

@dataclass
class SummaryConfig:
    max_sentences: int = 2
    language: str = "en"
    temperature: float = 0.2
    max_tokens: int = 256

def build_prompt(paragraph: str, cfg: SummaryConfig) -> str:
    return (
        "Summarize the following research paper paragraph in "
        f"{cfg.max_sentences} sentence(s) in {cfg.language}. Do NOT include reasoning, steps, or thoughts. "
        "Only provide the summary text.\n\nParagraph:\n" + paragraph.strip() + "\n\nSummary:"
    )

def _clean_text(t: str) -> str:
    t = t.replace("\r", "\n")
    t = re.sub(r"-\n(?=[a-z])", "", t)
    lines = [ln.strip() for ln in t.split("\n")]
    paras, buf = [], []
    for ln in lines:
        if ln == "":
            if buf:
                paras.append(" ".join(buf))
                buf = []
        else:
            buf.append(ln)
    if buf:
        paras.append(" ".join(buf))
    paras = [p.strip() for p in paras if len(p.strip()) > 20]
    return "\n\n".join(paras)

def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

def extract_paragraphs(pdf_bytes: bytes) -> List[ParagraphData]:
    pdf_stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_stream.read(), filetype="pdf")
    paragraphs: List[ParagraphData] = []
    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            if len(b) < 5:
                continue
            x0, y0, x1, y1, text = b[:5]
            if not text.strip():
                continue
            cleaned = _clean_text(text)
            if cleaned:
                for para in _split_paragraphs(cleaned):
                    if len(para.split()) >= 30:
                        paragraphs.append(ParagraphData(text=para, page_num=page_idx, bbox=(x0, y0, x1, y1)))
    return paragraphs

def highlight_paragraph(pdf_bytes: bytes, paragraph: ParagraphData) -> str:
    pdf_stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_stream.read(), filetype="pdf")
    page = doc[paragraph.page_num]
    rect = fitz.Rect(*paragraph.bbox)
    highlight = page.add_highlight_annot(rect)
    highlight.update()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    with open(tmp.name, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

class OllamaBackend:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        if requests is None:
            raise RuntimeError("Install requests.")
        self.model = model
        self.host = host.rstrip('/')

    def stream_summary(self, paragraph: str, cfg: SummaryConfig) -> Iterable[str]:
        prompt = build_prompt(paragraph, cfg)
        url = f"{self.host}/api/generate"
        data = {"model": self.model, "prompt": prompt, "stream": True, "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens}}
        with requests.post(url, json=data, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line.decode("utf-8"))
                if obj.get("done"):
                    break
                token = obj.get("response", "")
                if token:
                    token = re.sub(r"<\|.*?\|>", "", token)
                    token = token.replace("Thinking:", "")
                    yield token

st.set_page_config(page_title="PDF Summarizer (Ollama)", layout="wide")
st.title("📄➡️🧠 PDF Summarizer (Local LLM)")

with st.sidebar:
    model = st.text_input("Ollama model", "qwen3:8b")
    max_sent = st.slider("Sentences", 1, 4, 2)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max tokens", 64, 512, 256, 32)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    pdf_bytes = uploaded.read()

    left, right = st.columns([2, 2])
    with left:
        st.subheader("Original PDF")
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f"""
        <iframe src="https://mozilla.github.io/pdf.js/web/viewer.html?file=data:application/pdf;base64,{pdf_base64}" width="100%" height="600"></iframe>
        """
        pdf_container = st.empty()
        # pdf_container.html(pdf_display, height=600)
        with pdf_container:
            html(pdf_display, height=600)

    with st.spinner("Parsing PDF…"):
        paragraphs = extract_paragraphs(pdf_bytes)

    st.success(f"Found {len(paragraphs)} paragraph(s)")

    cfg = SummaryConfig(max_sentences=max_sent, temperature=temperature, max_tokens=max_tokens)
    backend = OllamaBackend(model=model)

    with right:
        st.subheader("Summaries (click to run)")
        for idx, para_obj in enumerate(paragraphs, start=1):
            with st.expander(f"Paragraph {idx}"):
                st.write(para_obj.text)
                if st.button(f"Summarize {idx}"):
                    # Highlight in PDF
                    highlighted_pdf_base64 = highlight_paragraph(pdf_bytes, para_obj)
                    pdf_display = f"""
                    <iframe src="https://mozilla.github.io/pdf.js/web/viewer.html?file=data:application/pdf;base64,{highlighted_pdf_base64}" width="100%" height="600"></iframe>
                    """
                    with pdf_container:
                        html(pdf_display, height=600)

                    # Stream summary
                    placeholder = st.empty()
                    def gen():
                        for tok in backend.stream_summary(para_obj.text, cfg):
                            yield tok
                    placeholder.write_stream(gen)
else:
    st.info("Upload a PDF to start.")
