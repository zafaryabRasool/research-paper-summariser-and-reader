#works
from __future__ import annotations
import io
import json
import re
import tempfile
import threading
import http.server
import socketserver
import os
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import streamlit as st
from streamlit.components.v1 import html
import fitz  # PyMuPDF

try:
    import requests
except Exception:
    requests = None

# -------------------------
# Static Server for pdf.js
# -------------------------
def get_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

STATIC_PORT = get_free_port()
STATIC_DIR = os.path.abspath("static")

def start_static_server():
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=STATIC_DIR, **kwargs)
    with socketserver.TCPServer(("", STATIC_PORT), handler) as httpd:
        httpd.serve_forever()

threading.Thread(target=start_static_server, daemon=True).start()

# -------------------------
# Data Classes
# -------------------------
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

# -------------------------
# Prompt builder
# -------------------------
def build_prompt(paragraph: str, cfg: SummaryConfig) -> str:
    return (
        "Summarize the following research paper paragraph in "
        f"{cfg.max_sentences} sentence(s) in {cfg.language}. Do NOT include reasoning, steps, or thoughts. "
        "Only provide the summary text.\n\nParagraph:\n" + paragraph.strip() + "\n\nSummary:"
    )

# -------------------------
# Helpers for text cleaning
# -------------------------
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

# -------------------------
# PDF Parsing
# -------------------------
def extract_paragraphs(pdf_bytes: bytes) -> List[ParagraphData]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
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

# -------------------------
# Save PDF to temp file
# -------------------------
def save_pdf_temp(pdf_bytes: bytes) -> str:
    os.makedirs(os.path.join(STATIC_DIR, "temp"), exist_ok=True)
    filename = f"{uuid.uuid4().hex}.pdf"
    path = os.path.join(STATIC_DIR, "temp", filename)
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return filename

# -------------------------
# Highlight paragraph
# -------------------------
def highlight_paragraph(pdf_bytes: bytes, paragraph: ParagraphData) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[paragraph.page_num]
    rect = fitz.Rect(*paragraph.bbox)
    highlight = page.add_highlight_annot(rect)
    highlight.update()
    return save_pdf_temp(doc.write())

# -------------------------
# Ollama Backend
# -------------------------
class OllamaBackend:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        if requests is None:
            raise RuntimeError("Install requests to use Ollama backend.")
        self.model = model
        self.host = host.rstrip('/')

    def stream_summary(self, paragraph: str, cfg: SummaryConfig) -> Iterable[str]:
        prompt = build_prompt(paragraph, cfg)
        url = f"{self.host}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens}
        }
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
                    token = re.sub(r"<\\|.*?\\|>", "", token)
                    token = token.replace("Thinking:", "")
                    yield token

# -------------------------
# Streamlit UI
# -------------------------
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
    pdf_filename = save_pdf_temp(pdf_bytes)

    left, right = st.columns([2, 2])
    with left:
        st.subheader("Original PDF")
        viewer_url = f"http://localhost:{STATIC_PORT}/pdfjs/web/viewer.html?file=../../temp/{pdf_filename}"
        iframe = f'<iframe src="{viewer_url}" width="100%" height="600" style="border:none;"></iframe>'
        pdf_container = st.empty()
        html(iframe, height=600)

    with st.spinner("Parsing PDF…"):
        paragraphs = extract_paragraphs(pdf_bytes)

    st.success(f"Found {len(paragraphs)} paragraph(s)")

    cfg = SummaryConfig(max_sentences=max_sent, temperature=temperature, max_tokens=max_tokens)
    backend = OllamaBackend(model=model)

    with right:
        st.subheader("Summaries (click to run)")
        for idx, para_obj in enumerate(paragraphs, start=1):
            with st.expander(f"Paragraph {idx}"):
                # st.write(para_obj.text)
                if st.button(f"Summarize {idx}"):
                    # Highlight in PDF
                    highlighted_filename = highlight_paragraph(pdf_bytes, para_obj)
                    viewer_url = f"http://localhost:{STATIC_PORT}/pdfjs/web/viewer.html?file=../../temp/{highlighted_filename}"
                    iframe = f'<iframe src="{viewer_url}" width="100%" height="600" style="border:none;"></iframe>'
                    html(iframe, height=600)

                    # Stream summary
                    placeholder = st.empty()
                    def gen():
                        for tok in backend.stream_summary(para_obj.text, cfg):
                            yield tok
                    placeholder.write_stream(gen)
else:
    st.info("Upload a PDF to start.")
