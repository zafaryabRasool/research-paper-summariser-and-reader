"""
PDF Paragraph Summarizer — fully local, paragraph-by-paragraph, with side‑by‑side view and streaming.

Setup
-----
1) Python deps (suggested):
   pip install streamlit pymupdf llama-cpp-python requests
   
   # If using Apple Silicon, consider: pip install llama-cpp-python==0.2.* --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal

2) Local LLM options (pick one):
   a) llama.cpp backend (recommended for full local, no server):
      - Download a GGUF model (e.g., Llama-3 8B Instruct, Mistral 7B Instruct) and note the file path.
      - Start the app with: streamlit run app.py
      - Select "llama.cpp (GGUF)" in the UI and provide the model_path.

   b) Ollama backend (easy if you already use Ollama):
      - Install and run Ollama (https://ollama.com/) locally.
      - Pull a model, e.g.:  ollama pull llama3:instruct  (or mistral:instruct, qwen2:7b-instruct, etc.)
      - Start the app and select "Ollama" with the model name.

Notes
-----
- Designed for research PDFs (10–15 pages). No OCR; works on digital text PDFs.
- Heuristic two‑column reading order handling. Falls back gracefully to single‑column.
- Streaming summaries paragraph-by-paragraph on the right column.
- Everything stays local (no cloud calls). Ollama uses your local http://localhost:11434.

"""

from __future__ import annotations
import io
import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import streamlit as st
import fitz  # PyMuPDF

# Optional: Only needed if you select llama.cpp backend
try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None  # Allows the app to load even if not installed

# Optional: Only needed if you select Ollama backend
try:
    import requests  # type: ignore
except Exception:
    requests = None


# -----------------------------
# PDF parsing & paragraphization
# -----------------------------

def _clean_text(t: str) -> str:
    # Normalize whitespace but keep paragraph breaks
    t = t.replace("\r", "\n")
    # Fix hyphenation at line breaks: e.g., "inter-
    # national" -> "international"
    t = re.sub(r"-\n(?=[a-z])", "", t)
    # Remove in-line newlines within sentences but keep empty lines as paragraph breaks
    lines = [ln.strip() for ln in t.split("\n")]
    paras: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if ln == "":
            if buf:
                paras.append(" ".join(buf))
                buf = []
        else:
            buf.append(ln)
    if buf:
        paras.append(" ".join(buf))
    # Trim and drop very short junk paragraphs
    paras = [p.strip() for p in paras if len(p.strip()) > 20]
    return "\n\n".join(paras)


def _split_paragraphs(text: str) -> List[str]:
    # Paragraphs separated by blank lines after cleaning
    parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    return parts


def extract_paragraphs(pdf_bytes: bytes) -> List[str]:
    """Extract paragraphs in reading order with a simple two‑column heuristic per page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    paragraphs: List[str] = []

    for page in doc:
        page_width = page.rect.width
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
        # Filter out empty / artifact blocks
        norm_blocks = []
        for b in blocks:
            if len(b) < 5:
                continue
            x0, y0, x1, y1, text = b[:5]
            if not text or not text.strip():
                continue
            cleaned = _clean_text(text)
            if not cleaned:
                continue
            norm_blocks.append((x0, y0, x1, y1, cleaned))

        if not norm_blocks:
            continue

        # Detect two columns by checking if blocks cluster left/right of midline with margin
        mid = page_width / 2
        margin = page_width * 0.05  # 5% dead zone around mid

        left_col = []
        right_col = []
        for (x0, y0, x1, y1, txt) in norm_blocks:
            cx = (x0 + x1) / 2
            if cx < (mid - margin):
                left_col.append((x0, y0, x1, y1, txt))
            elif cx > (mid + margin):
                right_col.append((x0, y0, x1, y1, txt))
            else:
                # Ambiguous blocks near the center -> treat as single-column flow later
                left_col.append((x0, y0, x1, y1, txt))

        # Heuristic: if both columns have significant content, treat as two-column order
        two_col = len(left_col) > 0 and len(right_col) > 0 and (
            min(len(left_col), len(right_col)) >= 0.3 * max(len(left_col), len(right_col))
        )

        ordered_blocks: List[Tuple[float, float, float, float, str]] = []
        if two_col:
            left_col.sort(key=lambda b: (b[1], b[0]))  # by y then x
            right_col.sort(key=lambda b: (b[1], b[0]))
            ordered_blocks = left_col + right_col
        else:
            # Single-column fallback: sort by y then x
            ordered_blocks = sorted(norm_blocks, key=lambda b: (b[1], b[0]))

        for (_x0, _y0, _x1, _y1, txt) in ordered_blocks:
            for para in _split_paragraphs(txt):
                paragraphs.append(para)

    return paragraphs


# -----------------------------
# Local LLM backends
# -----------------------------

@dataclass
class SummaryConfig:
    max_sentences: int = 2
    language: str = "en"
    temperature: float = 0.2
    max_tokens: int = 256


def build_prompt(paragraph: str, cfg: SummaryConfig) -> str:
    return (
        "You are an expert research assistant. Summarize the following research-paper paragraph in "
        f"{cfg.max_sentences} concise sentence(s) in {cfg.language}. Preserve key numbers, units, and named entities. "
        "Avoid adding facts not present. If the paragraph is a list or equation context, summarize the main idea.\n\n"
        "Paragraph:\n" + paragraph.strip() + "\n\nSummary:"
    )


class LlamaCppBackend:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: Optional[int] = None, n_gpu_layers: int = 0):
        if Llama is None:
            raise RuntimeError("llama-cpp-python is not installed. Install it or choose the Ollama backend.")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def stream_summary(self, paragraph: str, cfg: SummaryConfig) -> Iterable[str]:
        prompt = build_prompt(paragraph, cfg)
        # Use completion API for broad GGUF compatibility
        stream = self.llm(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop=["\n\n"],
            stream=True,
        )
        for chunk in stream:
            token = chunk.get("choices", [{}])[0].get("text", "")
            if token:
                yield token


class OllamaBackend:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        if requests is None:
            raise RuntimeError("The 'requests' package is not installed. Install it or choose the llama.cpp backend.")
        self.model = model
        self.host = host.rstrip('/')

    def stream_summary(self, paragraph: str, cfg: SummaryConfig) -> Iterable[str]:
        prompt = build_prompt(paragraph, cfg)
        url = f"{self.host}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
            },
        }
        with requests.post(url, json=data, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if obj.get("done"):
                    break
                token = obj.get("response", "")
                if token:
                    yield token


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="PDF Paragraph Summarizer (Local)", layout="wide")

st.title("📄➡️🧠 PDF Paragraph Summarizer (Local LLM)")
st.caption("Upload a research PDF and get per‑paragraph summaries. All processing is local.")

with st.sidebar:
    st.header("LLM Backend")
    backend_choice = st.selectbox("Backend", ["llama.cpp (GGUF)", "Ollama"], index=0)

    llama_model_path = ""
    ollama_model = ""

    if backend_choice == "llama.cpp (GGUF)":
        llama_model_path = st.text_input("GGUF model_path", placeholder="/path/to/model.gguf")
        n_ctx = st.number_input("Context tokens (n_ctx)", min_value=2048, max_value=32768, value=8192, step=1024)
        n_gpu_layers = st.number_input("n_gpu_layers (0 = CPU)", min_value=0, max_value=200, value=0, step=1)
    else:
        ollama_model = st.text_input("Ollama model name", placeholder="llama3:instruct")
        ollama_host = st.text_input("Ollama host", value="http://localhost:11434")

    st.header("Summary Settings")
    max_sent = st.slider("Sentences per summary", 1, 4, 2)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max tokens per summary", 64, 512, 256, 32)

    st.header("Parsing")
    max_paragraphs = st.number_input("Limit paragraphs (0 = no limit)", min_value=0, max_value=2000, value=0, step=10)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"]) 

if uploaded is not None:
    pdf_bytes = uploaded.read()

    with st.status("Parsing PDF…", expanded=False) as status:
        paragraphs = extract_paragraphs(pdf_bytes)
        if max_paragraphs and max_paragraphs > 0:
            paragraphs = paragraphs[: int(max_paragraphs)]
        status.update(label=f"Found {len(paragraphs)} paragraph(s)", state="complete")

    # Configure backend
    cfg = SummaryConfig(max_sentences=max_sent, temperature=temperature, max_tokens=max_tokens)
    backend = None
    backend_error = None
    try:
        if backend_choice == "llama.cpp (GGUF)":
            if not llama_model_path:
                st.error("Please provide a GGUF model_path in the sidebar.")
            else:
                backend = LlamaCppBackend(model_path=llama_model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        else:
            if not ollama_model:
                st.error("Please provide an Ollama model name in the sidebar.")
            else:
                backend = OllamaBackend(model=ollama_model, host=ollama_host)
    except Exception as e:
        backend_error = str(e)
        st.error(f"Backend init failed: {backend_error}")

    if backend is not None and paragraphs:
        left, right = st.columns(2)
        with left:
            st.subheader("Original Paragraphs")
        with right:
            st.subheader("Summaries (streaming)")

        # Display paragraphs with streaming summaries side by side
        for idx, para in enumerate(paragraphs, start=1):
            left.markdown(f"**Paragraph {idx}**\n\n{para}")
            placeholder = right.empty()

            def token_gen() -> Iterable[str]:
                try:
                    for tok in backend.stream_summary(para, cfg):
                        yield tok
                except Exception as e:
                    yield f"[Error: {e}]"

            # Stream into placeholder
            placeholder.write_stream(token_gen)

        st.success("Done summarizing.")

else:
    st.info("Upload a research PDF to begin. No data leaves your machine.")
