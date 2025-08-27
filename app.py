import io, os
from pathlib import Path
import numpy as np
import streamlit as st
import pypdfium2 as pdfium
import easyocr

st.set_page_config(page_title="PDF → OCR (local EasyOCR)", layout="wide")
st.title("📄 OCR per page — local EasyOCR weights")

# ---- controls ----
c1, c2 = st.columns(2)
with c1:
    dpi = st.slider("Rendering DPI", 150, 300, 220, step=10)
with c2:
    max_pages = st.number_input("Max pages", 1, 200, 20)

# ---- paths & checks ----
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "model" / "easyocr"
DET = MODEL_DIR / "detection" / "craft_mlt_25k.pth"
EN = MODEL_DIR / "recognition" / "english_g2.pth"
LA = MODEL_DIR / "recognition" / "latin_g2.pth"

def fmt_size(p: Path):
    try:
        return f"{p.stat().st_size/1024/1024:.1f} MB"
    except Exception:
        return "missing"

with st.expander("📁 Model files (diagnostics)"):
    st.write("Model dir:", str(MODEL_DIR))
    st.write("craft_mlt_25k.pth:", DET.exists(), fmt_size(DET))
    st.write("english_g2.pth  :", EN.exists(), fmt_size(EN))
    st.write("latin_g2.pth    :", LA.exists(), fmt_size(LA))
    # If a file is only a tiny few KB, it's likely a Git LFS pointer (LFS quota issue)

# Decide languages based on what’s actually present
langs = []
if LA.exists() and DET.exists():
    # Czech uses latin_g2
    langs.append("cs")
if EN.exists() and DET.exists():
    # add English only if its weight exists
    langs.append("en")

if not langs:
    st.error("No valid model files found. Ensure these exist in the repo:\n"
             "model/easyocr/detection/craft_mlt_25k.pth\n"
             "model/easyocr/recognition/latin_g2.pth (for 'cs')\n"
             "(optional) model/easyocr/recognition/english_g2.pth (for 'en')\n"
             "Also verify they’re real files, not small Git LFS pointers (check sizes above).")
    st.stop()

@st.cache_resource
def load_reader(model_dir: str, lang_list):
    # Strictly use local files; never download
    return easyocr.Reader(
        lang_list,
        gpu=False,
        model_storage_directory=model_dir,
        download_enabled=False
    )

def pdf_bytes_to_pil_pages(pdf_bytes: bytes, dpi_val: int = 220):
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    scale = dpi_val / 72.0
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        pages.append(page.render(scale=scale).to_pil())
        page.close()
    pdf.close()
    return pages

uploaded = st.file_uploader("Upload scanned PDF", type=["pdf"])

if uploaded:
    # render pages
    with st.spinner("Rendering pages…"):
        pages = pdf_bytes_to_pil_pages(uploaded.read(), dpi)
        if len(pages) > max_pages:
            st.warning(f"PDF has {len(pages)} pages. Processing first {max_pages}.")
            pages = pages[:max_pages]

    # init OCR
    try:
        reader = load_reader(str(MODEL_DIR), langs)
    except Exception as e:
        st.error("EasyOCR init failed. See diagnostics above.")
        st.exception(e)
        st.stop()

    # ocr per page
    all_text = []
    for i, pil in enumerate(pages, start=1):
        with st.spinner(f"OCR page {i}/{len(pages)}…"):
            try:
                arr = np.array(pil)
                lines = reader.readtext(arr, detail=0, paragraph=True)
                text = "\n".join(lines).strip()
            except Exception as e:
                text = f"[OCR error on page {i}: {e}]"
        all_text.append(text)
        with st.expander(f"Page {i} text", expanded=False):
            st.text_area(f"Page {i}", text, height=220)

    joined = "\n\n".join([f"=== Page {i+1} ===\n{t}" for i, t in enumerate(all_text)])
    st.download_button("Download all text (.txt)", joined, "ocr_pages.txt", "text/plain")
else:
    st.info("Upload a PDF to start.")
