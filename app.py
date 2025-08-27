# app.py
import io
import shutil
from pathlib import Path

import numpy as np
import streamlit as st
import pypdfium2 as pdfium
import easyocr


# ---------- Page config ----------
st.set_page_config(page_title="PDF → OCR (local EasyOCR)", layout="wide")
st.title("📄 OCR z PDF — lokální EasyOCR váhy (bez stahování)")


# ---------- Paths & model prep ----------
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "model" / "easyocr"

# Where your files currently are (as committed)
DET_SUB = MODEL_DIR / "detection" / "craft_mlt_25k.pth"
LAT_SUB = MODEL_DIR / "recognition" / "latin_g2.pth"
ENG_SUB = MODEL_DIR / "recognition" / "english_g2.pth"  # optional

# Where EasyOCR actually looks (root of model_storage_directory)
DET_ROOT = MODEL_DIR / "craft_mlt_25k.pth"
LAT_ROOT = MODEL_DIR / "latin_g2.pth"
ENG_ROOT = MODEL_DIR / "english_g2.pth"


def ensure_root_models():
    """Copy weights from subfolders into MODEL_DIR root if missing there."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if DET_SUB.exists() and not DET_ROOT.exists():
        shutil.copy2(DET_SUB, DET_ROOT)

    if LAT_SUB.exists() and not LAT_ROOT.exists():
        shutil.copy2(LAT_SUB, LAT_ROOT)

    if ENG_SUB.exists() and not ENG_ROOT.exists():
        shutil.copy2(ENG_SUB, ENG_ROOT)


def fmt_size(p: Path) -> str:
    try:
        return f"{p.stat().st_size/1024/1024:.1f} MB"
    except Exception:
        return "missing"


# Ensure files are where EasyOCR expects them
ensure_root_models()

with st.expander("📁 Diagnostika modelů"):
    st.write("Model dir:", str(MODEL_DIR))
    st.write("craft_mlt_25k.pth:", DET_ROOT.exists(), fmt_size(DET_ROOT))
    st.write("latin_g2.pth    :", LAT_ROOT.exists(), fmt_size(LAT_ROOT))
    st.write("english_g2.pth  :", ENG_ROOT.exists(), fmt_size(ENG_ROOT))
    st.caption("Pokud se soubory jeví jako velmi malé (KB), může jít o Git LFS pointer – zkontrolujte LFS kvóty a push.")


# Decide languages based on weights that really exist
LANGS = []
if DET_ROOT.exists() and LAT_ROOT.exists():
    LANGS.append("cs")
if DET_ROOT.exists() and ENG_ROOT.exists():
    LANGS.append("en")

if not LANGS:
    st.error(
        "Nenalezeny požadované soubory modelů v kořeni složky "
        "`model/easyocr`: očekáváno alespoň\n"
        "- craft_mlt_25k.pth (detekce)\n"
        "- latin_g2.pth (pro češtinu)\n"
        "Volitelně: english_g2.pth (pro angličtinu)\n\n"
        "Viz expander výše."
    )
    st.stop()


# ---------- Caching EasyOCR ----------
@st.cache_resource
def load_reader(model_dir: str, langs: list[str]) -> easyocr.Reader:
    # CPU only; never download (použijeme lokální váhy)
    return easyocr.Reader(
        langs,
        gpu=False,
        model_storage_directory=model_dir,
        download_enabled=False,
    )


# ---------- PDF rendering ----------
def pdf_bytes_to_pil_pages(pdf_bytes: bytes, dpi: int = 220):
    """Render each PDF page to a PIL image (no external deps)."""
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    scale = dpi / 72.0
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()
        pages.append(pil)
        page.close()
    pdf.close()
    return pages


# ---------- Controls ----------
c1, c2 = st.columns(2)
with c1:
    dpi = st.slider("Rendering DPI", 150, 300, 220, step=10,
                    help="Vyšší DPI = přesnější OCR, ale pomalejší a náročnější.")
with c2:
    max_pages = st.number_input("Max. počet stran", min_value=1, max_value=300, value=20,
                                help="Omezí počet zpracovaných stran (pro velká PDF).")

uploaded = st.file_uploader("Nahrajte skenované PDF", type=["pdf"])

# ---------- Main flow ----------
if uploaded:
    pdf_bytes = uploaded.read()

    with st.spinner("📄 Renderuji stránky…"):
        pages = pdf_bytes_to_pil_pages(pdf_bytes, dpi)
        if len(pages) > max_pages:
            st.warning(f"PDF má {len(pages)} stran. Zpracovávám pouze prvních {max_pages}.")
            pages = pages[:max_pages]

    # Init OCR (cached)
    try:
        reader = load_reader(str(MODEL_DIR), LANGS)
    except Exception as e:
        st.error("❌ Inicializace EasyOCR selhala.")
        st.exception(e)
        st.stop()

    # OCR per page
    all_text = []
    for idx, pil_img in enumerate(pages, start=1):
        with st.spinner(f"🔎 OCR strana {idx}/{len(pages)}…"):
            try:
                arr = np.array(pil_img)
                lines = reader.readtext(arr, detail=0, paragraph=True)
                page_text = "\n".join(lines).strip()
            except Exception as e:
                page_text = f"[OCR error na straně {idx}: {e}]"

        all_text.append(page_text)

        with st.expander(f"Strana {idx} – text", expanded=False):
            st.text_area(f"Strana {idx}", page_text, height=220)

    # Download all pages as one TXT
    combined = "\n\n".join([f"=== Strana {i+1} ===\n{t}" for i, t in enumerate(all_text)])
    st.download_button("Stáhnout všechny strany jako .txt", combined, "ocr_pages.txt", "text/plain")

else:
    st.info("Nahrajte PDF soubor pro zpracování.")
