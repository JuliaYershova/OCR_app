import io
import numpy as np
import streamlit as st
import pypdfium2 as pdfium
import easyocr

st.set_page_config(page_title="PDF → OCR (per page)", layout="wide")
st.title("📄 OCR z nahraného PDF (lokální EasyOCR modely)")

# ---------- Controls ----------
col1, col2 = st.columns(2)
with col1:
    dpi = st.slider("Rendering DPI", 150, 300, 220, step=10,
                    help="Vyšší DPI = přesnější OCR, ale pomalejší a náročnější")
with col2:
    max_pages = st.number_input("Max. počet stran ke zpracování", 1, 200, 20,
                                help="Omezí počet stran kvůli paměti")

uploaded = st.file_uploader("Nahrajte skenované PDF", type=["pdf"])

# ---------- Local EasyOCR models ----------
MODEL_DIR = "model/easyocr"  # <- your committed weights

@st.cache_resource
def load_reader(model_dir: str):
    # Use only local models; no downloads in the cloud
    return easyocr.Reader(
        ['en', 'cs'],
        gpu=False,
        model_storage_directory=model_dir,
        download_enabled=False
    )

def pdf_bytes_to_pil_pages(pdf_bytes: bytes, dpi_val: int = 220):
    """Render each PDF page to a PIL image with pypdfium2."""
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    scale = dpi_val / 72.0
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()
        pages.append(pil)
        page.close()
    pdf.close()
    return pages

def ocr_pil_page(reader: easyocr.Reader, pil_img):
    """Run EasyOCR on a PIL image and return list of lines/paragraphs."""
    arr = np.array(pil_img)
    return reader.readtext(arr, detail=0, paragraph=True)

if uploaded:
    pdf_bytes = uploaded.read()

    # Render
    with st.spinner("📄 Renderuji stránky…"):
        pages = pdf_bytes_to_pil_pages(pdf_bytes, dpi)
        if len(pages) > max_pages:
            st.warning(f"Soubor má {len(pages)} stran. Zpracovávám pouze prvních {max_pages}.")
            pages = pages[:max_pages]

    # Init OCR
    try:
        reader = load_reader(MODEL_DIR)
    except Exception as e:
        st.error("❌ EasyOCR se nepodařilo inicializovat s lokálními modely.")
        st.write("Zkontrolujte, že existují tyto soubory:")
        st.code(
            "model/easyocr/detection/craft_mlt_25k.pth\n"
            "model/easyocr/recognition/latin_g2.pth\n"
            "(volitelně) model/easyocr/recognition/english_g2.pth"
        )
        st.exception(e)
        st.stop()

    # OCR per page
    all_text = []
    for idx, pil_img in enumerate(pages, start=1):
        with st.spinner(f"🔎 OCR strana {idx}/{len(pages)}…"):
            try:
                lines = ocr_pil_page(reader, pil_img)
                page_text = "\n".join(lines).strip()
            except Exception as e:
                page_text = f"[OCR error na straně {idx}: {e}]"
        all_text.append(page_text)

        with st.expander(f"Strana {idx} – text", expanded=False):
            st.text_area(f"Strana {idx}", page_text, height=220)

    # Download all text
    combined = "\n\n".join([f"=== Strana {i+1} ===\n{t}" for i, t in enumerate(all_text)])
    st.download_button("Stáhnout všechny strany jako .txt", combined, "ocr_pages.txt", "text/plain")

else:
    st.info("Nahrajte PDF soubor pro zpracování.")
