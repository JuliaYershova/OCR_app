import io
import numpy as np
import streamlit as st
import pypdfium2 as pdfium

# ---- UI ----
st.set_page_config(page_title="PDF OCR per page", layout="wide")
st.title("📄 OCR: show text for each PDF page")

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    dpi = st.slider("Rendering DPI", 150, 300, 220, step=10,
                    help="Higher DPI = sharper OCR but more memory/CPU")
with col2:
    max_pages = st.number_input("Max pages to process", 1, 100, 10,
                                help="Useful to avoid huge PDFs on the cloud")
with col3:
    langs = st.multiselect("OCR languages", ["en", "cs"], default=["en", "cs"],
                           help="Add/remove languages for EasyOCR")

uploaded = st.file_uploader("Upload a scanned PDF", type=["pdf"])

# ---- Helpers ----
@st.cache_resource
def load_easyocr_reader(lang_list):
    import easyocr
    # Use CPU only; avoids GPU issues on cloud
    return easyocr.Reader(lang_list, gpu=False)

def pdf_bytes_to_pil_pages(pdf_bytes: bytes, dpi: int = 220):
    """Render each PDF page to a PIL image using pypdfium2 (no system deps)."""
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

def ocr_page(reader, pil_image):
    """Run EasyOCR on a single PIL image, return list of text lines."""
    img_np = np.array(pil_image)
    # Each element is a line/paragraph because we use detail=0 + paragraph=True
    lines = reader.readtext(img_np, detail=0, paragraph=True)
    return lines

# ---- Main flow ----
if uploaded:
    pdf_bytes = uploaded.read()

    with st.spinner("Rendering PDF pages…"):
        pages = pdf_bytes_to_pil_pages(pdf_bytes, dpi=dpi)
        if len(pages) > max_pages:
            st.warning(f"PDF has {len(pages)} pages. Processing only first {max_pages}.")
            pages = pages[:max_pages]

    # Load OCR once (cached)
    try:
        reader = load_easyocr_reader(langs if langs else ["en"])
    except Exception as e:
        st.error("Failed to initialize EasyOCR reader.")
        st.exception(e)
        st.stop()

    all_text = []
    for idx, pil_img in enumerate(pages, start=1):
        with st.spinner(f"OCR page {idx}/{len(pages)}…"):
            try:
                lines = ocr_page(reader, pil_img)
                page_text = "\n".join(lines).strip()
            except Exception as e:
                page_text = f"[OCR error on page {idx}: {e}]"

        all_text.append(page_text)

        with st.expander(f"Page {idx} text", expanded=False):
            st.text_area(f"Page {idx}", page_text, height=200)

    # Download full text (joined with page markers)
    joined = "\n\n".join([f"=== Page {i+1} ===\n{t}" for i, t in enumerate(all_text)])
    st.download_button("Download all pages as .txt", joined, "ocr_pages.txt", "text/plain")

else:
    st.info("Upload a scanned PDF to begin.")
