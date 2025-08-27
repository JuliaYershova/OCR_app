import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

st.title("📄 OCR PDF Extractor")

uploaded_file = st.file_uploader("Upload a scanned PDF", type=["pdf"])
lang = st.text_input("Language code (default = eng)", "eng")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Convert PDF to list of PIL images
        images = convert_from_bytes(uploaded_file.read(), dpi=300)

        all_text = []
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_column_width=True)
            text = pytesseract.image_to_string(img, lang=lang)
            all_text.append(f"--- Page {i+1} ---\n{text}")

        result_text = "\n\n".join(all_text)

        st.subheader("Extracted Text")
        st.text_area("OCR Result", result_text, height=400)

        # Option to download text
        st.download_button(
            "Download OCR Text",
            result_text,
            file_name="ocr_result.txt"
        )
