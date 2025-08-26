import io, re, json
from pathlib import Path

import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd

st.set_page_config(page_title="Medical PDF OCR", layout="wide")

# ---------------- OCR + Parsing ----------------
NUM_PAT  = r"[-+]?\d+(?:[.,]\d+)?"
UNIT_PAT = r"[A-Za-z/%µ\^0-9\.]+"
FLAG_PAT = r"(>>|<<|\+\+|--|↑|↓)"

def pdf_to_images(pdf_bytes, zoom=3):
    """Convert PDF (bytes) -> list of PIL images using PyMuPDF"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        mat = fitz.Matrix(zoom, zoom)  # zoom controls DPI (~72*zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages

def ocr_image(pil_img):
    """OCR a PIL image with preprocessing"""
    open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, lang="ces")
    return text

def extract_metadata(text: str) -> dict:
    meta = {}
    m = re.search(r"Zdravotní pojišťovna[:\s]+(\d+)", text)
    if m: meta["kod_pojistovny"] = m.group(1)
    m = re.search(r"Adresa\s*[:\-]\s*(.+)", text)
    if m: meta["adresa"] = m.group(1).strip()
    m = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
    if m: meta["datum"] = m.group(1)
    m = re.search(r"(?:Jméno\s*pacienta|Pacient)\s*[:\-]\s*([^\n]+)", text, flags=re.I)
    if m: meta["jmeno_pacienta"] = m.group(1).strip()
    return meta

def parse_tests(text: str):
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        if re.match(r"(?i)^(zdravotní pojišťovna|adresa|datum|jméno|pacient)", line):
            continue
        m = re.match(
            rf"^(?P<code>\d{{3,6}})?\s*(?P<name>[A-Za-zÁ-ž0-9\-\.\s]+?)\s+"
            rf"(?P<value>{NUM_PAT}|neg|pos|\+\+)\s*"
            rf"(?P<flag>{FLAG_PAT})?\s*"
            rf"(?P<unit>{UNIT_PAT})?"
            rf"(?:\s*\((?P<ref>[^\)]*)\))?$",
            line
        )
        if m:
            d = m.groupdict()
            d = {k:(v.strip() if v else None) for k,v in d.items()}
            rows.append(d)
    return rows

def tests_to_df(tests: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(tests, columns=["code","name","value","flag","unit","ref"])
    if df.empty: return df
    for c in ["code","name","flag","unit","ref"]:
        df[c] = df[c].astype(str).str.strip().replace({"None": np.nan})
    def _to_float(x):
        if x is None or str(x).strip()=="":
            return np.nan
        s = str(x).replace(",", ".")
        return float(s) if re.fullmatch(NUM_PAT, s) else np.nan
    df["value_num"] = df["value"].apply(_to_float)
    df["value_text"] = df.apply(lambda r: r["value"] if pd.isna(r["value_num"]) else None, axis=1)
    def _parse_ref(s):
        if not isinstance(s, str): return (np.nan, np.nan)
        m = re.search(r"([\-+]?\d+(?:[.,]\d+)?)\s*[-–]\s*([\-+]?\d+(?:[.,]\d+)?)", s)
        if not m: return (np.nan, np.nan)
        return (float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", ".")))
    df[["ref_low","ref_high"]] = df["ref"].apply(_parse_ref).to_list()
    return df

# ---------------- Streamlit UI ----------------
st.title("📑 Medical PDF OCR Extractor")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    with st.spinner("Converting PDF to images..."):
        pages = pdf_to_images(pdf_bytes, zoom=4)  # ~300 dpi

    all_text = []
    for i, page in enumerate(pages):
        with st.spinner(f"OCR page {i+1}..."):
            text = ocr_image(page)
            all_text.append(text)
            with st.expander(f"Raw text page {i+1}"):
                st.text(text)

    full_text = "\n".join(all_text)
    meta = extract_metadata(full_text)
    tests = parse_tests(full_text)
    df = tests_to_df(tests)

    st.subheader("📋 Metadata")
    st.json(meta)
    st.subheader("🧪 Tests Table")
    st.dataframe(df, use_container_width=True)

    # Downloads
    st.download_button("⬇️ Download JSON",
                       data=json.dumps({"metadata": meta, "tests": tests}, ensure_ascii=False, indent=2),
                       file_name="results.json",
                       mime="application/json")
    st.download_button("⬇️ Download CSV",
                       data=df.to_csv(index=False),
                       file_name="tests.csv",
                       mime="text/csv")