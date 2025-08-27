import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import easyocr
import pypdfium2 as pdfium
from rapidfuzz import process, fuzz
from unidecode import unidecode

# ========== UI SETUP ==========
st.set_page_config(page_title="Medical OCR Parser", layout="wide")
st.title("📄 OCR → Strukturovaná zdravotní data (bez LLM)")

# ========== OCR LOADER ==========
@st.cache_resource
def load_reader():
    # Keep it light for Streamlit Cloud; add 'la' if you really need it and model exists.
    return easyocr.Reader(['en', 'cs'])

reader = load_reader()

# ========== PDF → PIL ==========
def pdf_bytes_to_pil_images(pdf_bytes: bytes, dpi: int = 280):
    pdf = pdfium.PdfDocument(pdf_bytes)
    scale = dpi / 72.0
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        pages.append(page.render(scale=scale).to_pil())
        page.close()
    pdf.close()
    return pages

# ========== TEXT NORMALIZATION ==========
def norm_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "")
    s = s.replace(" ,", ",").replace(" .", ".")
    # fix OCR decimals like "1 . 23" -> "1.23", "1 , 23" -> "1,23"
    s = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", s)
    s = re.sub(r"(\d)\s*,\s*(\d)", r"\1,\2", s)
    return s.strip()

def deaccent(s: str) -> str:
    return unidecode(s or "")

# ========== LAB DICTIONARY (extend as needed) ==========
LAB_DICT = {
    "Urea": ["Urea", "Ureá"],
    "Kreatinin": ["Kreatinin", "Creatinine", "Kreatinín"],
    "Kyselina močová": ["Kyselina močová", "Kyselina mocova", "Uric acid", "K. močová"],
    "GF CKD-EPI": ["GF CKD-EPI", "eGFR CKD-EPI", "GFR CKD-EPI"],
    "Bilirubin celkový": ["Bilirubin celk", "Bil. celk", "Bilirubin celkovy"],
    "ALT": ["ALT", "ALAT"],
    "AST": ["AST", "ASAT"],
    "GGT": ["GGT", "GMT"],
    "Cholesterol": ["Cholesterol", "Chol."],
    "HDL cholesterol": ["HDL cholesterol", "HDL"],
    "non-HDL cholesterol": ["non-HDL cholesterol", "non HDL"],
    "LDL cholesterol": ["LDL cholesterol", "LDL"],
    "Triglyceridy": ["Triglyceridy", "Triacylglyceroly", "TAG", "Iriglyceridy"],
    "Glukóza": ["Glukóza", "Glukoza", "Glucose"],
    "WBC leukocyty": ["WBC", "Leukocyty"],
    "RBC erytrocyty": ["RBC", "Erytrocyty"],
    "HB hemoglobin": ["HB hemoglobin", "Hb", "Hemoglobin"],
    "HCT hematokrit": ["HCT hematokrit", "HCT", "Hematokrit"],
    "MCV": ["MCV", "MCV-stř.obj ery", "MCV stredni objem", "MCV-stř.obj.ery"],
    "MCHC": ["MCHC", "MCHC st.bar k", "MCHC st. barvy"],
    "MCH": ["MCH", "MCH bar.k.ery"],
    "PLT trombocyty": ["PLT", "Trombocyty"],
    "RDW": ["RDW"],
    "Bílkovina": ["Bílkovina", "Bilkovina", "Protein"],
    "Bilirubin": ["Bilirubin"],
    "Urobilinogen": ["Urobilinogen"],
    "pH": ["pH"],
    "Krev": ["Krev"],
    "Leukocyty": ["Leukocyty"],
    "Ketolátky": ["Ketolátky", "Ketolatky"],
    "Nitrity": ["Nitrity"],
    "Specifická hustota": ["Specifická hustota", "Specificka hustota"],
    "Epitel plochý": ["Epitel plochý", "Epitel plochy"],
    "Hlen": ["Hlen"]
}

VAR2CAN = {}
for can, variants in LAB_DICT.items():
    for v in variants + [can]:
        VAR2CAN[v.lower()] = can
VAR_KEYS = list(VAR2CAN.keys())

def canonicalize_lab_name(raw_name: str) -> str:
    q = deaccent(raw_name.lower())
    best = process.extractOne(q, VAR_KEYS, scorer=fuzz.WRatio)
    if best and best[1] >= 85:
        return VAR2CAN[best[0]]
    toks = q.split()
    for span in [toks, toks[-2:], toks[:2]]:
        cand = " ".join(span)
        best2 = process.extractOne(cand, VAR_KEYS, scorer=fuzz.WRatio)
        if best2 and best2[1] >= 85:
            return VAR2CAN[best2[0]]
    return raw_name.strip()

# ========== METADATA EXTRACTION ==========
def extract_metadata(ocr_text: str) -> dict:
    t = norm_spaces(ocr_text)
    meta = {
        "Pacient": {"Jméno": "", "Adresa": "", "Diagnóza": ""},
        "Doktor": {"Jméno": ""},
        "Dokument": {"Datum": ""}
    }
    m = re.search(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b", t)
    if m: meta["Dokument"]["Datum"] = m.group(1)

    m = re.search(r"\b(MUDr\.|MDDr\.)\s+[A-ZÁ-Ž][a-zá-ž]+(?:\s+[A-ZÁ-Ž][a-zá-ž]+)*", t)
    if m: meta["Doktor"]["Jméno"] = m.group(0)

    m = re.search(r"(Pacient|Jméno)\s*[:\-]\s*([A-ZÁ-Ž][a-zá-ž]+(?:\s+[A-ZÁ-Ž][a-zá-ž]+)+)", t)
    if m: meta["Pacient"]["Jméno"] = m.group(2)

    icd = re.findall(r"\b[A-Z]\d{2}(?:\.\d+)?\b", t)
    diag_line = ""
    m = re.search(r"(Diagnóza|Diag\.)\s*[:\-]\s*([^\n]+)", ocr_text, re.IGNORECASE)
    if m:
        diag_line = norm_spaces(m.group(2))
    meta["Pacient"]["Diagnóza"] = (diag_line or ", ".join(icd)).strip()

    m = re.search(r"\b\d{3}\s?\d{2}\b.*", ocr_text)  # line with postal code
    if m:
        meta["Pacient"]["Adresa"] = norm_spaces(m.group(0))

    return meta

# ========== LAB PARSERS ==========
UNIT_CHUNK = r"[A-Za-z%/µ\^\-\d\.]*"
VALUE_RE = rf"[><]?\s*[-+]?\d[\d\s\,\.]*\s*{UNIT_CHUNK}"

def parse_line_by_line(ocr_text: str):
    pat = re.compile(
        rf"(?P<code>\d{{5}})\s+(?P<name>[A-Za-zÁ-Žá-ž\.\-\s]+?)\s+(?P<val>{VALUE_RE})(?:\s*\((?P<rng>[^\)]+)\))?"
    )
    rows = []
    for m in pat.finditer(ocr_text):
        code = m.group("code")
        raw_name = norm_spaces(m.group("name"))
        val = norm_spaces(m.group("val"))
        rng = norm_spaces(m.group("rng") or "")
        rmin, rmax = "", ""
        if rng:
            rng_n = rng.replace("~", "-").replace("–", "-").replace("—", "-")
            rng_n = re.sub(r"\s*-\s*", "-", rng_n)
            parts = re.split(r"-| až ", rng_n)
            if len(parts) >= 2:
                rmin, rmax = parts[0].strip(), parts[1].strip()
            else:
                rmin = rng_n
        name = canonicalize_lab_name(raw_name)
        rows.append([code, name, val, rmin, rmax, raw_name])
    return rows

def parse_two_streams(ocr_text: str):
    # First non-empty line: codes+names in one long line. Remaining lines: values (+ ranges).
    lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
    if not lines: return []
    first = lines[0]
    rest = " ".join(lines[1:])

    code_name_pat = re.compile(r"(\d{5})\s+([A-Za-zÁ-Žá-ž\.\-\s]+?)(?=\s+\d{5}|$)")
    codes_names = [(m.group(1), norm_spaces(m.group(2))) for m in code_name_pat.finditer(first)]

    val_pat = re.compile(rf"(?P<val>{VALUE_RE})(?:\s*\((?P<rng>[^\)]+)\))?")
    vals = []
    for m in val_pat.finditer(rest):
        val = norm_spaces(m.group("val") or "")
        rng = norm_spaces(m.group("rng") or "")
        rmin, rmax = "", ""
        if rng:
            rng_n = rng.replace("~", "-").replace("–", "-").replace("—", "-")
            rng_n = re.sub(r"\s*-\s*", "-", rng_n)
            parts = re.split(r"-| až ", rng_n)
            if len(parts) >= 2:
                rmin, rmax = parts[0].strip(), parts[1].strip()
            else:
                rmin = rng_n
        vals.append((val, rmin, rmax))

    n = min(len(codes_names), len(vals))
    rows = []
    for i in range(n):
        code, raw_name = codes_names[i]
        val, rmin, rmax = vals[i]
        name = canonicalize_lab_name(raw_name)
        rows.append([code, name, val, rmin, rmax, raw_name])
    return rows

def build_lab_df(ocr_text: str) -> pd.DataFrame:
    rows = parse_line_by_line(ocr_text)
    if not rows:
        rows = parse_two_streams(ocr_text)
    cols = ["Kód", "Název", "Výsledek", "Min", "Max", "_RawName"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    if "_RawName" in df.columns:
        df = df.drop(columns=["_RawName"])
    return df

def style_out_of_range(df: pd.DataFrame):
    def to_num(x):
        if pd.isna(x): return None
        s = str(x).replace(",", ".")
        m = re.search(r"[-+]?\d*\.?\d+", s)
        return float(m.group(0)) if m else None
    vals = df["Výsledek"].apply(to_num)
    mins = df["Min"].apply(to_num)
    maxs = df["Max"].apply(to_num)
    def row_style(row):
        i = row.name
        v, mn, mx = vals[i], mins[i], maxs[i]
        if v is not None and (mn is not None or mx is not None):
            if (mn is not None and v < mn) or (mx is not None and v > mx):
                return ["background-color:#ffe5e5"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_style, axis=1)

# ========== APP FLOW ==========
uploaded = st.file_uploader("Nahrajte skenovaný PDF soubor", type=["pdf"])

if uploaded:
    pdf_bytes = uploaded.read()

    with st.spinner("📄 Konvertuji PDF na obrázky…"):
        images = pdf_bytes_to_pil_images(pdf_bytes, dpi=280)

    all_pages = []
    for i, im in enumerate(images, start=1):
        with st.spinner(f"🔎 OCR stránka {i}/{len(images)}…"):
            img_np = np.array(im)
            lines = reader.readtext(img_np, detail=0, paragraph=True)
            all_pages.append("\n".join(lines))

    ocr_text = "\n\n".join(all_pages)

    st.success("✅ OCR hotovo")
    st.subheader("Rozpoznaný text")
    st.text_area("OCR výstup", ocr_text, height=220)

    # Metadata
    st.subheader("🧾 Metadata")
    meta = extract_metadata(ocr_text)
    st.json(meta)

    # Lab table
    st.subheader("📊 Laboratorní výsledky")
    df = build_lab_df(ocr_text)
    if not df.empty:
        st.dataframe(style_out_of_range(df), use_container_width=True)
        st.download_button("Stáhnout CSV", df.to_csv(index=False), "lab_results.csv", "text/csv")
    else:
        st.info("Nenašel jsem strukturovatelné laboratorní výsledky.")

    # Full JSON export
    full_json = {
        "Pacient": meta["Pacient"],
        "Doktor": meta["Doktor"],
        "Dokument": meta["Dokument"],
        "Laboratorní výsledky": json.loads(df.to_json(orient="records", force_ascii=False)) if not df.empty else []
    }
    st.download_button(
        "Stáhnout JSON",
        json.dumps(full_json, ensure_ascii=False, indent=2),
        "structured_medical.json",
        "application/json",
    )
