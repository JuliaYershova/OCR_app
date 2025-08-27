# app.py
import streamlit as st
import numpy as np
import pandas as pd
import json
import re

import easyocr
import pypdfium2 as pdfium
from transformers import pipeline

# =========================
# Load OCR (EasyOCR)
# =========================
@st.cache_resource
def load_reader():
    # Add 'la' if you have the model available; keeping it lean helps stability
    return easyocr.Reader(['en', 'cs'])

reader = load_reader()

# =========================
# Load LLM (local Qwen causal LM -> fallback to FLAN-T5 small)
# =========================
@st.cache_resource
def load_llm():
    """
    Tries to load a local causal LM from ./Qwen_local (downloaded & committed via Git LFS).
    If unavailable, falls back to a tiny seq2seq model 'google/flan-t5-small' that runs on CPU.
    No internet download of big models is attempted here.
    """
    try:
        # Local causal LM (e.g., Qwen2.5-0.5B-Instruct saved to ./Qwen_local)
        return pipeline(
            task="text-generation",
            model="./Qwen_local",     # put your local model folder here if you have one
            device_map="auto",
            torch_dtype="auto"
        )
    except Exception:
        # Lightweight, reliable CPU fallback
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-small"
        )

llm = load_llm()

def generate_text(llm_pipe, prompt: str, max_new_tokens: int = 700) -> str:
    """Call the pipeline and return plain text from its output."""
    task = getattr(llm_pipe, "task", "")
    if task == "text-generation":  # causal LM (e.g., local Qwen)
        out = llm_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"]
    else:  # seq2seq (flan-t5)
        out = llm_pipe(prompt, max_new_tokens=max_new_tokens)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", out[0].get("summary_text", ""))
        return ""

# =========================
# Helpers
# =========================
def pdf_bytes_to_pil_images(pdf_bytes: bytes, dpi: int = 300):
    pdf = pdfium.PdfDocument(pdf_bytes)
    scale = dpi / 72.0
    pil_images = []
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()
        pil_images.append(pil)
        page.close()
    pdf.close()
    return pil_images

def extract_json_block(text: str):
    """
    Try to parse JSON directly. If that fails, try to extract the last {...} block.
    Returns dict or list on success, else None.
    """
    if not text:
        return None
    # direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # find a JSON object or array block
    m_obj = re.search(r'\{[\s\S]*\}\s*$', text)
    m_arr = re.search(r'\[[\s\S]*\]\s*$', text)
    for m in [m_obj, m_arr]:
        if m:
            block = m.group(0)
            try:
                return json.loads(block)
            except Exception:
                continue
    return None

def normalize_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ,", ",").replace(" .", ".")
    return s.strip()

# Regex fallback parser (in case LLM JSON fails completely)
def parse_two_streams(text: str):
    """
    Handles the 'messed up' case:
    First line contains 'codes + names' sequence,
    subsequent lines contain 'values + (ranges)' sequence in the same order.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return []

    codes_names_line = lines[0]
    values_blob = " ".join(lines[1:])

    # (code, name)
    code_name_pattern = re.compile(r"(\d{5})\s+([A-Za-zÁ-Žá-ž\.\-\s]+?)(?=\s+\d{5}|$)")
    codes_names = [(m.group(1).strip(), normalize_spaces(m.group(2)))
                   for m in code_name_pattern.finditer(codes_names_line)]

    # values + optional (range)
    value_pattern = re.compile(
        r"(?P<val>(?:[><]?\s*[\d\.,]+(?:\s*[xX^/\-\+\*]*\s*[\d\.,]*)*\s*[A-Za-z/%\^\d\.\-]*|\bneg\b|poz\b)[^(\n]*?)\s*(?:\((?P<rng>[^)]+)\))?"
    )
    values = []
    for m in value_pattern.finditer(values_blob):
        val = normalize_spaces(m.group("val") or "")
        rng = (m.group("rng") or "").strip()
        rng = rng.replace("~", "-").replace("–", "-").replace("—", "-")
        rng = re.sub(r"\s*-\s*", "-", rng)
        rmin, rmax = "", ""
        if rng:
            parts = re.split(r"-| až ", rng)
            if len(parts) >= 2:
                rmin, rmax = parts[0].strip(), parts[1].strip()
            else:
                rmin = rng
        values.append((val, rmin, rmax))

    n = min(len(codes_names), len(values))
    return [[codes_names[i][0], codes_names[i][1], values[i][0], values[i][1], values[i][2]]
            for i in range(n)]

def parse_normal_lines(text: str):
    """Normal line-by-line: CODE NAME RESULT (MIN–MAX)"""
    pattern = re.compile(
        r"(?P<code>\d{5})\s+(?P<name>[A-Za-zÁ-Žá-ž\.\-\s]+?)\s+(?P<result>[^()\n]+?)\s*(?:\((?P<range>[^)]+)\))?(?=\n|$)"
    )
    recs = []
    for m in pattern.finditer(text):
        code = m.group("code").strip()
        name = normalize_spaces(m.group("name"))
        result = normalize_spaces(m.group("result"))
        rng = (m.group("range") or "").strip()
        rng = rng.replace("~", "-").replace("–", "-").replace("—", "-")
        rng = re.sub(r"\s*-\s*", "-", rng)
        rmin, rmax = "", ""
        if rng:
            parts = re.split(r"-| až ", rng)
            if len(parts) >= 2:
                rmin, rmax = parts[0].strip(), parts[1].strip()
            else:
                rmin = rng.strip()
        recs.append([code, name, result, rmin, rmax])
    return recs

def regex_fallback_table(ocr_text: str) -> pd.DataFrame:
    recs = parse_normal_lines(ocr_text)
    if not recs:
        recs = parse_two_streams(ocr_text)
    if recs:
        return pd.DataFrame(recs, columns=["Kód", "Název", "Výsledek", "Min", "Max"])
    return pd.DataFrame(columns=["Kód", "Název", "Výsledek", "Min", "Max"])

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

# =========================
# Streamlit UI
# =========================
st.title("📄 OCR → Strukturovaná zdravotní data (Streamlit Cloud, bez klíčů)")
uploaded_file = st.file_uploader("Nahrajte skenovaný PDF soubor", type=["pdf"])

if uploaded_file:
    pdf_bytes = uploaded_file.read()

    # PDF → images
    with st.spinner("📄 Konvertuji PDF na obrázky…"):
        images = pdf_bytes_to_pil_images(pdf_bytes, dpi=300)

    # OCR all pages
    all_text = []
    for i, img in enumerate(images):
        with st.spinner(f"🔎 OCR stránka {i+1}/{len(images)}…"):
            img_np = np.array(img)
            lines = reader.readtext(img_np, detail=0, paragraph=True)
            page_text = "\n".join(lines)
            all_text.append(page_text)

    ocr_text = "\n\n".join(all_text)

    st.success("✅ OCR hotovo")
    st.subheader("Rozpoznaný text")
    st.text_area("OCR výstup", ocr_text, height=240)

    # =========================
    # LLM structuring prompt (Czech, strict JSON)
    # =========================
    prompt = f"""
Jsi asistent pro zpracování lékařských dat. Z následujícího textu extrahuj a vrať
validní JSON přesně v této struktuře (žádný doprovodný text):

{{
  "Pacient": {{
    "Jméno": "",
    "Adresa": "",
    "Diagnóza": ""
  }},
  "Doktor": {{
    "Jméno": ""
  }},
  "Dokument": {{
    "Datum": ""
  }},
  "Laboratorní výsledky": [
    {{
      "Kód": "XXXXX",
      "Název": "Název vyšetření",
      "Výsledek": "hodnota + jednotka",
      "Min": "referenční minimum",
      "Max": "referenční maximum"
    }}
  ]
}}

Pravidla:
- Nepřidávej vymyšlené informace; používej pouze to, co je ve vstupním textu.
- Pokud je údaj nejasný nebo chybí, ponech pole prázdné.
- Odpověz pouze validním JSON (začni {{ a skonči }}).

Vstupní text:
{ocr_text}
""".strip()

    st.subheader("🔄 Strukturování (LLM)")
    if st.button("➡️ Extrahovat strukturovaná data"):
        with st.spinner("🧠 Model připravuje JSON…"):
            raw = generate_text(llm, prompt, max_new_tokens=900)

        data = extract_json_block(raw)
        if data is None:
            st.error("❌ Model nevrátil validní JSON. Zobrazuji regex záložní parsování.")
            # Regex fallback for lab table
            df_fallback = regex_fallback_table(ocr_text)
            if not df_fallback.empty:
                st.subheader("📊 Tabulka laboratorních výsledků (záložní regex parsování)")
                st.dataframe(style_out_of_range(df_fallback), use_container_width=True)
                st.download_button("Stáhnout CSV", df_fallback.to_csv(index=False), "lab_results.csv", "text/csv")
            # Show raw model output for debugging
            with st.expander("Zobrazit surovou odpověď modelu"):
                st.code(raw)
        else:
            st.success("✅ Strukturovaná data")
            st.json(data)

            # Render lab table if present
            if isinstance(data, dict) and "Laboratorní výsledky" in data and isinstance(data["Laboratorní výsledky"], list):
                df = pd.DataFrame(data["Laboratorní výsledky"])
                if not df.empty:
                    st.subheader("📊 Tabulka laboratorních výsledků")
                    st.dataframe(style_out_of_range(df), use_container_width=True)
                    st.download_button("Stáhnout CSV", df.to_csv(index=False), "lab_results.csv", "text/csv")

            # JSON download
            st.download_button(
                "Stáhnout JSON",
                json.dumps(data, indent=2, ensure_ascii=False),
                "structured_medical.json",
                "application/json"
            )
