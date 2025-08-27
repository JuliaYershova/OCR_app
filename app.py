import streamlit as st
import numpy as np
import pandas as pd
import json
import re

import easyocr
import pypdfium2 as pdfium
from transformers import pipeline

# ---------- OCR ----------
@st.cache_resource
def load_reader():
    # Add 'la' back if you know EasyOCR has the model in your env
    return easyocr.Reader(['en', 'cs'])

reader = load_reader()

# ---------- LLM (tries Qwen 0.5B, falls back to FLAN-T5-small) ----------
@st.cache_resource
def load_llm():
    try:
        return pipeline("text2text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    except Exception:
        return pipeline("text2text-generation", model="google/flan-t5-small")

llm = load_llm()

st.title("📄 OCR → Strukturovaná zdravotní data (Streamlit Cloud, bez klíčů)")

uploaded_file = st.file_uploader("Nahrajte skenovaný PDF soubor", type=["pdf"])

def pdf_bytes_to_pil_images(pdf_bytes, dpi=300):
    # Render with pypdfium2
    pdf = pdfium.PdfDocument(pdf_bytes)
    scale = dpi / 72.0
    pil_images = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale).to_pil()
        pil_images.append(bitmap)
        page.close()
    pdf.close()
    return pil_images

if uploaded_file:
    pdf_bytes = uploaded_file.read()

    # PDF -> PIL pages via pypdfium2
    with st.spinner("📄 Konvertuji PDF na obrázky…"):
        images = pdf_bytes_to_pil_images(pdf_bytes, dpi=300)

    # OCR
    all_text = []
    for i, img in enumerate(images):
        with st.spinner(f"🔎 OCR stránka {i+1}/{len(images)}…"):
            img_np = np.array(img)  # PIL -> numpy
            # EasyOCR returns list[str] using detail=0 + paragraph=True
            lines = reader.readtext(img_np, detail=0, paragraph=True)
            page_text = "\n".join(lines)
            all_text.append(page_text)

    ocr_text = "\n\n".join(all_text)
    st.success("✅ OCR hotovo")
    st.subheader("Rozpoznaný text")
    st.text_area("OCR výstup", ocr_text, height=250)

    # ---------- Ask the model to structure data in Czech ----------
    st.subheader("🔄 Extrakce a strukturování (LLM)")

    prompt = f"""
Jsi asistent pro zpracování lékařských dat. Z následujícího textu extrahuj a vrať
validní JSON se strukturou:

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
- Výsledkem musí být čistý validní JSON (bez komentářů, bez vysvětlení).

Vstupní text:
{ocr_text}
"""

    if st.button("➡️ Extrahovat strukturovaná data"):
        with st.spinner("🧠 Model připravuje JSON…"):
            out = llm(prompt, max_new_tokens=1000)[0]["generated_text"]

        # Try parsing JSON; if model added prose, try to extract the JSON block
        def try_parse_json(text):
            # First, direct parse
            try:
                return json.loads(text)
            except Exception:
                pass
            # Fallback: grab nearest {...} block
            m = re.search(r'\{[\s\S]*\}\s*$', text)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

        data = try_parse_json(out)

        if data is None:
            st.error("❌ Nepodařilo se převést odpověď na validní JSON.")
            st.caption("Surová odpověď modelu:")
            st.code(out)
        else:
            st.success("✅ Strukturovaná data")
            st.json(data)

            # Lab table if present
            if isinstance(data, dict) and "Laboratorní výsledky" in data and isinstance(data["Laboratorní výsledky"], list):
                df = pd.DataFrame(data["Laboratorní výsledky"])
                if not df.empty:
                    st.subheader("📊 Tabulka laboratorních výsledků")
                    st.dataframe(df, use_container_width=True)
                    st.download_button("Stáhnout CSV", df.to_csv(index=False), "lab_results.csv", "text/csv")
            st.download_button(
                "Stáhnout JSON",
                json.dumps(data, indent=2, ensure_ascii=False),
                "structured_medical.json",
                "application/json"
            )
