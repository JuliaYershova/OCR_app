import streamlit as st
import easyocr
import numpy as np
import pandas as pd
import json
from pdf2image import convert_from_bytes
from transformers import pipeline

# ===== Load OCR =====
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en', 'cs', 'la'])

reader = load_reader()

# ===== Load small LLM (Qwen or FLAN-T5) =====
@st.cache_resource
def load_model():
    # lightweight model for structuring tasks
    return pipeline("text2text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

llm = load_model()

st.title("📄 Medical PDF OCR")

uploaded_file = st.file_uploader("Nahrajte skenovaný PDF soubor", type=["pdf"])

if uploaded_file:
    # Convert PDF → images
    images = convert_from_bytes(uploaded_file.read(), dpi=300)

    all_text = []
    for i, img in enumerate(images):
        with st.spinner(f"Zpracovávám stránku {i+1}..."):
            img_np = np.array(img)  # PIL → numpy
            results = reader.readtext(img_np, detail=0, paragraph=True)
            page_text = "\n".join(results)
            all_text.append(page_text)

    ocr_text = "\n\n".join(all_text)

    st.success("✅ OCR hotovo")
    st.subheader("Rozpoznaný text")
    st.text_area("OCR výstup", ocr_text, height=300)

    # ===== Structuring with Qwen (small local model) =====
    st.subheader("🔄 Strukturovaná data")

    prompt = f"""
    Jsi asistent pro zpracování lékařských dat. Tvým úkolem je z následujícího textu 
    extrahovat a vrátit ve struktuře JSON následující informace:

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

    - Nepřidávej žádné vymyšlené informace, použij jen to, co je ve vstupním textu.
    - Pokud je údaj nejasný nebo chybí, ponech pole prázdné.
    - Odpověď vrať jako validní JSON.
    - Výstup piš v češtině.

    Vstupní text:
    {ocr_text}
    """

    if st.button("➡️ Extrahovat strukturovaná data"):
        with st.spinner("Model připravuje JSON..."):
            output = llm(prompt, max_new_tokens=800)[0]["generated_text"]
            try:
                data = json.loads(output)
                # zobraz pacient/doctor info
                st.json(data)

                # pokud existují výsledky, ukaž tabulku
                if "Laboratorní výsledky" in data:
                    df = pd.DataFrame(data["Laboratorní výsledky"])
                    st.dataframe(df)

                    st.download_button("Stáhnout CSV", df.to_csv(index=False), "lab_results.csv", "text/csv")
                    st.download_button("Stáhnout JSON", json.dumps(data, indent=2, ensure_ascii=False),
                                       "lab_results.json", "application/json")
            except Exception as e:
                st.error(f"❌ Nepodařilo se převést na JSON: {e}")
                st.write("Model vrátil:", output)
