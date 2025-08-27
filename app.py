import re
from typing import Dict, Optional, List

import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import pandas as pd


# ===== UI & config =====
st.set_page_config(page_title="OCR skenovaného PDF", layout="wide")
st.title("📄 OCR skenovaného PDF")
st.markdown(
    "Nahrajte **skenované PDF**. Každá stránka se vykreslí jako obrázek a "
    "proběhne **OCR v češtině**. Surový text uvidíte u každé stránky. "
    "**Extrakce identifikačních údajů a laboratorních testů** se provádí "
    "na konci z **kompletního textu celého dokumentu**."
)

uploaded_file = st.file_uploader("Nahrajte PDF", type=["pdf"])
lang = "ces"
dpi = st.slider("Render DPI (vyšší = ostřejší OCR, ale pomalejší)", 150, 400, 300, 50)


# ===== Helpers =====
@st.cache_data(show_spinner=False)
def pdf_to_images_from_bytes(file_bytes: bytes, dpi_val: int = 300) -> List[Image.Image]:
    """Render PDF na seznam PIL obrázků přes pdf2image (vyžaduje Poppler)."""
    return convert_from_bytes(file_bytes, dpi=dpi_val)

def ocr_image(img: Image.Image, lang_code: str) -> str:
    """OCR přes Tesseract (bez cache kvůli nehashovatelným objektům)."""
    return pytesseract.image_to_string(img, lang=lang_code)

def _find_first(patterns: List[re.Pattern], text: str) -> Optional[str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return re.sub(r"[,\s]+$", "", m.group(1).strip())
    return None

def extract_id_fields(text: str) -> Dict[str, Optional[str]]:
    """Extrahuje jméno pacienta, ZP, RČ, adresu a lékaře (MUDr.)."""
    norm = re.sub(r"[ \t]+", " ", text)

    jmeno_pats = [
        re.compile(r"Jméno\s*pacienta[:\s]*([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][^\n,]+)", re.IGNORECASE),
        #re.compile(r"Pacient(?:ka)?[:\s]*([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][^\n,]+)", re.IGNORECASE),
        #re.compile(r"Jméno[:\s]*([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][^\n,]+)", re.IGNORECASE),
    ]
    zdr_poj = [
        re.compile(r"Zdravotní\s+pojišťovna[:\s]*([^\n,]+)", re.IGNORECASE),
        #re.compile(r"ZP[:\s]*([^\n,]+)", re.IGNORECASE),
        #re.compile(r"Pojišťovna[:\s]*([^\n,]+)", re.IGNORECASE),
    ]
    rc_patterns = [
        re.compile(r"Rodné\s*číslo[:\s]*([0-9]{2,6}\s*/?\s*[0-9]{3,4})", re.IGNORECASE),
        #re.compile(r"RČ[:\s]*([0-9]{2,6}\s*/?\s*[0-9]{3,4})", re.IGNORECASE),
    ]
    adresa_patterns = [
        # pouze přesně "Adresa:" na začátku řádku (s volitelnými mezerami)
        re.compile(r"^\s*Adresa\s*:\s*([^\n]+)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*Bydliště\s*:\s*([^\n]+)", re.IGNORECASE | re.MULTILINE),
    ]
    mudr_patterns = [
        re.compile(r"(?:MUDr\.?|MUDR\.?)\s*([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][^\n,]+)"),
    ]

    return {
        "Jméno pacienta": _find_first(jmeno_pats, norm),
        "Zdravotní pojišťovna": _find_first(zdr_poj, norm),
        "Rodné číslo": _find_first(rc_patterns, norm),
        "Adresa": _find_first(adresa_patterns, norm),
        "Doktor (MUDr.)": _find_first(mudr_patterns, norm),
    }

def _to_float_maybe(val: str) -> Optional[float]:
    s = val.strip().replace(" ", "").replace("\u202f", "").replace("\xa0", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def parse_labs(text: str) -> pd.DataFrame:
    """
    Parsuje lab. řádky jako:
    03085 Urea 4.0 mmol/L (2.8 - 8.1)
    03077 Kyselina močová 447 >> umol/L (202 - 417)
    03364 Glukóza neg arb.j. (0 - 1)
    -> kód, název, hodnota, jednotka, norma_min, norma_max, poznámka
    """
    rows = []
    line_re = re.compile(
        r"""^\s*
        (?P<kod>\d{3,6})\s+
        (?P<nazev>[A-Za-zÁ-Žá-ž0-9\.\-/%\s]+?)\s+
        (?P<hodnota_raw>
            (?:[<>]*\s*[+-]*\s*\d+(?:[.,]\d+)?)|
            (?:neg|poz|poz\.)|
            (?:trace|stopa|stop\.)
        )
        [\s>]*                       # volitelné >> / <<
        (?P<jednotka>[^\s(]+)?       # jednotka
        \s*
        (?:\((?P<rozsah>[^)]*)\))?   # (a - b)
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    arrow_after_val = re.compile(r"\s*(>>|<<)\s*")

    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        m = line_re.match(line)
        if not m:
            continue

        kod = m.group("kod")
        nazev = (m.group("nazev") or "").strip(" :;.-")
        hodnota_raw = (m.group("hodnota_raw") or "").strip()
        jednotka = (m.group("jednotka") or "").strip()
        rozsah = (m.group("rozsah") or "").strip()

        pozn = ""
        hodnota_clean = arrow_after_val.sub("", hodnota_raw).strip("<> ")
        hodnota_num = _to_float_maybe(hodnota_clean)
        hodnota_out = str(hodnota_num) if hodnota_num is not None else hodnota_clean

        norma_min = norma_max = None
        if rozsah:
            r = rozsah.replace(",", ".")
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", r)
            if len(nums) >= 2:
                norma_min = _to_float_maybe(nums[0])
                norma_max = _to_float_maybe(nums[1])

        if ">>" in line:
            pozn = "výrazně zvýšeno"
        elif "<<" in line:
            pozn = "výrazně sníženo"
        elif "neg" in hodnota_raw.lower():
            pozn = "negativní"
        elif "poz" in hodnota_raw.lower():
            pozn = "pozitivní"

        rows.append(
            {
                "kód": kod,
                "název": nazev,
                "hodnota": hodnota_out,
                "jednotka": jednotka or None,
                "norma_min": norma_min,
                "norma_max": norma_max,
                "poznámka": pozn or None,
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        df["název"] = df["název"].str.replace(r"\s+", " ", regex=True).str.strip()
        with pd.option_context("mode.copy_on_write", True):
            df["kód_num"] = pd.to_numeric(df["kód"], errors="coerce")
            df = df.sort_values(["kód_num", "název"]).drop(columns=["kód_num"])
        return df
    return pd.DataFrame(columns=["kód", "název", "hodnota", "jednotka", "norma_min", "norma_max", "poznámka"])


# ===== Main flow =====
if uploaded_file:
    file_bytes = uploaded_file.read()

    with st.spinner("Vykresluji stránky PDF…"):
        pages = pdf_to_images_from_bytes(file_bytes, dpi_val=dpi)

    vsechny_texty = []           # [(index_strany, text)]
    vsechny_texty_only = []      # [text_bez_hlavicek] pro join
    vsechny_laby_per_page = []   # případně pro debug

    for i, page_img in enumerate(pages, start=1):
        st.markdown(f"## Stránka {i}")
        st.image(page_img, caption=f"Stránka {i}", use_container_width=True)

        with st.spinner(f"OCR stránka {i}/{len(pages)}…"):
            text = ocr_image(page_img, lang_code=lang)

        # uložení textu
        vsechny_texty.append((i, text))
        vsechny_texty_only.append(text)

        # zobraz jen surový text této stránky (na přání)
        with st.expander("Zobrazit surový text (OCR)"):
            st.text_area("Text", text, height=220)

        st.divider()

    # ===== EXTRAKCE NA KONCI Z CELÉHO TEXTU =====
    st.markdown("## Extrakce z celého dokumentu")

    full_text = "\n\n".join(vsechny_texty_only)

    # 1) Identifikační údaje (celý dokument)
    st.markdown("### Identifikační údaje")
    meta_all = extract_id_fields(full_text)
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Jméno pacienta:** {meta_all.get('Jméno pacienta') or '—'}")
        st.write(f"**Zdravotní pojišťovna:** {meta_all.get('Zdravotní pojišťovna') or '—'}")
        st.write(f"**Rodné číslo:** {meta_all.get('Rodné číslo') or '—'}")
    with c2:
        st.write(f"**Adresa:** {meta_all.get('Adresa') or '—'}")
        st.write(f"**Doktor (MUDr.):** {meta_all.get('Doktor (MUDr.)') or '—'}")

    # 2) Laboratorní nálezy (celý dokument)
    st.markdown("### Laboratorní nálezy")
    df_all_labs = parse_labs(full_text)
    if not df_all_labs.empty:
        st.dataframe(df_all_labs, use_container_width=True)
        st.download_button(
            "⬇️ Stáhnout všechny laby (CSV)",
            df_all_labs.to_csv(index=False).encode("utf-8-sig"),
            file_name="lab_vsechny.csv",
            mime="text/csv",
        )
    else:
        st.info("V celém dokumentu se nepodařilo bezpečně rozpoznat standardní laboratorní řádky.")

    # 3) Stažení kompletního OCR textu (pro audit/debug)
    full_text_with_headers = "\n\n".join(
        [f"--- Stránka {i} ---\n{t.strip()}" for i, t in vsechny_texty]
    )
    st.download_button("⬇️ Stáhnout veškerý OCR text", full_text_with_headers, file_name="ocr_text.txt")

    # Diagnostika Tesseractu
    with st.expander("Diagnostika"):
        try:
            ver = pytesseract.get_tesseract_version()
            st.write(f"Verze Tesseract: {ver}")
            st.write(f"Použité jazyky: {lang}")
        except Exception as e:
            st.error(f"Tesseract nenalezen: {e}")

else:
    st.info("Nahrajte skenované PDF pro zpracování.")
