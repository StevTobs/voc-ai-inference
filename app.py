import streamlit as st
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag

import torch
import re
from tqdm import tqdm
import time

# -----------------------------
# Pandas display setting
# -----------------------------
pd.set_option('display.max_colwidth', None)

st.set_page_config(
    page_title="VOC CSV Viewer + Thai NER",
    layout="wide"
)

st.title("📊 VOC CSV Viewer + Thai NER +++")

# -----------------------------
# Session state for model
# -----------------------------
if "ner_model_loaded" not in st.session_state:
    st.session_state.ner_model_loaded = False
    st.session_state.tokenizer = None
    st.session_state.model = None

# -----------------------------
# Load NER model with progress
# -----------------------------
@st.cache_resource
def load_ner_model_cached():
    name = "pythainlp/thainer-corpus-v2-base-model"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForTokenClassification.from_pretrained(name)
    return tokenizer, model

st.subheader("🧠 Thai NER Model")

load_btn = st.button("🚀 Load NER Model")

if load_btn and not st.session_state.ner_model_loaded:
    status = st.status("กำลังโหลดโมเดล NER...", expanded=True)
    progress = st.progress(0)

    try:
        status.write("🔹 Step 1/2: โหลด tokenizer")
        progress.progress(30)
        time.sleep(0.5)

        status.write("🔹 Step 2/2: โหลด model")
        tokenizer, model = load_ner_model_cached()
        progress.progress(90)
        time.sleep(0.5)

        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.ner_model_loaded = True

        progress.progress(100)
        status.update(label="✅ โหลดโมเดลสำเร็จ", state="complete")

    except Exception as e:
        status.update(label=f"❌ โหลดโมเดลไม่สำเร็จ: {e}", state="error")

elif st.session_state.ner_model_loaded:
    st.success("✅ NER Model พร้อมใช้งานแล้ว")

# -----------------------------
# CSV reader (robust encoding)
# -----------------------------
def read_csv_auto(file):
    encodings = ["utf-8", "utf-8-sig", "cp874", "tis-620", "latin1"]
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            return df, enc
        except Exception:
            continue
    raise ValueError("Cannot decode CSV with known encodings")

st.subheader("📁 Upload VOC CSV")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df, enc = read_csv_auto(uploaded_file)
        st.success(f"Loaded CSV with encoding: {enc}")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
