import streamlit as st
from transformers import pipeline
import torch


pipe = pipeline(task="summarization", model="Falconsai/text_summarization")

st.header("Summerise your text")
text = st.text_area("Paste your text here")

if st.button("Summarize"):
    with st.spinner("Summarizing your text"):
        summary = pipe(text, max_length=100, min_length=30, do_sample=False)
        st.write(summary[0]['summary_text'])