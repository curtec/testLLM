import streamlit as st
from transformers import pipeline
import torch


pipe = pipeline(task="summarization", model="Falconsai/text_summarization",
                      torch_dtype=torch.bfloat16)

st.header("Summerise your text")
text = st.text_area("Paste your text here")

if st.button("Summarise"):
    with st.spinner("Summarising your text"):
        summary = pipe(text, max_length=50, min_length=3, do_sample=False)
        st.write(summary[0]['summary_text'])