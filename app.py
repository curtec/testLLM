import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline
import torch

load_dotenv()
pipe = pipeline(task="summarization", model="Falconsai/text_summarization",
                      torch_dtype=torch.bfloat16)

st.header("Summerise your text")
text = st.text_area("Paste your text here")

if st.button("Summarise"):
    with st.spinner("Summarising your text"):
        summary = pipe(text, max_length=200, min_length=50, do_sample=False)
        st.write(summary[0]['summary_text'])