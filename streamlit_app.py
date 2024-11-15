import streamlit as st
import pandas as pd

st.title('Evaluating Biochar')

df = pd.read_csv("https://raw.githubusercontent.com/OohaSpn/Biochar-Analysis/refs/heads/main/Updated_dataset.csv")
df
