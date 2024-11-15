import streamlit as st
import pandas as pd

st.title('Evaluating Biochar')
st.info('')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/OohaSpn/Biochar-Analysis/refs/heads/main/Updated_dataset.csv",  usecols=lambda column: column != 'Unnamed: 0')
  df
  
  st.write('**Numeric Columns**')
  numeric_columns = ['TemP', 'Time (min)', 'PS', 'BET', 'PV', 'C', 'N', 'H', 'O', 'Qm (mg/g)']
  numeric_columns
  
  st.write('**Categorical Columns**')
  cat_columns = ['raw_material', 'TP']
  cat_columns
  
  st.write('**X**')
  X_raw = df.drop('Qm (mg/g)', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.Qm (mg/g)
  y_raw
