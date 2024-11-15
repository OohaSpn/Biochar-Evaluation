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
  y_raw = df['Qm (mg/g)']
  y_raw
  
# Input features in the sidebar
with st.sidebar:
    st.header('Input features')
    # Dropdown menu for 'raw_material'
    raw_material = st.selectbox(
        'Select Raw Material',
        df['raw_material'].unique()  # Populate options dynamically
    )

# Filter the dataset based on the selected raw_material
filtered_df = df[df['raw_material'] == raw_material]

# Display the filtered data
st.write(f"Filtered data for raw_material: **{raw_material}**")
st.dataframe(filtered_df)
with st.sidebar:
    st.header('Input features')
    # Dropdown menu for 'TP'
    tp_value = st.selectbox(
        'Select TP Value',
        df['TP'].unique()  # Populate options dynamically
    )

# Filter the dataset based on the selected TP value
filtered_df = df[df['TP'] == tp_value]

# Display the filtered data
st.write(f"Filtered data for TP: **{tp_value}**")
st.dataframe(filtered_df)
