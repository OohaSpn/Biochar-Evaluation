import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    st.header('Input feature For Biomass')
    # Dropdown menu for 'raw_material'
    raw_material = st.selectbox(
        'Select Raw Material',
        df['raw_material'].unique()  # Populate options dynamically
    )

    st.header('Input feature for Type of Pollutant')
    # Dropdown menu for 'TP'
    tp_value = st.selectbox(
        'Select TP',
        df['TP'].unique()  # Populate options dynamically
    )

# Filter the dataset based on the selected raw_material
filtered_df_biomass = df[df['raw_material'] == raw_material]

# Filter the dataset based on the selected TP value
filtered_df_tp = df[df['TP'] == tp_value]

# Expander to display filtered data
with st.expander("Filtered Data"):
    st.write(f"Filtered data for raw_material: **{raw_material}**")
    st.dataframe(filtered_df_biomass)

    st.write(f"Filtered data for TP: **{tp_value}**")
    st.dataframe(filtered_df_tp)
with st.expander("Data Visualizations"):
    st.write("Boxplot for Each Column")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))  # Adjust rows and columns as needed
    axes = axes.flatten()  # Flatten the 2D axes array to make indexing easier

    # Loop through the columns and plot each one in a separate subplot
    for i, column in enumerate(numeric_columns):
        df.boxplot(column=column, ax=axes[i])
        axes[i].set_title(f'Box plot for {column}')
        axes[i].set_xlabel('')

    # Adjust layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
  st.info('Boxplots help identify outliers and show how the data is distributed. Some columns are skewed to the right. While a few points appear beyond the upper whisker, they aren't considered outliers since biochar properties depend on pyrolysis conditions.')
