import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

    # Add description below the visualization
    st.info('Boxplots help identify outliers and show how the data is distributed. Here some columns are skewed to the right. While a few points appear beyond the upper whisker, they aren\'t considered outliers since biochar properties depend on pyrolysis conditions.')

    st.write('Distribution After Applying 'Log' to Skewed Data')

    # Apply log transformation to skewed data
    df['Time_log'] = np.log(df['Time (min)'] + 1)  # Add 1 to avoid log(0)
    df['BET_log'] = np.log(df['BET'] + 1)
    df['PS_log'] = np.log(df['PS'] + 1) 

    # Create a figure with subplots for the log-transformed distributions
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))  # Adjust the number of subplots as needed
    sns.histplot(df['Time_log'], kde=True, ax=axes[0])
    axes[0].set_title('Log-Transformed Distribution of Time (min)')
    sns.histplot(df['BET_log'], kde=True, ax=axes[1])
    axes[1].set_title('Log-Transformed Distribution of BET')
    sns.histplot(df['PS_log'], kde=True, ax=axes[2])  # Changed to 2 for correct indexing
    axes[2].set_title('Log-Transformed Distribution of PS')

    # Adjust layout and display in Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    df = df.drop(['Time' , 'BET' , 'PS'], axis = 1)
    st.write("Pearson Correlation Between Features")
    updated_columns = ['TemP', 'Time_log', 'PS_log', 'BET_log', 'PV', 'C', 'H', 'N', 'O', 'Qm (mg/g)']
    corr_matrix = df[updated_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
    st.pyplot(fig)

