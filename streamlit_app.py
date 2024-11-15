import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_squared_error

st.title('Evaluating Biochar')
st.info('hi')

# Data Display
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv("https://raw.githubusercontent.com/OohaSpn/Biochar-Analysis/refs/heads/main/Updated_dataset.csv",  usecols=lambda column: column != 'Unnamed: 0')
    st.dataframe(df)

    st.write('**Numeric Columns**')
    numeric_columns = ['TemP', 'Time (min)', 'PS', 'BET', 'PV', 'C', 'N', 'H', 'O', 'Qm (mg/g)']
    st.write(numeric_columns)

    st.write('**Categorical Columns**')
    cat_columns = ['raw_material', 'TP']
    st.write(cat_columns)

    st.write('**X**')
    X_raw = df.drop('Qm (mg/g)', axis=1)
    st.write(X_raw)

    st.write('**y**')
    y_raw = df['Qm (mg/g)']
    st.write(y_raw)

# Sidebar Inputs
with st.sidebar:
    st.header('Input feature For Biomass')
    raw_material = st.selectbox('Select Raw Material', df['raw_material'].unique())

    st.header('Input feature for Type of Pollutant')
    tp_value = st.selectbox('Select TP', df['TP'].unique())

# Filtered Data
filtered_df_biomass = df[df['raw_material'] == raw_material]
filtered_df_tp = df[df['TP'] == tp_value]

with st.expander("Filtered Data"):
    st.write(f"Filtered data for raw_material: **{raw_material}**")
    st.dataframe(filtered_df_biomass)

    st.write(f"Filtered data for TP: **{tp_value}**")
    st.dataframe(filtered_df_tp)

# Data Visualizations
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

    plt.tight_layout()
    st.pyplot(fig)

    st.info('Boxplots help identify outliers and show how the data is distributed.')

    st.write('Distribution After Applying Log to Skewed Data')

    # Apply log transformation to skewed data
    df['Time_log'] = np.log(df['Time (min)'] + 1)  # Add 1 to avoid log(0)
    df['BET_log'] = np.log(df['BET'] + 1)
    df['PS_log'] = np.log(df['PS'] + 1)

    # Create a figure with subplots for the log-transformed distributions
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    sns.histplot(df['Time_log'], kde=True, ax=axes[0])
    axes[0].set_title('Log-Transformed Distribution of Time (min)')
    sns.histplot(df['BET_log'], kde=True, ax=axes[1])
    axes[1].set_title('Log-Transformed Distribution of BET')
    sns.histplot(df['PS_log'], kde=True, ax=axes[2])
    axes[2].set_title('Log-Transformed Distribution of PS')

    plt.tight_layout()
    st.pyplot(fig)

    df = df.drop(['Time (min)', 'BET', 'PS'], axis=1)
    
    # Pearson Correlation
    st.write("Pearson Correlation Between Features")
    corr_matrix = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
    st.pyplot(fig)
    st.info("There is a positive correlation between PV and BET with correlation value 0.67.")

    # Preprocessing the Data
    label_encoder = LabelEncoder()
    df['raw_material'] = label_encoder.fit_transform(df['raw_material'])
    columns = ['TemP', 'Time_log', 'PS_log', 'BET_log', 'PV', 'C', 'H', 'N', 'O', 'Qm (mg/g)', 'raw_material']
    scaler = StandardScaler()
    
    # Apply standardization to the selected columns
    df[columns] = scaler.fit_transform(df[columns])
    df = df[columns]
    X = df.drop(columns=['Qm (mg/g)'])  # Drop target column
    y = df['Qm (mg/g)']  # Target column
    # Display the preprocessed X and y in Streamlit for checking
 with st.expander('Preprocessed Data (X, y)'):
    st.write('**X (Features)**')
    st.dataframe(X)

    st.write('**y (Target)**')
    st.dataframe(y)



