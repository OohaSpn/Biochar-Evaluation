import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, cross_validate, cross_val_predict

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
    columns = ['TemP', 'Time_log', 'PS_log', 'BET_log', 'PV', 'C', 'H', 'N', 'O', 'Qm (mg/g)']
    corr_matrix = df[columns].corr()
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

# Model Training
with st.expander("Model Training"):
    st.write("Training a Random Forest Regressor model with GridSearchCV for hyperparameter tuning.")
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    # Set up K-Fold cross-validation and grid search parameters
    k_folds = KFold(n_splits=5)
    param_rf = {
        'n_estimators': [15, 25, 50, 100, 150],
        'max_depth': [None, 6, 8],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_regressor = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_regressor,
        param_grid=param_rf,
        scoring={'neg_mean_squared_error': 'neg_mean_squared_error', 'mape': mape_scorer},
        refit="neg_mean_squared_error",
        cv=5,
        verbose=1,
        n_jobs=-1
    )
# Perform the grid search
    grid_search.fit(X, y)
    
    # Display best parameters found by GridSearchCV
    best_params = grid_search.best_params_
    st.write("Initial Parameters for Tuning:", param_rf)
    st.write("Best Hyperparameters:", best_params)

# K-Fold Cross-Validation with Best Model
with st.expander("K-Fold Cross-Validation Results"):
    st.write("Evaluating model performance with K-Fold cross-validation.")
    k_rf_mape = cross_val_score(grid_search.best_estimator_, X, y, cv=k_folds, scoring=mape_scorer) * -1
    k_rf_rmse = np.sqrt(-cross_val_score(grid_search.best_estimator_, X, y, cv=k_folds, scoring="neg_mean_squared_error"))

    st.write(f"Mean K-Fold MAPE: {np.mean(k_rf_mape):.4f}")
    st.write(f"Mean K-Fold RMSE: {np.mean(k_rf_rmse):.4f}")

# User File Upload
with st.expander("Predict Using Your Data"):
    uploaded_file = st.file_uploader("Upload your CSV file for predictions:", type=["csv"])
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(user_df)

        try:
            # Preprocess user data
            user_df['Time_log'] = np.log(user_df['Time (min)'] + 1)
            user_df['BET_log'] = np.log(user_df['BET'] + 1)
            user_df['PS_log'] = np.log(user_df['PS'] + 1)
            user_df = user_df.drop(['Time (min)', 'BET', 'PS'], axis=1)

            user_df['raw_material'] = label_encoder.transform(user_df['raw_material'])
            user_df[columns[:-1]] = scaler.transform(user_df[columns[:-1]])

            predictions = grid_search.best_estimator_.predict(user_df)
            user_df['Predicted Qm (mg/g)'] = predictions

            st.write("Predictions:")
            st.dataframe(user_df[['Predicted Qm (mg/g)']])

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
