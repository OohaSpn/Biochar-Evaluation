import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_squared_error

st.title('Evaluating Biochar')
st.info('A tool to analyze biochar data and make predictions.')

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

# Data Preprocessing
df['Time_log'] = np.log(df['Time (min)'] + 1)
df['BET_log'] = np.log(df['BET'] + 1)
df['PS_log'] = np.log(df['PS'] + 1)
df = df.drop(['Time (min)', 'BET', 'PS'], axis=1)

label_encoder = LabelEncoder()
df['raw_material'] = label_encoder.fit_transform(df['raw_material'])

columns = ['TemP', 'Time_log', 'PS_log', 'BET_log', 'PV', 'C', 'H', 'N', 'O', 'Qm (mg/g)', 'raw_material']
scaler = StandardScaler()
df[columns] = scaler.fit_transform(df[columns])
X = df.drop(columns=['Qm (mg/g)'])
y = df['Qm (mg/g)']

# Model Training
with st.expander("Model Training"):
    st.write("Training a Random Forest Regressor model with GridSearchCV for hyperparameter tuning.")
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

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
    grid_search.fit(X, y)

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
