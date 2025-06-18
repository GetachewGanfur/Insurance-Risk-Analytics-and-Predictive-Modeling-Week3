
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap 
import warnings
warnings.filterwarnings("ignore")


# Load & Clean Data
file_path = "../data/MachineLearningRating_v3.txt"

df = pd.read_csv(file_path, sep="|")

# Remove policies without any claims for claim severity prediction
df_claim = df[df['TotalClaims'] > 0].copy()

# Convert 'CapitalOutstanding' to numeric, handling non-numeric values
df_claim['CapitalOutstanding'] = pd.to_numeric(df_claim['CapitalOutstanding'], errors='coerce')

# Drop rows with missing or invalid values
df_claim = df_claim.dropna(subset=['TotalClaims', 'CapitalOutstanding'])



# Feature Selection
# Example: select only numeric and categorical columns
feature_cols = [
    'Cubiccapacity', 'Kilowatts', 'CapitalOutstanding', 'SumInsured', 
    'NewVehicle', 'Gender', 'Province', 'VehicleType'
]

df_model = df_claim[feature_cols + ['TotalClaims']].dropna()

# Encode categoricals
df_model = pd.get_dummies(df_model, columns=['NewVehicle', 'Gender', 'Province', 'VehicleType'], drop_first=True)

# Train-Test Split
X = df_model.drop('TotalClaims', axis=1)
y = df_model['TotalClaims']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training & Evaluation Function
def train_and_evaluate_model(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name} — RMSE: {rmse:.2f}, R²: {r2:.4f}")
    return model

# Linear Regression
lr = train_and_evaluate_model(LinearRegression(), "Linear Regression")

# Random Forest
rf = train_and_evaluate_model(RandomForestRegressor(random_state=42), "Random Forest")

# XGBoost
xgb = train_and_evaluate_model(XGBRegressor(random_state=42, n_jobs=-1), "XGBoost")

# Step 6: Model Explainability (SHAP)
explainer = shap.Explainer(rf)
shap_values = explainer(X_test)

# Plot SHAP Summary
shap.plots.beeswarm(shap_values) 