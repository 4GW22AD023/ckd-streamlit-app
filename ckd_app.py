import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import pickle

# ✅ Load dataset
file_path = "Chronic_Kidney_Dsease_data.csv"
df = pd.read_csv(file_path)

# ------------------ Show raw data option ------------------
st.title("💉 Chronic Kidney Disease Prediction App")

if st.checkbox("📂 Show Raw Data"):
    st.write(df)

# Target column
target_col = 'Diagnosis'

# Features (drop ID, Doctor, target)
features = df.drop(columns=['PatientID', 'DoctorInCharge', target_col])
labels = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Preprocessing: numeric + categorical
numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Build pipeline with XGBoost
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)

st.success("✅ Model trained successfully!")

st.subheader("📊 Model Evaluation")
st.text(classification_report(y_test, y_pred))
st.write("ROC-AUC:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# Save model
with open("ckd_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# SHAP feature importance
explainer = shap.Explainer(
    pipeline.named_steps['classifier'],
    pipeline.named_steps['preprocessor'].transform(X_test)
)
shap_values = explainer(pipeline.named_steps['preprocessor'].transform(X_test))

st.subheader("🔑 Feature Importance (SHAP)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, pipeline.named_steps['preprocessor'].transform(X_test), show=False)
st.pyplot(fig)

# Prediction UI
st.subheader("🧪 Predict CKD for New Patient")

new_patient = {}
for col in features.columns:
    val = st.number_input(col, value=float(features[col].mean()))
    new_patient[col] = val

new_data = pd.DataFrame([new_patient])
prediction = pipeline.predict(new_data)[0]
probability = pipeline.predict_proba(new_data)[0][1]

st.write("Prediction Probability (CKD):", round(probability, 2))
st.write("Predicted Class:", "CKD" if prediction == 1 else "Not CKD")
