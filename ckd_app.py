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
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --------------------------
# Load Dataset
# --------------------------
file_path = "Chronic_Kidney_Dsease_data.csv"
df = pd.read_csv(file_path)

target_col = "Diagnosis"
features = df.drop(columns=["PatientID", "DoctorInCharge", target_col], errors="ignore")
labels = df[target_col]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Preprocessing: numeric + categorical
numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

# Build pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# --------------------------
# Streamlit UI
# --------------------------
st.title("💉 Chronic Kidney Disease Prediction App")
st.success("✅ Model trained successfully!")

# --------------------------
# Model Evaluation
# --------------------------
st.subheader("📊 Model Evaluation")
st.text(classification_report(y_test, y_pred))
st.write("ROC-AUC:", round(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]), 3))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# Save model
with open("ckd_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
st.write("💾 Model saved as `ckd_model.pkl`")

# --------------------------
# Summary Statistics
# --------------------------
st.subheader("📈 Dataset Summary Statistics")
st.write(df.describe(include="all"))

# Average values for CKD vs Non-CKD
st.subheader("🧮 Average Feature Values: CKD vs Non-CKD")
avg_df = df.groupby("Diagnosis")[numeric_features].mean().reset_index()
st.write(avg_df)

# Interactive bar chart
feature_to_compare = st.selectbox("Select feature to compare averages", numeric_features, key="compare")
fig_bar = px.bar(
    avg_df,
    x="Diagnosis",
    y=feature_to_compare,
    color="Diagnosis",
    barmode="group",
    title=f"Average {feature_to_compare}: CKD vs Non-CKD"
)
st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------
# Charts
# --------------------------
st.subheader("📊 Visualizations")

# Histogram
feature_to_plot = st.selectbox("Select numeric feature for histogram", numeric_features, key="hist")
fig_hist = px.histogram(df, x=feature_to_plot, color="Diagnosis", marginal="box", nbins=30, barmode="overlay")
st.plotly_chart(fig_hist, use_container_width=True)

# Violin Plot
feature_to_violin = st.selectbox("Select numeric feature for violin plot", numeric_features, key="violin")
fig_violin = px.violin(df, x="Diagnosis", y=feature_to_violin, color="Diagnosis", box=True, points="all")
st.plotly_chart(fig_violin, use_container_width=True)

# Correlation heatmap
st.subheader("🧩 Correlation Heatmap")
corr = df[numeric_features].corr()
fig_heatmap, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", ax=ax)
st.pyplot(fig_heatmap)

# --------------------------
# SHAP Feature Importance
# --------------------------
st.subheader("🔑 Feature Importance (SHAP)")

# Transform test data
X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)
explainer = shap.Explainer(pipeline.named_steps["classifier"], X_test_transformed)
shap_values = explainer(X_test_transformed)

# Summary Plot
fig_shap = plt.figure()
shap.summary_plot(shap_values, X_test_transformed, show=False)
st.pyplot(fig_shap, bbox_inches="tight")

# --------------------------
# Predict CKD for New Patient
# --------------------------
st.subheader("🧪 Predict CKD for New Patient")
new_patient = {}
for col in features.columns:
    if col in numeric_features:
        val = st.number_input(col, value=float(features[col].mean()))
    else:
        val = st.selectbox(col, options=df[col].dropna().unique())
    new_patient[col] = val

new_data = pd.DataFrame([new_patient])
prediction = pipeline.predict(new_data)[0]
probability = pipeline.predict_proba(new_data)[0][1]

st.write("Prediction Probability (CKD):", round(probability, 2))
st.write("Predicted Class:", "CKD" if prediction == 1 else "Not CKD")
