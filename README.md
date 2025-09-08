# Chronic Kidney Disease Prediction App

## 1. Objective

This project aims to build a predictive system for early detection of Chronic Kidney Disease (CKD). It helps in real-world healthcare by providing doctors and patients with a tool that assists in early diagnosis, improving treatment outcomes.

## 2. Data Source

The dataset used is the **Chronic\_Kidney\_Dsease\_data.csv**, containing 50+ medical, lifestyle, and demographic features. It includes parameters like age, gender, blood pressure, serum creatinine, BUN levels, protein in urine, cholesterol, and lifestyle factors. The target variable is `Diagnosis` (1 = CKD, 0 = Not CKD).

## 3. Methodology

* Data preprocessing using **scikit-learn** pipelines.
* Handling numerical and categorical features with scaling and one-hot encoding.
* Model training and evaluation using classification metrics.
* SHAP (SHapley Additive exPlanations) for feature importance.
* Deployment using **Streamlit** for an interactive web app.

## 4. Model Used

* **XGBoost Classifier**: A powerful gradient boosting model chosen for its high performance and ability to handle tabular healthcare data effectively.

## 5. Results & Findings

* **Accuracy:** 91%
* **ROC-AUC:** 0.73
* The model performs well in detecting CKD but shows slight misclassification for healthy patients due to class imbalance.
* **Key Features Influencing CKD:** Serum Creatinine, BUN Levels, GFR, Protein in Urine, and Blood Pressure.

The model is saved as `ckd_model.pkl` and can be used for predictions in real-time applications.
