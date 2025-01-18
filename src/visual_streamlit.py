import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap
import warnings

warnings.filterwarnings('ignore')

# Load the saved model
model_data = joblib.load('../models/random_forest_model_vis.pkl')

# Extract the model and other necessary data
model = model_data['model']  
x_test = model_data['x_test']  
y_test = model_data['y_test']  
feature_names = model_data['feature_names']  

# Streamlit title
st.title("Bank Customer Churn Prediction")

# Sidebar for metric selection
st.sidebar.title("Visualization Options")
selected_metric = st.sidebar.selectbox(
    "Select Metric", ["Confusion Matrix", "Feature Importance", "ROC Curve"]
)

# Display the selected visualization
if selected_metric == "Confusion Matrix":
    st.header("Confusion Matrix")
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
elif selected_metric == "Feature Importance":
    st.header("Feature Importance")
    imp_features = pd.DataFrame({
        "Feature Name": feature_names,  
        "Importance": model.feature_importances_  
    })
    ft = imp_features.sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(15, 10))
    sns.barplot(x="Importance", y="Feature Name", data=ft, hue="Importance", palette="plasma")
    plt.title("Feature importance in Model Prediction", fontweight="black", size=20, pad=20)
    plt.yticks(size=12)
    st.pyplot(plt.gcf()) 
elif selected_metric == "ROC Curve":
    st.header("ROC Curve")
    y_probs = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)


    