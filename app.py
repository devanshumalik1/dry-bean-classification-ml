import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dry Bean Classification By Devanshu",
    page_icon="🌱𖠗",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Dry Bean Classification By Devanshu")
st.markdown(
    """
    ### Machine Learning Classification Dashboard

    This application demonstrates five classification models
    trained on the Dry Bean dataset.

    **Models implemented:**
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Gaussian Naive Bayes
    - Random Forest
    """
)


# ============================================================
# LOAD MODELS
# ============================================================

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
   # "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}


@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, filename in model_files.items():

        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            st.error(f"Model file not found: {path}")
            st.stop()

        loaded_models[model_name] = joblib.load(path)

    return loaded_models


@st.cache_resource
def load_label_encoder():

    return joblib.load(
        os.path.join(
            MODEL_DIR,
            "label_encoder.pkl"
        )
    )


@st.cache_data
def load_model_metrics():

    return pd.read_csv(
        os.path.join(
            MODEL_DIR,
            "model_metrics.csv"
        )
    )


models = load_models()
label_encoder = load_label_encoder()
model_metrics = load_model_metrics()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📊 Model Performance")

selected_metrics = model_metrics[
    model_metrics["ML Model"] == selected_model_name
].iloc[0]


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{selected_metrics['Accuracy']:.4f}"
    )

with col2:
    st.metric(
        "AUC",
        f"{selected_metrics['AUC']:.4f}"
    )

with col3:
    st.metric(
        "Precision",
        f"{selected_metrics['Precision']:.4f}"
    )


col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Recall",
        f"{selected_metrics['Recall']:.4f}"
    )

with col5:
    st.metric(
        "F1 Score",
        f"{selected_metrics['F1 Score']:.4f}"
    )

with col6:
    st.metric(
        "MCC",
        f"{selected_metrics['MCC']:.4f}"
    )


# ============================================================
# UPLOAD TEST DATA
# ============================================================

st.header("📁 Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.success("Test data uploaded successfully!")

    st.subheader("Uploaded Dataset Preview")

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )


    # ========================================================
    # CHECK TARGET COLUMN
    # ========================================================

    if "Class" not in test_data.columns:

        st.error(
            "The uploaded CSV must contain the 'Class' column "
            "containing the actual target values."
        )

    else:

        X_test = test_data.drop(
            columns=["Class"]
        )

        y_test_original = test_data["Class"]


        # ====================================================
        # ENCODE ACTUAL TARGET
        # ====================================================

        try:

            y_test = label_encoder.transform(
                y_test_original
            )

        except ValueError as e:

            st.error(
                "The uploaded data contains a class that was "
                "not present during model training."
            )

            st.stop()


        # ====================================================
        # PREDICTION
        # ====================================================

        y_pred = selected_model.predict(
            X_test
        )

        y_proba = selected_model.predict_proba(
            X_test
        )


        # ====================================================
        # CALCULATE METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        auc = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="weighted"
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )


        # ====================================================
        # DISPLAY TEST DATA METRICS
        # ====================================================

        st.header("📈 Evaluation on Uploaded Test Data")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

        with c2:
            st.metric(
                "AUC",
                f"{auc:.4f}"
            )

        with c3:
            st.metric(
                "Precision",
                f"{precision:.4f}"
            )


        c4, c5, c6 = st.columns(3)

        with c4:
            st.metric(
                "Recall",
                f"{recall:.4f}"
            )

        with c5:
            st.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

        with c6:
            st.metric(
                "MCC",
                f"{mcc:.4f}"
            )


        # ====================================================
        # PREDICTION RESULTS
        # ====================================================

        st.header("🔍 Prediction Results")

        predictions = pd.DataFrame({

            "Actual Class":
                y_test_original.values,

            "Predicted Class":
                label_encoder.inverse_transform(
                    y_pred
                )
        })

        st.dataframe(
            predictions,
            use_container_width=True
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.header("📊 Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(
            f"Confusion Matrix - {selected_model_name}"
        )

        st.pyplot(fig)


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.header("📋 Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )


else:

    st.info(
        "Please upload test_data.csv to generate predictions "
        "and display the evaluation results."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.header("🏆 Model Comparison")

comparison_columns = [
    "ML Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

comparison_df = model_metrics[
    comparison_columns
].copy()

st.dataframe(
    comparison_df.round(4),
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ML Classification Assignment | Dry Bean Dataset"
)
