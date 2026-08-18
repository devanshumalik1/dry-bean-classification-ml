# dry-bean-classification-ml
# Dry Bean Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for classifying different varieties of dry beans based on their geometric and morphological characteristics.

Five classification algorithms were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)

The models were evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The trained models were then integrated into an interactive Streamlit web application that allows users to upload test data, select a machine learning model, view predictions, and evaluate model performance.

---

## 2. Dataset Description

### Dataset Name - **Dry Bean Dataset**

### Dataset Source

The Dry Bean Dataset is a publicly available classification dataset containing measurements of dry bean images.

### Dataset Characteristics

* Number of instances: **13,611**
* Number of input features: **16**
* Number of target classes: **7**
* Target variable: **Class**
* Problem type: **Multiclass Classification**

The seven bean classes are:

* BARBUNYA
* BOMBAY
* CALI
* DERMASON
* HOROZ
* SEKER
* SIRA

### Features

The dataset contains the following morphological and geometric features:

* Area
* Perimeter
* MajorAxisLength
* MinorAxisLength
* AspectRation
* Eccentricity
* ConvexArea
* EquivDiameter
* Extent
* Solidity
* roundness
* Compactness
* ShapeFactor1
* ShapeFactor2
* ShapeFactor3
* ShapeFactor4

### Data Preprocessing

The target class was encoded using `LabelEncoder`.

The dataset was divided into training and testing sets using an **80:20 stratified train-test split**.

Feature scaling using `StandardScaler` was applied to the models where it was appropriate, particularly Logistic Regression and K-Nearest Neighbors.

---

## 3. GitHub Repository Link

**GitHub Repository:** - https://github.com/devanshumalik1/dry-bean-classification-ml

The repository contains:

* `app.py` — Streamlit application
* `requirements.txt` — Python dependencies
* `README.md` — Project documentation
* `test_data.csv` — Test dataset used for Streamlit evaluation
* `model/` — Saved machine learning models and evaluation metrics
* `model_training.ipynb` — Model development and evaluation notebook

---

## 4. Models Used

### 4.1 Logistic Regression

Logistic Regression was implemented as a baseline classification model. StandardScaler was used before model training to normalize the feature values.

### 4.2 Decision Tree Classifier

A Decision Tree classifier was implemented to model nonlinear relationships between the input features and the bean classes. Decision Trees do not require feature scaling.

### 4.3 K-Nearest Neighbors

KNN classifies observations based on the nearest training samples. StandardScaler was used because KNN is sensitive to the scale of input features.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes was used as a probabilistic classification model. It assumes that the continuous input features follow Gaussian distributions within each class.

### 4.5 Random Forest Classifier

Random Forest is an ensemble learning method consisting of multiple decision trees. The predictions from the individual trees are combined to obtain the final classification.

---

## 5. Model Evaluation

The following evaluation metrics were calculated for every model:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model                 |    Accuracy |         AUC |   Precision |      Recall |    F1 Score |         MCC |
| ------------------------ | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| Logistic Regression      | **0.9207**  | **0.9934**  | **0.9215**  | **0.9207**  | **0.9209**  | **0.9042**  |
| Decision Tree            | **0.892**   | **0.9334**  | **0.8917**  | **0.892**   | **0.8916**  | **0.8696**  |
| KNN                      | **0.9166**  | **0.9812**  | **0.9174**  | **0.9166**  | **0.9168**  | **0.8992**  |
| Naive Bayes              | **0.7639**  | **0.9644**  | **0.7654**  | **0.7639**  | **0.7615**  | **0.7154**  |
| Random Forest (Ensemble) | **0.9214**  | **0.9921**  | **0.9215**  | **0.9214**  | **0.9214**  | **0.9049**  |

---

## 6. Observations on Model Performance

| ML Model                     | Observation about Model Performance                                                                                                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression**      | Logistic Regression provides a strong baseline for the multiclass classification problem. Its performance demonstrates how effectively a linear model can classify the bean varieties based on the available features.   |
| **Decision Tree**            | The Decision Tree is capable of capturing nonlinear relationships between the morphological features and bean classes. Its performance is competitive, although individual tree models can be more prone to overfitting. |
| **KNN**                      | KNN performs classification based on the similarity between observations. Feature scaling is important for this model because distance calculations are affected by feature magnitudes.                                  |
| **Naive Bayes**              | Gaussian Naive Bayes provides a computationally efficient probabilistic approach. Its performance may be affected by the assumption that features are conditionally independent and approximately Gaussian.              |
| **Random Forest (Ensemble)** | Random Forest combines predictions from multiple decision trees and generally provides robust performance by reducing the variance associated with a single decision tree.                                               |
| **Overall Winner**           | **Random Forest** achieved the best overall performance based on the comparison of the evaluation metrics.                                                                                                      |

### Overall Performance

Based on the calculated evaluation metrics, **Random Forest** was identified as the overall winner for this dataset.

The model was selected based on its overall performance across Accuracy, AUC, Precision, Recall, F1 Score, and MCC rather than relying on a single evaluation metric.

---

## 7. Streamlit Application

An interactive Streamlit application was developed to demonstrate the trained classification models.

### Features of the Application

The Streamlit application provides:

1. **Test Dataset Upload**

   * Users can upload a CSV test dataset.

2. **Model Selection**

   * Users can select one of the implemented classification models from a dropdown menu.

3. **Evaluation Metrics**

   * Accuracy
   * AUC
   * Precision
   * Recall
   * F1 Score
   * MCC

4. **Prediction Results**

   * The application displays the actual and predicted bean classes.

5. **Confusion Matrix**

   * A confusion matrix is generated for the selected model.

6. **Classification Report**

   * Precision, recall and F1-score are displayed for individual classes.

7. **Model Comparison**

   * A comparison table containing the evaluation metrics of all implemented models is displayed.

---

## 8. Live Streamlit Application

**Live Application:** - https://dry-bean-classification-ml.streamlit.app/

---

## 9. Repository Structure

```text
dry-bean-classification-ml/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_training.ipynb
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── label_encoder.pkl
    └── model_metrics.csv
```

---

## 10. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* Google Colab
* GitHub
* Streamlit Community Cloud

---

## 11. Deployment

The Streamlit application was deployed using **Streamlit Community Cloud**.

The application source code and saved machine learning models are maintained in the GitHub repository.

The deployed application loads the saved models and allows users to upload test data and interactively evaluate the selected classification model.

---

## 12. Conclusion

This project demonstrates an end-to-end machine learning classification workflow, starting from dataset preparation and preprocessing, followed by model training and evaluation, and finally deployment through an interactive Streamlit application.

The comparison of multiple classification algorithms provides insight into how different machine learning approaches perform on the Dry Bean multiclass classification problem.

The project also demonstrates the practical deployment workflow of saving trained models, maintaining source code through GitHub, and making the machine learning solution accessible through a web-based interface.
