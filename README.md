# Backorder_Prediction_Using_ML
This project builds an end-to-end machine learning pipeline to predict whether a product is likely to go on backorder, enabling proactive inventory planning and risk mitigation.
Backorders occur when customer demand exceeds available inventory, leading to delayed fulfillment, customer dissatisfaction, and operational inefficiencies.

The pipeline follows industry best practices and covers data preprocessing, feature engineering, model development, evaluation, and deployment-aware design, with a clear separation of stages to ensure reproducibility and scalability.

## Business Problem

### Backorders negatively impact:

- Customer satisfaction and retention
- Supply chain efficiency
- Revenue forecasting and operational planning

The goal of this project is to identify high-risk items before a backorder occurs, allowing businesses to take preventive actions such as stock reallocation, supplier adjustments, or demand planning.

## Analytical Approach

The project is structured as a modular machine learning pipeline, with each stage implemented in a separate notebook to ensure clarity, reproducibility, and ease of extension.

### Stage 1: Data Preprocessing & Feature Engineering

Notebook: Part_1_Data_Preprocessing.ipynb

- Analyzed the structure and quality of the inventory dataset to understand key drivers of backorders.

- Cleaned and prepared data by handling missing values, correcting data types, and validating feature consistency.

- Performed exploratory data analysis (EDA) to examine feature distributions, correlations, and class imbalance.

- Engineered relevant features to improve predictive performance while minimizing noise.

- Prepared a clean, model-ready dataset with proper train–test separation to prevent data leakage.

### Stage 2: Model Development

Notebook: Part_2_Model_Development.ipynb

- Developed multiple supervised machine learning models to predict backorders.

- Addressed class imbalance, a critical challenge in backorder prediction, to improve minority-class detection.

- Built reusable training pipelines using Python and scikit-learn.

- Performed hyperparameter tuning to improve model generalization.

- Compared model performance using consistent evaluation criteria.

#### Models Developed

- Logistic Regression (baseline model)

- Random Forest Classifier

- Gradient Boosting–based models (XGBoost-style boosting)

### Stage 3: Model Evaluation & Validation

Notebook: Part_3_Model_Evaluation.ipynb

##### Evaluated models using business-relevant metrics:
```
- Precision

- Recall

- F1-score

- ROC-AUC
```
- Analyzed confusion matrices to assess the cost of false negatives (missed backorders).

- Validated model robustness using a clean evaluation pipeline.

- Interpreted results to translate model outputs into actionable inventory insights.

- Prepared the pipeline for model persistence using pickle, supporting reproducibility and future deployment.
