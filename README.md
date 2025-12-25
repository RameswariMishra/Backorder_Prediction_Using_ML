# Backorder Prediction using Machine Learning Pipelines
Backorders occur when customer demand exceeds available inventory, leading to delayed fulfillment, reduced customer satisfaction, and operational inefficiencies.

This project implements an end-to-end machine learning pipeline to predict whether a product is likely to go on backorder. A central focus of the work is the systematic use of anomaly detection, feature selection, and supervised learning pipelines, designed and evaluated through multiple iterations to identify the most robust approach for noisy and highly imbalanced inventory data.

The final solution demonstrates how pipeline-based experimentation can support proactive inventory planning and risk mitigation.

## Business Problem

### Backorders negatively impact:

- Customer satisfaction and retention
- Supply chain efficiency
- Revenue forecasting and operational planning

The goal of this project is to identify high-risk items before a backorder occurs, allowing businesses to take preventive actions such as stock reallocation, supplier adjustments, or demand planning.

## Analytical Approach

The project follows a modular, pipeline-driven machine learning approach, where preprocessing, anomaly detection, feature selection, and classification are combined and evaluated systematically.

Multiple pipeline configurations were explored during experimentation. From these, the three best-performing pipelines were selected for detailed analysis and showcase based on their ability to balance minority-class recall (backorders) with overall prediction stability.

### Stage 1: Data Preprocessing & Feature Engineering
Notebook: Part_1_Data_Preprocessing.ipynb

- Analyzed the structure and quality of the inventory dataset to understand key drivers of backorders.

- Cleaned and prepared data by handling missing values, correcting data types, and validating feature consistency.

- Performed exploratory data analysis (EDA) to examine feature distributions, correlations, and class imbalance.

- Engineered relevant features to improve predictive performance while minimizing noise.

- Prepared a clean, model-ready dataset with proper train–test separation to prevent data leakage.

### Stage 2: Pipeline Strategy & Outlier Handling
Notebook: Part_2_Model_Development.ipynb
Inventory datasets often contain anomalous and noisy records due to data entry errors, rare demand spikes, or supply chain disruptions. To mitigate their impact, outlier detection was treated as an explicit preprocessing stage in the pipeline design.

#### Strategy:

- Outlier detection models were fitted only on the training dataset to avoid data leakage.

- Detected outliers were removed, producing an inlier-only training dataset.

- Supervised learning pipelines were trained exclusively on this inlier dataset.

- The held-out test set remained untouched until final evaluation.

Several combinations of outlier detection algorithms, feature selection techniques, and classifiers were tested iteratively. The three strongest pipelines are documented below.

##### Pipeline 1: Isolation Forest + PCA + SVC
```
Outlier Detection: Isolation Forest
Detects anomalies by isolating observations using random partitioning, effective for high-dimensional data.

Feature Selection / Transformation: Principal Component Analysis (PCA)
Applied after feature scaling to reduce dimensionality, capture informative components, and mitigate noise and multicollinearity.

Classification Model: Support Vector Classifier (SVC) with RBF kernel
Models nonlinear decision boundaries in the transformed feature space.

This pipeline served as a strong nonlinear baseline under anomaly-aware preprocessing.
```
##### Pipeline 2: One-Class SVM + Factor Analysis + Logistic Regression
```
Outlier Detection: One-Class Support Vector Machine (One-Class SVM)
Learns a boundary around normal inventory records and flags deviations as anomalies.

Feature Selection / Transformation: Factor Analysis (FA)
Extracts latent factors that explain observed correlations among features, reducing dimensionality while preserving interpretability.

Classification Model: Logistic Regression
Provides an interpretable linear baseline for backorder prediction.

This pipeline emphasized interpretability and dimensionality reduction, making it suitable for explainable inventory analytics.
```
##### Pipeline 3: Elliptic Envelope + RFE + Random Forest (Best Performing)
```
Outlier Detection: Elliptic Envelope
Assumes a multivariate Gaussian distribution and detects anomalies based on covariance structure.

Feature Selection: Recursive Feature Elimination (RFE)
Iteratively removes less important features using a supervised estimator to retain the most predictive subset.

Classification Model: Random Forest Classifier
Captures nonlinear relationships and feature interactions using an ensemble of decision trees.

This pipeline demonstrated the strongest balance between robustness, predictive power, and stability.
```
### Stage 3: Model Evaluation & Validation

Notebook: Part_3_Model_Evaluation.ipynb

##### Evaluated models using business-relevant metrics:
```
Precision

Recall

F1-score

ROC-AUC
```
- Analyzed confusion matrices to assess the cost of false negatives (missed backorders).

- Validated model robustness using a clean evaluation pipeline.

- Interpreted results to translate model outputs into actionable inventory insights.

- Prepared the pipeline for model persistence using pickle, supporting reproducibility and future deployment.
