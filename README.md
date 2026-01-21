# Project Title: Predictive Analytics for Optimizing "Time-to-Export" Lead Times in Kenya’s Horticultural Supply Chain
![horticulture](https://github.com/user-attachments/assets/ae3e540a-07e4-4602-b5e6-1f6a71e76238)

## 1.0 Introduction

This project develops a machine learning solution to predict the **time-to-export** (total lead time from packhouse/farm to final departure at Jomo Kenyatta International Airport — JKIA or Mombasa Port) for Kenya’s horticultural exports — primarily cut flowers, fruits, and vegetables.

The goal is to help exporters anticipate delays, reduce post-harvest losses, improve cold-chain planning, negotiate better freight contracts, and enhance overall supply chain resilience in one of Kenya’s most critical foreign exchange earning sectors.

## 2.0 Research Problem & Objectives

### Problem Statement

Kenya’s horticulture sector faces chronic and worsening logistics challenges including:

- Severe airfreight capacity shortages and skyrocketing costs at JKIA
- Frequent customs clearance delays and electronic system failures
- Port congestion and slow turnaround at Mombasa
- External disruptions (weather events, strikes, global shipping crises, ad hoc levies)

These bottlenecks result in substantial spoilage (especially for highly perishable products), missed market windows (particularly in Europe), and multi-billion-shilling revenue losses annually.

Currently, most exporters rely on experience-based estimates rather than reliable, data-driven predictions.

### Main Objective

To develop a robust, accurate machine learning regression model that predicts total export lead time (in hours or days) for Kenyan horticultural shipments, enabling proactive planning and risk mitigation.

### Specific Objectives

-Load and inspect the 5,000-record consignment dataset
- Perform initial data exploration and quality checks
- Conduct descriptive analytics: overall delay rate, processing times, and key distributions
- Identify delay patterns by origin/destination, commodity, time/day, and ports
- Uncover early red-flag signals (document completeness, amendments, congestion)
- Build and evaluate a baseline Random Forest Classifier for delay prediction
- Visualize insights and model performance (classification report, confusion matrix)

### Stakeholders

* International freight forwarders
* Customs & port operators
* Exporters of perishable commodities
* Supply chain analytics teams

### Project Scope

- Load and inspect the 5,000-record consignment dataset
- Perform exploratory data analysis (EDA) and quality checks
- Conduct descriptive analytics (delay rate, processing times, key distributions)
- Identify delay patterns by origin/destination, commodity, time/day, and ports
- Detect early red-flag signals (document completeness, amendments, congestion)
- Build and evaluate baseline models (Logistic Regression, Random Forest, Decision Tree, XGBoost, LightGBM)
- Apply hyperparameter tuning and stratified cross-validation
- Handle class imbalance using class weights
- Compare models using classification metrics, confusion matrices, ROC curves
- Perform model iteration (Ridge, Lasso, ElasticNet Logistic Regression)
- Select features and re-evaluate models with reduced feature sets
- Diagnose best model and interpret feature importance
-Translate findings into operational insights for risk reduction

## 3.0 Data Understanding

### Dataset Overview

The dataset consists of 5,000 synthetic but realistic international consignments, stored in JSON Lines format .  
Each record represents a single shipment (mostly air freight of perishable goods such as fruits and cut flowers) and contains information about origin/destination, commodity, weight, document status, congestion, processing times, and whether the shipment was delayed beyond its SLA.


**Primary anticipated sources include:**
- KenTrade Single Window (TradeNet) — clearance times, permit records
- KNBS (Kenya National Bureau of Statistics) — monthly/quarterly export statistics
- AFA Horticultural Crops Directorate & KEPHIS — export volumes & phytosanitary certificates
- External event data — weather records, port/airport disruptions, freight cost trends


### Loading the Data

The dataset is loaded into a pandas DataFrame using:

df = pd.read_json("tlip_like_consignments_5000.jsonl", lines=True)
df.head(5)

File: tlip_like_consignments_5000.jsonl  
Records: 5,000  
Columns: 27  
No missing values across the entire dataset

### Variable Description

| Column                     | Type          | Description                          |
|----------------------------|---------------|--------------------------------------|
| consignment_id             | string        | Unique ID                            |
| created_at                 | datetime      | Creation timestamp                   |
| origin_country             | string        | Origin country                       |
| destination_country        | string        | Destination country                  |
| origin_port                | string        | Origin port                          |
| destination_port           | string        | Destination port                     |
| shipment_mode              | string        | AIR or SEA                           |
| commodity                  | string        | Product type                         |
| hs_code                    | integer       | HS code                              |
| gross_weight_kg            | float         | Weight (kg)                          |
| declared_value_usd         | float         | Value (USD)                          |
| doc_completeness_score     | float         | Doc readiness (0-1)                  |
| missing_docs_proxy         | integer       | Missing docs count                   |
| doc_amendments             | integer       | Doc amendments                       |
| congestion_index           | float         | Congestion level (0-1)               |
| is_weekend_created         | integer       | Weekend flag                         |
| customs_release_hours      | float         | Customs time (h)                     |
| terminal_dwell_hours       | float         | Terminal time (h)                    |
| sla_hours                  | integer       | SLA target (h)                       |
| total_processing_hours     | float         | Total time (h)                       |
| delayed_flag               | integer       | Delayed? (1=yes)                     |
| delay_hours                | float         | Delay hours                          |
| documents                  | list of dicts | Documents info                       |
| events                     | list of dicts | Events timeline                      |

## 4.0 Exploratory Data Analysis (EDA)

### Objectives:
- Explore distributions of key numeric features.
- Identify factors contributing to shipment delays.
- Validate assumptions before modeling.

  
**Key Analysis:**
- Numeric Features: Most shipments have high doc_completeness_score and low doc_amendments, while congestion_index shows variability.
**Congestion vs Delay: Higher congestion is associated with delays.**
  
<img width="532" height="410" alt="image" src="https://github.com/user-attachments/assets/f8de8045-92cc-4232-9da9-89212bd2b191" />

**Overall Delay: ~35% of shipments were delayed, ~65% on time.**

<img width="683" height="497" alt="image" src="https://github.com/user-attachments/assets/6bf0eabe-b96a-4ede-96e1-1ec950901413" />

**Shipment Mode vs Delay: Certain modes experience more delays; e.g., Air shipments delayed ~20%, Sea shipments delayed ~50%.**
  
<img width="679" height="215" alt="image" src="https://github.com/user-attachments/assets/686b113e-8fba-4d9b-862c-b4b6e557c845" />

## 5.0 Data Preparation
Data preparation focused on organizing features, splitting the dataset, and applying preprocessing pipelines to ensure clean inputs for modeling while preventing data leakage.

### Actions:

1. Feature Grouping:
- Numeric Features: gross_weight_kg, declared_value_usd, doc_completeness_score, missing_docs_proxy, doc_amendments, congestion_index
- Categorical Features: shipment_mode, commodity, hs_code, origin_country, destination_country, exporter_profile
-Binary Features: is_weekend_created

2.Target Variable: delayed_flag (0 = not delayed, 1 = delayed)

3.Train–Test Split:
- 80% training, 20% testing
-Stratified to preserve the proportion of delayed vs non-delayed shipments

4.Preprocessing Pipeline:
- Numeric: Standard scaled (StandardScaler)
- Binary: Passed through unchanged
- Categorical: One-hot encoded (OneHotEncoder) with handle_unknown='ignore' to accommodate unseen categories in the test set


## 6.0 Modeling 
Several models were evaluated to predict shipment delays, chosen based on the data structure and feature types:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

### Modeling Approach:
- Hyperparameter tuning via GridSearchCV with stratified 5-fold cross-validation.
- Class imbalance handled using class weights where applicable.
- Preprocessing pipeline applied to prevent data leakage (numeric scaling, binary passthrough, categorical one-hot encoding).
- Evaluation metrics included Accuracy, Precision, Recall, F1-score, and ROC-AUC.

| Model               | Accuracy | Precision | Recall | F1    | ROC_AUC |
| ------------------- | -------- | --------- | ------ | ----- | ------- |
| Logistic Regression | 0.813    | 0.809     | 0.804  | 0.806 | 0.898   |
| XGBoost             | 0.806    | 0.813     | 0.779  | 0.795 | 0.890   |
| Random Forest       | 0.797    | 0.800     | 0.775  | 0.787 | 0.877   |
| LightGBM            | 0.795    | 0.794     | 0.779  | 0.786 | 0.888   |
| Decision Tree       | 0.777    | 0.810     | 0.705  | 0.754 | 0.853   |

Results:
- Logistic Regression achieved the best overall balance (highest F1 and ROC-AUC).
- Tree-based models (XGBoost, Random Forest, LightGBM) performed competitively, but slightly lower.
- Decision Tree was weakest, showing lower recall and F1.

## 7.0 Hyperparameter Tuning
 Hyperparameter tuning was performed to optimize performance.

Tuned parameters included:
- Number of trees (n_estimators)
- Tree depth (max_depth)
- Minimum samples per split and leaf(min_samples_split, min_samples_leaf)
- Feature selection strategy (max_features)

**Tuning balanced:**
- Predictive performance
- Generalization
- Computational efficiency

## 8.0 Model Evaluation.
Evaluation Focus:
- Compare all models both numerically and visually.
- Select the best-performing model based on classification metrics.
- Examine error trade-offs using confusion matrices.
- Assess discrimination ability via ROC curves.
- Explain model predictions using feature importance.

  **ROC-AUC Performance:**

| Model               | ROC_AUC |
| ------------------- | ------- |
| Logistic Regression | 0.898   |
| XGBoost             | 0.890   |
| LightGBM            | 0.888   |
| Random Forest       | 0.877   |
| Decision Tree       | 0.853   |
  
<img width="691" height="351" alt="image" src="https://github.com/user-attachments/assets/c9c2d20a-edc4-406a-bf54-b656f8178f04" />          <img width="630" height="505" alt="image" src="https://github.com/user-attachments/assets/2674e795-1168-4605-b44c-ff3f12fd9b89" />

## 9.0 Best Model Diagnostics.

The model with the highest F1-score was chosen as the best, balancing precision and recall due to potential class imbalance.

Logistic Regression achieved the best F1-score of 0.81.

### Confusion Matrix:
- A confusion matrix was generated to examine true positives, true negatives, false positives, and false negatives.
- This highlights the error trade-offs and provides insight into which classes are more likely to be misclassified.

<img width="492" height="405" alt="image" src="https://github.com/user-attachments/assets/eb81dbfc-33bc-42d3-a46f-3cfc96ba5993" />

### Feature Importance / Coefficients:
- For Logistic Regression, the absolute values of coefficients were used to rank feature importance.
- The top 15 features influencing predictions were plotted, showing which shipment attributes most strongly affect the likelihood of delay.
- This helps explain model behavior and informs potential operational interventions.
  
<img width="652" height="392" alt="image" src="https://github.com/user-attachments/assets/c8883cbc-4960-4421-bfda-6ab42c3e4162" />


## 10.0 Model Iteration: Ridge, Lasso, & Elastic Net Logistic Regression
After confirming baseline Logistic Regression as the best model, regularized variants were tested to optimize performance and identify key predictive features.

**Variants Evaluated:**
- Logistic (Unregularized)
- Logistic Ridge (L2)
- Logistic Lasso (L1)
- Logistic Elastic Net

**Approach::**
- Hyperparameter tuning (C, l1_ratio) via GridSearchCV with stratified 5-fold CV
- Preprocessing via pipeline (scaling numeric, passing binary, one-hot encoding categorical)
-Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

| Model                    | Accuracy | F1    | ROC_AUC |
| ------------------------ | -------- | ----- | ------- |
| Logistic (Unregularized) | 0.813    | 0.806 | 0.898   |
| Logistic Ridge (L2)      | 0.813    | 0.806 | 0.898   |
| Logistic Lasso (L1)      | 0.813    | 0.806 | 0.898   |
| Logistic Elastic Net     | 0.813    | 0.806 | 0.898   |

### Feature Importance:

- Top drivers of delay: is_weekend_created, congestion_index, missing_docs_proxy, doc_amendments
- shipment_mode_AIR reduces likelihood of delay
- Horizontal bar plots visualize top 15 features; red = increases delay, green = reduces delay

<img width="696" height="543" alt="image" src="https://github.com/user-attachments/assets/70298c9f-7401-4eb1-bf71-258ee759d02e" />


## 11.0 Model Re-Evaluation Using Selected Features:

Re-assess model performance using only the selected features identified by the Elastic Net logistic regression model.
This ensures focus on the most influential predictors and tests whether tree-based models can benefit from reduced feature space.

**Perfomance Using Selected Features**
| Model         | Accuracy | Precision | Recall | F1    | ROC_AUC |
| ------------- | -------- | --------- | ------ | ----- | ------- |
| Decision Tree | 0.733    | 0.727     | 0.717  | 0.722 | 0.733   |
| Random Forest | 0.799    | 0.802     | 0.777  | 0.789 | 0.878   |
| XGBoost       | 0.788    | 0.789     | 0.767  | 0.778 | 0.871   |
| LightGBM      | 0.797    | 0.801     | 0.773  | 0.787 | 0.888   |

**Insights from Features:**

- Operational Timing & Congestion: Weekend shipments and high congestion drive delays.
- Shipment Mode: AIR reduces risk; SEA increases risk.
- Documentation: Missing or amended documents increase likelihood of delay.
- Commodities & HS Codes: Some commodities and codes slightly increase or reduce risk.

## Conclusion:

- The best-performing model is Elastic Net logistic regression, effectively combining L1 feature selection and L2 coefficient stability.
- It outperforms tree-based models on this dataset while maintaining strong interpretability.
- Model coefficients provide direct insights into critical risk factors, enabling informed operational decisions.

**Operational Implications:**

- Prioritize monitoring high-risk shipments, especially those on weekends or passing through congested points.
- Ensure document completeness to minimize delays.
- Use insights from the model to guide resource allocation and process improvements for smoother operations.

**Business Value:**
Logistic regression offers interpretability advantage, allowing direct insights into feature effects compared to black-box ensembles.

## 12.0 Model Deployment

- The final Random Forest model was deployed as a web-based application for real-time shipping delay prediction.
- Users input shipment details and receive immediate delay predictions using the same preprocessing pipeline applied during training, demonstrating a transition from model development to practical deployment.

link: https://predictive-horticulture.streamlit.app/


## Tools & Technologies

-Programming & Environment: Python, Jupyter Notebook
- Data Manipulation: Pandas, NumPy
- Machine Learning & Modeling: Scikit-learn, XGBoost, LightGBM
- Preprocessing & Pipelines: Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder
- Model Evaluation & Metrics: GridSearchCV, StratifiedKFold, classification_report, confusion_matrix, accuracy, precision, recall, F1-score, ROC-AUC
- Visualization: Matplotlib, Seaborn

## Contributors
- **David Munyiri* -Scrumaster
- **Dennis Irimu*
- **Lydiah Onkundi*
- **Eugene Kiprop*
- **James Ouma*
- **Cindy Achieng*
- **Barclay Koin*



