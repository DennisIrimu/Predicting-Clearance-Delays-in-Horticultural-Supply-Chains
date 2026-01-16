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

* In-Scope:
  - Loading and basic exploration of the  dataset
  - Data structure & quality check (df.info(), df.describe())
  - Statistical summary of numerical features
  - Training a baseline Random Forest Classifier to predict delayed_flag
  - Automatic categorical & numerical preprocessing via Pipeline
  - Model evaluation via classification report and confusion matrix
  - Visualization of precision/recall/F1-score and confusion matrix heatmap

* Out-of-Scope :
  - Feature engineering from nested columns (documents, events)
  - Regression modeling
  - Hyperparameter tuning or cross-validation
  - Additional algorithms (XGBoost, etc.)
  - Model deployment or dashboard

## 3.0 Data Understanding

**3.1 Dataset Overview**

The dataset consists of 5,000 synthetic but realistic international consignments, stored in JSON Lines format .  
Each record represents a single shipment (mostly air freight of perishable goods such as fruits and cut flowers) and contains information about origin/destination, commodity, weight, document status, congestion, processing times, and whether the shipment was delayed beyond its SLA.

File: tlip_like_consignments_5000.jsonl  
Records: 5,000  
Columns: 27  
No missing values across the entire dataset

**3.2 Loading the Data**

The dataset is loaded into a pandas DataFrame using:


df = pd.read_json("tlip_like_consignments_5000.jsonl", lines=True)
df.head(5)

**Primary anticipated sources include:**
- KenTrade Single Window (TradeNet) — clearance times, permit records
- KNBS (Kenya National Bureau of Statistics) — monthly/quarterly export statistics
- AFA Horticultural Crops Directorate & KEPHIS — export volumes & phytosanitary certificates
- External event data — weather records, port/airport disruptions, freight cost trends


### 3.3 Variable Description

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

## 4.0 Data Cleaning & Pre-processing

#### 4.1 Handling Missing Values
- The notebook runs `df.info()`
- Result: All 27 columns have 5000 non-null values → **no missing data**
- No imputation, dropping, or filling of missing values is performed (none needed)
  

###  5.0 Descriptive Analytics - Pre-Modeling Analysis

Quick summary of delay patterns across 5,000 consignments before modeling.

**5.1. Overall Delay Landscape**
- Total: 5,000 consignments
- Delayed: 2,419 (48.38%)
- On time: 2,581 (51.62%)
- Avg total processing: 98.6 hours
- Avg delay (when delayed): 17.65 hours
- Long tail: 5% exceed 155 hours, 1% exceed 173 hours


**5.2. Overall Delay Landscape**
- Total consignments: **5,000**
- Delayed: **2,419** (48.38%)
- On time: **2,581** (51.62%)
- Average total processing time: **98.6 hours** (median 96.8h)
- Long tail: 5% exceed **155h**, 1% exceed **173h**, max **201h**
- Average delay (when delayed): **17.65 hours** (P75: 25.55h, max 77h)

**5.3. Key Patterns by Origin, Destination, Ports & Time**
- **Highest risk origins**: Rwanda (52%), Uganda (50%), Ethiopia (48%)
- **Highest risk destinations**: Belgium & Germany (51%)
- **Top ports**: Kigali (origin 52%), Felixstowe (destination 53%)
- **Weekend creation penalty**: 69% delay rate (vs 40% weekdays)
- **Night creation** (00:00–05:59): slightly higher risk (~50%)

**5.4 Commodity-Level Insights**
- **Highest delay** (mostly SEA): Tea (60%), Vegetables Mix (54%), Pineapples (52%)
- **Lowest delay** (all AIR): Fresh Beans (41%), Mangoes (43%), Cut Flowers/Herbs (~44%)
- SEA shipments consistently 2–3× more delayed than AIR


### 5.6. Early Red-Flag Signals (Pre-Clearance Predictors)

**5.6.1 Document Completeness Score**  
- Very High (0.9–1.0): **38%** delay rate, 93.18h avg processing  
- High (0.7–0.9): **63%** delay rate, 106.10h avg processing  
→ Higher completeness → much lower risk (strong negative correlation)

**5.6.2 Number of Document Amendments**  
- 0 amendments: **37%** delay rate  
- 1 amendment: **58%**  
- 2 amendments: **78%**  
- 3+ amendments: **93%** delay rate  
→ Each amendment is a major red flag (positive correlation)

**5.6.3 Congestion Index**  
- Low (0–0.3): **26%** delay rate  
- Medium (0.3–0.5): **46%**  
- High (0.5–0.7): **65%**  
- Very High (0.7–1.0): **78%** delay rate  
→ Strongest single early predictor (positive correlation)

**5.6.4 Weight & Value Quartiles**  
- Heavier/higher-value shipments: slightly higher delay risk (49–51%)

**5.6.5 Correlation Summary** (Early Flags vs Delay)  
- doc_completeness_score: **−0.3361** (strong negative)  
- doc_amendments: **+0.3012**  
- congestion_index: **+0.3793** (highest among early flags)

**5.6.6 Visual Highlights**
- Pie chart: overall delay split (48.4% delayed)
- Bar charts: delay rates by origin/destination, commodity, port, time bucket, day of week, document completeness, amendments, congestion, weight/value
- Histograms & box plots: processing time & delay distributions
- Scatter plots: customs vs dwell time (bottleneck detection), processing time vs delay rate by country
- Trend line: document completeness vs delay (clear downward slope)

**5.6.7 Final Key Takeaways for Modeling**
- **Strongest early red flags**: high congestion, many document amendments, low document completeness  
- **Highest risk profile**: Weekend-created SEA shipments of Tea/Vegetables/Pineapples from Rwanda to Belgium/Germany  
- **Weekend penalty**: nearly doubles delay risk  
- **AIR vs SEA**: AIR shipments are far more reliable  
- Nearly half of consignments delayed — balanced and realistic classification problem



