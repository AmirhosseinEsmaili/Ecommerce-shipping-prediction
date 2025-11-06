# E-Commerce Shipment Delivery Forecast

## 1. Project Overview

This project analyzes an e-commerce shipping dataset to predict whether a shipment will arrive on time (`1`) or be delayed (`0`). In the logistics and e-commerce industry, on-time delivery is a critical KPI for customer satisfaction.

The primary business goal is to **identify the key drivers of shipment delays** and build a classification model that can accurately predict which shipments are at the highest risk of being late.

## 2. Dataset

The data is from the "Customer Analytics" dataset on Kaggle:
[https://www.kaggle.com/datasets/prachi13/customer-analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)

**Key Features:**
* `Warehouse_block`, `Mode_of_Shipment`, `Gender` (Nominal Categorical)
* `Product_importance` (Ordinal Categorical)
* `Customer_care_calls`, `Customer_rating`, `Cost_of_the_Product`, `Prior_purchases`, `Discount_offered`, `Weight_in_gms` (Numerical)
* **Target:** `Reached.on.Time_Y.N` (1 = On Time, 0 = Delayed)

## 3. Methodology

1.  **Exploratory Data Analysis (EDA):** I analyzed the data to find patterns, check for outliers (especially in `Discount_offered`), and visualize feature correlations with the target variable.
2.  **Data Preprocessing & Feature Engineering:**
    * One-Hot Encoded *nominal* features (`Warehouse_block`, `Mode_of_Shipment`, `Gender`).
    * Used Ordinal Mapping for the *ordinal* feature (`Product_importance` mapped to 0, 1, 2).
    * Engineered a `Discount_Log` feature to normalize the skewed `Discount_offered` column for linear and distance-based models.
    * **Critically, I performed a train-test split *before* scaling** to prevent any data leakage from the test set into the training process.
    * Applied `StandardScaler` to all numerical features for the linear and distance-based models (LogReg, KNN, SVM). Tree-based models used the unscaled data for better interpretability.
3.  **Model Building & Comparison:**
    * Established a **baseline model** using `LogisticRegression`.
    * Trained and compared a leaderboard of models: `K-Nearest Neighbors`, `SVM`, `Decision Tree`, `Random Forest`, and `XGBoost`.
4.  **Model Tuning & Selection:**
    * Identified that the dataset is imbalanced (more 'On Time' shipments than 'Late' ones). The primary business goal is to find the **late shipments (Class 0)**.
    * Tuned all models using hyperparameters like `class_weight='balanced'`, `max_depth`, and `scale_pos_weight` to improve their **Recall for Class 0**.

## 4. Model Results

The primary metric for success was **Recall on Class 0 (Late Shipments)**, as it is more costly for the business to *miss* a late shipment (a False Negative) than to *falsely flag* an on-time one (a False Positive).

Based on this, I built a leaderboard of all tuned models.

| Model | Accuracy | Recall (Class 0) | F1-Score (Macro) | AUC |
| :--- | :--- | :--- | :--- | :--- |
| Baseline LogReg (Balanced) | 66% | 93% | 0.66 | 0.736 |
| Tuned KNN (k=25) | 65% | 61% | 0.63 | 0.702 |
| Tuned SVM (Balanced) | 66% | 99% | 0.66 | 0.735 |
| Pruned Decision Tree (Balanced)| 68% | 98% | 0.68 | 0.730 |
| **Tuned Random Forest (Balanced)**| **68%** | **99%** | **0.67** | **0.736** |
| Tuned XGBoost (Balanced) | 66% | 83% | 0.66 | 0.718 |

<br>

The **Tuned Random Forest** (`max_depth=5`, `class_weight='balanced'`) was selected as the champion model. It has the joint-highest accuracy (68%), the best-in-class **Recall (99%)** for finding late shipments, and the highest AUC score, making it the most robust and specialized model for our business goal.

## 5. Key Findings & Insights

* The single most important feature for predicting a late shipment is **`Discount_offered`**.
* The Decision Tree analysis revealed that shipments with a discount **greater than 20%** were responsible for a huge portion of late deliveries.
* The second most important feature was `Weight_in_gms`, indicating that heavier shipments are also at higher risk.
* This suggests a business strategy: the company's policy of offering deep discounts may be tied to a lower-priority shipping tier that results in more delays.

## 6. How to Run

1.  Clone the repository.
2.  Install the required libraries:
    `pip install -r requirements.txt`
3.  Run the `E-Commerce_Shipment_Analysis.ipynb` Jupyter Notebook.
