

# Donor Model Training

## 📌 Project Overview

This project focuses on building **machine learning models** to analyze donor data and predict donation behaviors.
The primary objectives are:

1. Identify **non-donors** (who does not donate).
2. Determine **top donors** and donation trends.
3. Achieve at least **85% prediction accuracy** for donation behavior.
4. Analyze **donation statistics** (maximum and average donations).
5. Evaluate whether the **dataset is balanced and sufficient** for model training.
6. Provide actionable insights for improving donation strategies.

The project uses **Random Forest** and **Neural Network** models with class balancing via **SMOTE** to handle data imbalance.

---

## 🗂 Dataset Details

The dataset (`donor_data.csv`) contains information about individuals and their donation history.

### Key Columns:

| Column Name            | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `TARGET_B`             | Donation flag (1 = Donor, 0 = Non-Donor)              |
| `LIFETIME_GIFT_AMOUNT` | Total amount donated by an individual                 |
| `URBANICITY`, `SES`    | Demographic and socio-economic indicators             |
| `CONTROL_NUMBER`       | Unique identifier for records (removed)               |
| `TARGET_D`             | Target donation amount (many missing values, removed) |

### Dataset Balance:

* **Non-Donors:** 14,529
* **Donors:** 4,843

> **Observation:**
> The dataset is **imbalanced**, requiring **SMOTE** or collection of more donor data.

---

## ⚙️ Installation & Requirements

Before running the code, install the necessary dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow matplotlib seaborn
```

---

## 🧹 Data Preprocessing

Steps performed before training:

1. **Handling Missing Values:**

   * Replaced `?` and blank spaces with `NaN`.
   * Filled missing numerical values with **median**.
   * Filled missing categorical values with **mode**.

2. **Dropped Columns:**

   * `CONTROL_NUMBER` (identifier)
   * `TARGET_D` (too many missing values)

3. **Encoded Categorical Features:**

   * Used `LabelEncoder` for categorical columns such as `URBANICITY`, `SES`, etc.

4. **Feature Scaling:**

   * Applied `StandardScaler` to normalize numeric features.

5. **SMOTE Oversampling:**

   * Balanced classes before training models.

---

## 🤖 Machine Learning Models

### 1. **Random Forest (Donation Prediction)**

* Used for predicting whether a person will **donate**.
* Tuned hyperparameters:

  * `n_estimators=800`
  * `max_depth=30`
  * `class_weight='balanced'`

**Accuracy Achieved:** 74.71% (below 85% requirement)

---

### 2. **Neural Network (Donation Prediction)**

* 3-layer neural network with dropout regularization.

**Accuracy Achieved:** 66.66% (below 85% requirement)

---

### 3. **Random Forest (Generous Donors)**

* Identifies **top 20% donors** based on `LIFETIME_GIFT_AMOUNT`.

---

### 4. **Neural Network (Generous Donors)**

* Neural network model specifically for predicting generous donors.

---

## 📊 Key Results

| **Question**                      | **Answer**                                      |
| --------------------------------- | ----------------------------------------------- |
| **1. Who doesn’t donate?**        | Non-Donors: **14,529**, Donors: **4,843**       |
| **2. Who donates the most?**      | Top Donor ID: **878**, Amount: **2200.00**      |
| **3. 85% Accuracy Requirement?**  | RF: **74.71%**, NN: **66.66%** → **Not Met**    |
| **4. Maximum & Average Donation** | Max: **2200.00**, Avg: **112.00**               |
| **5. Donor & Non-Donor Counts**   | Donors: **4,843**, Non-Donors: **14,529**       |
| **6. Dataset Sufficiency**        | **Imbalanced** → Needs more donor data or SMOTE |

---

## 📈 Insights

1. The **dataset is heavily skewed** toward non-donors, requiring oversampling or more data collection.
2. The **highest donation** recorded is **2200.00** with an **average of 112.00**.
3. Both Random Forest and Neural Network **did not meet the 85% accuracy goal**, indicating a need for:

   * More features
   * Hyperparameter tuning
   * Additional data
4. **Top features** influencing generous donation:

   * Socio-economic status (`SES`)
   * Lifetime gift amount
   * Urbanicity
   * Recency of donations

---

## 🚀 How to Run the Project

1. Place the dataset file `donor_data.csv` in the project directory.
2. Run the Python script:

   ```bash
   python donor_ml_model_trainning.py
   ```
3. The program will:

   * Preprocess data
   * Train models
   * Display key insights and metrics

---

## 📌 Final Recommendations

1. **Collect more donor data** to improve class balance.
2. **Feature engineering:** Create new features such as:

   * Donation frequency
   * Time since last donation
   * Income group segmentation
3. Explore **other algorithms**:

   * Gradient Boosting (XGBoost, LightGBM)
   * Logistic Regression for baseline
4. Perform **hyperparameter tuning** using GridSearchCV or Optuna.

---

## 👥 Team Members

| Name          | Role                          |
| ------------- | ----------------------------- |
| Heshan Wijeweera | Data Cleaning & Preprocessing |
| Damayanthi Alahakon    | Random Forest Modeling        |
| Wihangi Sakunika   | Neural Network Development    |
| Chathuri Alwis    | Documentation & Insights      |


