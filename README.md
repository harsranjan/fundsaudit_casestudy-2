### README: Mutual Fund Returns Prediction

---

#### **Overview**

This project builds a predictive model for estimating mutual fund returns over multiple time horizons (1 month, 3 months, 6 months, 1 year, 3 years, and 5 years). The objective is to assist investors and fund managers in identifying top-performing funds and understanding the key factors influencing their performance.

The project leverages machine learning techniques, specifically Random Forest Regression, to handle multi-output prediction tasks. Additionally, it provides insights into feature importance and highlights the top mutual funds based on predicted returns.

---


---

#### **Getting Started**

1. **Prerequisites**:
   - Python 3.8+
   - Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
     ```

2. **Dataset**:
   - The dataset `comprehensive_mutual_funds_data.csv` should be placed in the `data/` directory.

3. **Run the Model**:
   - Execute the main script to preprocess data, train the model, and evaluate performance:
     ```bash
     python app.py
     ```

4. **Output**:
   - Performance metrics will be displayed in the console.
   - Top-performing mutual funds and feature importance data.
   - Visualizations, such as actual vs. predicted returns, will be generated .

---

#### **Key Features**

1. **Multi-Output Regression**:
   - Predict returns across multiple time horizons simultaneously.

2. **Hyperparameter Tuning**:
   - Utilizes Grid Search Cross-Validation for optimizing Random Forest hyperparameters.

3. **Top Funds Identification**:
   - Ranks mutual funds based on their predicted average returns across all horizons.

4. **Feature Importance**:
   - Highlights the most significant features influencing fund performance.

5. **Visualization**:
   - Includes scatter plots for actual vs. predicted returns and bar charts for feature importance.

---



#### **Usage**

1. **Preprocessing**:
   - Missing values are handled using median imputation for numerical columns.
   - Categorical features are one-hot encoded.

2. **Model Training**:
   - Random Forest is used within a `MultiOutputRegressor` wrapper for multi-output predictions.
   - Grid Search is performed to identify the best parameters.

3. **Evaluation**:
   - Metrics such as MAE, RMSE, and R² are calculated for each target variable.
   - Visual comparisons between actual and predicted values are provided.

4. **Insights**:
   - A ranked list of top mutual funds based on predicted returns.
   - Feature importance analysis for interpretability.

---

#### **Results**

- **Model Performance**:
  - High accuracy for short-term predictions (1M, 3M, 6M).
  - Reasonable performance for long-term predictions (1Y, 3Y, 5Y).

- **Top Features**:
  - Identified key drivers influencing mutual fund returns.

- **Top Mutual Funds**:
  - Predicted rankings to assist in investment decisions.

---

#### **Contributions**

Contributions to improve the model, add new features, or enhance visualizations are welcome. Feel free to fork the repository and create a pull request.

---
