import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

##############################################################################
# 1. LOAD DATA
##############################################################################
# Adjust path as needed
data_path = "comprehensive_mutual_funds_data.csv"
df_original = pd.read_csv(data_path)

print("Initial columns:", df_original.columns.tolist())
print("Initial shape of DataFrame:", df_original.shape)

##############################################################################
# 2. DROP ROWS WHERE ANY TARGET IS MISSING
##############################################################################
target_cols = ["returns_1yr", "returns_3yr", "returns_5yr"]
df = df_original.dropna(subset=target_cols).copy()
print("Shape after dropping missing targets:", df.shape)

##############################################################################
# 3. KEEP scheme_name & fund_manager FOR LATER (OPTIONAL)
##############################################################################
id_cols = ["scheme_name", "fund_manager"]
for col in id_cols:
    if col not in df.columns:
        df[col] = "Unknown"  # Fallback if missing

# We'll drop them from features, but store them in a separate DataFrame for reference
df_ref = df[id_cols].copy()

drop_cols = id_cols + target_cols
X = df.drop(columns=drop_cols, errors='ignore')
y = df[target_cols]  # multi-output target

print("Features shape:", X.shape, "| Targets shape:", y.shape)

##############################################################################
# 4. BASIC DATA CLEANING
##############################################################################
# A) Fill numeric NaNs with median
numeric_cols = X.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)

# B) One-hot encode any remaining categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("Features shape after encoding:", X.shape)

##############################################################################
# 5. TRAIN-TEST SPLIT
#    Random split because we have no time dimension in the data
##############################################################################
X_train, X_test, y_train, y_test, ref_train, ref_test = train_test_split(
    X, y, df_ref,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

##############################################################################
# 6. SET UP A MULTI-OUTPUT RANDOM FOREST + GRIDSEARCHCV FOR HYPERPARAM TUNING
##############################################################################
base_rf = RandomForestRegressor(random_state=42)
multi_rf = MultiOutputRegressor(base_rf)

param_grid = {
    # Because we're in MultiOutputRegressor, we prepend 'estimator__' to refer
    # to the underlying RandomForestRegressor hyperparams:
    "estimator__n_estimators": [100, 200, 300],
    "estimator__max_depth": [5, 10, 15, None],
    "estimator__min_samples_leaf": [1, 2, 5]
}

grid_search = GridSearchCV(
    estimator=multi_rf,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='r2',  # you can also use 'neg_mean_squared_error' or 'neg_mean_absolute_error'
    verbose=1,
    n_jobs=-1
)

print("\nStarting GridSearchCV...")
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)

best_model = grid_search.best_estimator_

##############################################################################
# 7. EVALUATE THE BEST MODEL
##############################################################################
y_pred = best_model.predict(X_test)  # NumPy array shape: (n_samples, 3)

# Convert y_pred to a DataFrame for convenience
y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_test.index)

def print_metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"MAE:        {mae:.4f}")
    print(f"RMSE:       {rmse:.4f}")
    print(f"R^2:        {r2:.4f}")
    print(f"Accuracy:   {r2*100:.2f}%")

for i, col in enumerate(target_cols):
    print_metrics(y_test[col], y_pred_df[col], col)

##############################################################################
# 8. OPTIONAL: MERGE PREDICTIONS WITH SCHEME_NAME & FUND_MANAGER
##############################################################################
test_results = ref_test.copy()             # scheme_name, fund_manager
test_results[target_cols] = y_test         # actual returns
test_results[["pred_1yr", "pred_3yr", "pred_5yr"]] = y_pred_df

# Example: Print top funds by predicted 1-yr return
def print_top_funds(df_in, pred_col, actual_col, top_n=10):
    df_sorted = df_in.sort_values(by=pred_col, ascending=False).head(top_n)
    print(f"\nTop {top_n} by {pred_col}:")
    print(df_sorted[[ "scheme_name", "fund_manager", pred_col, actual_col ]])

print_top_funds(test_results, "pred_1yr", "returns_1yr", top_n=10)

##############################################################################
# 9. OPTIONAL: PLOT ACTUAL vs. PREDICTED SCATTER FOR 1-YEAR
##############################################################################
plt.figure(figsize=(6,6))
plt.scatter(y_test["returns_1yr"], y_pred_df["returns_1yr"], alpha=0.6, edgecolor='k')
plt.plot([y_test["returns_1yr"].min(), y_test["returns_1yr"].max()],
         [y_test["returns_1yr"].min(), y_test["returns_1yr"].max()],
         'r--')
plt.xlabel("Actual 1-Year Return")
plt.ylabel("Predicted 1-Year Return")
plt.title("Actual vs. Predicted (1-Year Return)")
plt.show()

##############################################################################
# 10. OPTIONAL: INSPECT FEATURE IMPORTANCES FOR THE FIRST SUB-MODEL (1Y)
##############################################################################
first_estimator = best_model.estimators_[0]  # random forest for returns_1yr
importances = first_estimator.feature_importances_
features = X_train.columns

feat_imp_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Features (1Y Model):")
print(feat_imp_df.head(10))



