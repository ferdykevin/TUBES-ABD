# ==========================================
# RANDOM FOREST REGRESSION
# Prediksi Kasus DBD
# ==========================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD GOLD DATASET
# ==========================================
df = pd.read_parquet(
    "lakehouse/gold/analytics_ready"
)

print("Jumlah data :", df.shape)
print(df.head())

# ==========================================
# KONVERSI DATA KATEGORIK
# ==========================================
df["city"] = df["city"].map({
    "sj": 0,
    "iq": 1
})

# ==========================================
# FEATURE DAN TARGET
# ==========================================
X = df.drop(
    columns=[
        "total_cases",       # target
        "risk_label",        # mencegah data leakage
        "week_start_date"    # kolom tanggal
    ]
)

y = df["total_cases"]

print("\nJumlah fitur :", X.shape[1])
print("Nama fitur:")
print(X.columns)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

# Training model
rf.fit(X_train, y_train)

# Prediksi
y_pred = rf.predict(X_test)

# ==========================================
# EVALUASI MODEL
# ==========================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== HASIL EVALUASI =====")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 2))

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== TOP 15 FEATURE IMPORTANCE =====")
print(importance_df.head(15))

# ==========================================
# SIMPAN FEATURE IMPORTANCE
# ==========================================
importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

# ==========================================
# SIMPAN HASIL PREDIKSI
# ==========================================
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results.to_csv(
    "models/prediction_results.csv",
    index=False
)

# ==========================================
# SIMPAN MODEL
# ==========================================
joblib.dump(
    rf,
    "models/random_forest_model.pkl"
)

# ==========================================
# VISUALISASI FEATURE IMPORTANCE
# ==========================================
plt.figure(figsize=(10, 6))

top_features = importance_df.head(10)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

# Simpan gambar
plt.savefig(
    "models/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nModel berhasil disimpan!")