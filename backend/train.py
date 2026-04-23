import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


# 1. LOAD DATA

df = pd.read_csv("../dataset/student_data.csv")

print("Dataset Shape:", df.shape)

#.......................................................
# 2. DATA CLEANING


# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# (This dataset usually has no missing values)
df = df.dropna()

#....................................................
# 3. FEATURE ENGINEERING

# Target is the final grade G3
df["target"] = df["G3"]

# Select only the specified features
selected_features = ["studytime", "failures", "absences", "Medu", "Fedu", "famrel", "goout", "Dalc", "Walc", "health"]
df = df[selected_features + ["target"]]

# ...............................................
# 4. ENCODE CATEGORICAL DATA

# No categorical data in selected features

print("\nUsing selected features:", selected_features)

# ........................................................
# 5. SPLIT DATA


X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# .........................................................
# 6. FEATURE SCALING


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# .........................................................
# 7. TRAIN MULTIPLE MODELS


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor()
}

best_model = None
best_mae = float('inf')
metrics = {}

print("\nModel Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics[name] = {"MAE": mae, "R2": r2}

    print(f"{name} MAE: {mae:.4f}, R2: {r2:.4f}")
    print("-" * 50)

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_model_name = name

# Save feature importances if available
feature_importances = None
if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_

#.............................................
# 8. SAVE MODEL & OBJECTS


pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(list(X.columns), open("columns.pkl", "wb"))
pickle.dump(metrics, open("metrics.pkl", "wb"))
if feature_importances is not None:
    pickle.dump(feature_importances, open("feature_importances.pkl", "wb"))

print(f"\n Best Model: {best_model_name}")
print(f" MAE: {best_mae:.4f}")
print(" Model saved successfully!")