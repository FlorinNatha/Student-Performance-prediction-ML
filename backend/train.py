import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


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

# Create PASS/FAIL target
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Drop unnecessary columns
df = df.drop(["G1", "G2", "G3"], axis=1)

# ...............................................
# 4. ENCODE CATEGORICAL DATA


categorical_cols = df.select_dtypes(include=["object"]).columns

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nCategorical columns encoded!")

# ........................................................
# 5. SPLIT DATA


X = df.drop("pass", axis=1)
y = df["pass"]

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
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_accuracy = 0

print("\nModel Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, predictions))
    print("-" * 50)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

#.............................................
# 8. SAVE MODEL & OBJECTS


pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("encoders.pkl", "wb"))

print(f"\n Best Model: {best_model_name}")
print(f" Accuracy: {best_accuracy:.4f}")
print(" Model saved successfully!")