import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ─────────────────────────────────────────
# 1. LOAD & PREPARE
# ─────────────────────────────────────────
print("=" * 50)
print("TRAINING FINAL MODEL")
print("=" * 50)

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 2. TRAIN FINAL SVM
# ─────────────────────────────────────────
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train_sc, y_train)

acc = model.score(X_test_sc, y_test)
print(f"Final SVM accuracy: {acc*100:.1f}% ✅")

# ─────────────────────────────────────────
# 3. SAVE MODEL & SCALER
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SAVING MODEL & SCALER")
print("=" * 50)

joblib.dump(model,  'iris_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')

import os
model_kb  = os.path.getsize('iris_model.pkl')  / 1024
scaler_kb = os.path.getsize('iris_scaler.pkl') / 1024

print(f"iris_model.pkl  saved → {model_kb:.1f} KB")
print(f"iris_scaler.pkl saved → {scaler_kb:.1f} KB")

# ─────────────────────────────────────────
# 4. VERIFY — LOAD BACK & PREDICT
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("VERIFYING — LOADING BACK")
print("=" * 50)

loaded_model  = joblib.load('iris_model.pkl')
loaded_scaler = joblib.load('iris_scaler.pkl')
class_names   = iris.target_names

print("Both files loaded successfully ✅")

test_flowers = [
    [5.1, 3.5, 1.4, 0.2],
    [6.4, 3.2, 4.5, 1.5],
    [6.3, 3.3, 6.0, 2.5],
]

print("\nPredictions from loaded model:")
print("-" * 45)
for flower in test_flowers:
    sc   = loaded_scaler.transform([flower])
    pred = loaded_model.predict(sc)[0]
    prob = loaded_model.predict_proba(sc)[0].max()
    print(f"  {flower} → {class_names[pred]:12s} ({prob*100:.1f}%)")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Model   : iris_model.pkl  ({model_kb:.1f} KB)")
print(f"  Scaler  : iris_scaler.pkl ({scaler_kb:.1f} KB)")
print(f"  Classes : {list(class_names)}")
print(f"  Features: sepal_length, sepal_width, petal_length, petal_width")

print("\n✅ Model saved — ready for Step 6: Flask API")
