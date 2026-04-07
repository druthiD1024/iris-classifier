import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

iris = load_iris()
X    = iris.data
y    = iris.target

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ─────────────────────────────────────────
# 2. WHY SCALE? — show raw ranges
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("RAW FEATURE RANGES (before scaling)")
print("=" * 50)

for i, name in enumerate(iris.feature_names):
    print(f"  {name:25s} → min={X[:,i].min():.1f}  max={X[:,i].max():.1f}")

print("\n  KNN & SVM are distance-based — bigger ranges dominate!")
print("  Scaling makes all features equally important.")

# ─────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # keep class balance in both splits
)

print(f"Train size : {len(X_train)} samples")
print(f"Test  size : {len(X_test)} samples")
print(f"\nTrain class counts: { {iris.target_names[i]: int(np.sum(y_train==i)) for i in range(3)} }")
print(f"Test  class counts: { {iris.target_names[i]: int(np.sum(y_test==i))  for i in range(3)} }")

# ─────────────────────────────────────────
# 4. FEATURE SCALING — StandardScaler
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE SCALING (StandardScaler)")
print("=" * 50)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train only!
X_test_sc  = scaler.transform(X_test)        # apply same scale to test

print("Scaler fitted on training data only ✅")
print(f"\nMean per feature (should be ~0): {X_train_sc.mean(axis=0).round(2)}")
print(f"Std  per feature (should be ~1): {X_train_sc.std(axis=0).round(2)}")

print("\n" + "=" * 50)
print("SCALED FEATURE RANGES (after scaling)")
print("=" * 50)

for i, name in enumerate(iris.feature_names):
    print(f"  {name:25s} → min={X_train_sc[:,i].min():.2f}  max={X_train_sc[:,i].max():.2f}")

# ─────────────────────────────────────────
# 5. SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Train : {X_train_sc.shape}")
print(f"  Test  : {X_test_sc.shape}")
print(f"  Scaler: StandardScaler (mean=0, std=1)")
print(f"  Split : 80/20 stratified")

print("\n✅ Preprocessing complete — ready for Step 4: Train Models")
